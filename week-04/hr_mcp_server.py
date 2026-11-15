"""MCP server for an organization's HR: policy search and leave balances.

Tools
    search_policy(query)      the 4 HR-document chunks closest to the query, with source files
    get_leave_balance(email)  an employee's leave balances (HR staff mode)
    get_leave_balance()       with --as-employee: only that one employee's balance, no inputs

--org names an organization folder (see org_data.py) holding hr-docs/ and
employees.csv; without it, week-04/hartwell-hr is used. The employee file's
email and balance columns are worked out from its header row.

The HR documents are split into one chunk per <h2> section and stored in a
Chroma database, one per organization, in week-04/chroma-dbs/ (git-ignored). It is
built on first start, rebuilt automatically when the documents change, and can be
rebuilt by hand.

Usage (from the top of the repository):
    .venv/bin/python week-04/hr_mcp_server.py                                   # Hartwell, HR staff mode
    .venv/bin/python week-04/hr_mcp_server.py --org ridgeline-hotels            # Ridgeline
    .venv/bin/python week-04/hr_mcp_server.py --org ridgeline-hotels --as-employee jenna.morales@ridgelinehotels.example
    .venv/bin/python week-04/hr_mcp_server.py --org ridgeline-hotels --rebuild  # rebuild that database and exit

The server talks MCP over stdin/stdout, so it must never print to stdout:
messages go to stderr.
"""

import argparse
import contextlib
import hashlib
import re
import sys
from pathlib import Path

import chromadb
from bs4 import BeautifulSoup
from chromadb.config import Settings
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

import org_data  # same folder: organization folders and employee-file columns

WEEK_DIR = Path(__file__).resolve().parent
DBS_DIR = WEEK_DIR / "chroma-dbs"  # one database per organization (git-ignored)

COLLECTION = "hr_docs"
TOP_K = 4
MAX_CHUNK_WORDS = 400  # longer sections are split at paragraph breaks


def log(message):
    print(message, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Reading and chunking the HR documents (adapted from week-03/ask.py)
# ---------------------------------------------------------------------------

def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def read_html(path):
    """Return (title, status, [(heading, [paragraphs])]) for one HTML document."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "nav", "footer"]):
        tag.decompose()
    root = soup.find("main") or soup.body or soup

    h1 = root.find("h1")
    page_title = soup.title.get_text() if soup.title else ""
    title = clean(h1.get_text()) if h1 else clean(page_title.split("|")[0]) or path.stem
    meta = root.find(class_="meta")
    status = clean(meta.get_text()) if meta else ""
    if meta:
        meta.decompose()

    # Sections start at the highest heading level below the title (usually <h2>).
    levels = [lvl for lvl in ("h2", "h3", "h4") if root.find(lvl)]
    section_tag = levels[0] if levels else None
    blocks = ["p", "li", "pre", "blockquote", "table", "dt", "dd"]

    sections, heading, paras = [], "Overview", []
    for el in root.find_all(["h1", "h2", "h3", "h4", "h5", "h6"] + blocks):
        if any(parent.name in blocks for parent in el.parents):
            continue  # already included as part of an enclosing block
        if el.name == "h1":
            continue
        if el.name == section_tag:
            if paras:
                sections.append((heading, paras))
            heading, paras = clean(el.get_text()), []
        elif el.name == "li":
            paras.append("- " + clean(el.get_text(" ")))
        elif el.name == "table":
            for row in el.find_all("tr"):
                paras.append(" | ".join(clean(c.get_text(" ")) for c in row.find_all(["th", "td"])))
        else:
            text = clean(el.get_text(" "))
            if text:
                paras.append(text)
    if paras:
        sections.append((heading, paras))
    return title, status, sections


def split_long(paras):
    """Group paragraphs into parts of at most MAX_CHUNK_WORDS words."""
    parts, current, words = [], [], 0
    for p in paras:
        n = len(p.split())
        if current and words + n > MAX_CHUNK_WORDS:
            parts.append(current)
            current, words = [], 0
        current.append(p)
        words += n
    if current:
        parts.append(current)
    return parts


def make_chunks(path, docs_dir):
    title, status, sections = read_html(path)
    source = path.relative_to(docs_dir).as_posix()
    chunks = []
    for heading, paras in sections:
        parts = split_long(paras)
        for n, part in enumerate(parts, start=1):
            label = heading if len(parts) == 1 else f"{heading} (part {n} of {len(parts)})"
            chunks.append({
                "id": f"{source}#{len(chunks) + 1:02d}",
                "text": f"{title} › {label}\n\n" + "\n".join(part),
                "metadata": {"source": source, "title": title, "section": label, "status": status},
            })
    return chunks


def document_files(docs_dir):
    return sorted(p for p in docs_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in {".html", ".htm"})


def fingerprint(files, docs_dir):
    """Changes whenever a document is added, removed, or edited."""
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.relative_to(docs_dir)}|{p.stat().st_size}|{p.stat().st_mtime_ns}".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Chroma database (one per organization)
# ---------------------------------------------------------------------------

def db_dir_for(org_dir):
    """e.g. chroma-dbs/ridgeline-hotels-1a2b3c4d; the hash keeps two folders with the same name apart."""
    tag = hashlib.sha256(str(org_dir).encode()).hexdigest()[:8]
    return DBS_DIR / f"{org_dir.name}-{tag}"


def open_db(org_dir):
    return chromadb.PersistentClient(path=str(db_dir_for(org_dir)), settings=Settings(anonymized_telemetry=False))


def build(client, docs_dir, db_dir):
    files = document_files(docs_dir)
    if not files:
        sys.exit(f"No HTML documents found in {docs_dir}")
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(
        COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},  # Chroma's default embedding model
        metadata={"fingerprint": fingerprint(files, docs_dir)},
    )
    log(f"Building the HR policy database from {docs_dir} ...")
    total = 0
    # Anything a library prints while building goes to stderr, never into the MCP stream.
    with contextlib.redirect_stdout(sys.stderr):
        for path in files:
            chunks = make_chunks(path, docs_dir)
            if not chunks:
                log(f"  skipped {path.name} (no text found)")
                continue
            collection.add(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                metadatas=[c["metadata"] for c in chunks],
            )
            total += len(chunks)
            log(f"  {path.name}: {len(chunks)} chunks")
    log(f"Done: {total} chunks stored in {db_dir}")
    return collection


def get_collection(client, docs_dir, db_dir):
    """Open the collection, building or rebuilding it if missing or out of date."""
    if COLLECTION not in [c.name for c in client.list_collections()]:
        log("No HR policy database yet for this organization, so building it.")
        return build(client, docs_dir, db_dir)
    collection = client.get_collection(COLLECTION)
    if (collection.metadata or {}).get("fingerprint") != fingerprint(document_files(docs_dir), docs_dir):
        log("The HR documents changed since the database was built, so rebuilding it.")
        return build(client, docs_dir, db_dir)
    return collection


# ---------------------------------------------------------------------------
# Employee records
# ---------------------------------------------------------------------------

def balance_text(org_dir, email):
    """The employee's balances, re-reading employees.csv so edits show up at once."""
    try:
        table = org_data.employee_table(org_dir)
    except org_data.OrgDataError as e:
        raise ToolError(f"The employee file could not be used: {e}. Do not state a leave balance.") from None
    row = table.find(email)
    if row is None:
        raise ToolError(
            f"No employee record matches {email.strip()!r}. Do not state, guess, or estimate "
            "a leave balance for this person. Say that their balance could not be found and "
            "that someone from HR will check it."
        )
    address = (row.get(table.email_column) or "").strip()
    name = table.name_of(row)
    lines = [f"Employee: {name} <{address}>" if name else f"Employee: {address}"]
    lines += [f"{label}: {value}" for label, value in table.balances_of(row)]
    lines.append(f"Source: {table.path.name}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True)


def create_server(org_dir, collection, as_employee=None):
    server = MCPServer(
        name=f"hr-{org_dir.name}",
        instructions=(
            "HR tools for one organization. Use search_policy for questions about HR policies "
            "and cite the source file of each policy fact. Use get_leave_balance for leave "
            "balances. Only state policies and numbers that these tools return."
        ),
    )

    @server.tool(annotations=READ_ONLY)
    def search_policy(query: str) -> str:
        """Search the organization's HR policy documents.

        Returns the 4 document sections closest in meaning to the query. Each one is
        labeled with its source file (cite it, e.g. [pto-policy.html]), its section
        heading, and the document's version line. The closest match is not always
        relevant, so check that a section actually answers the question.

        Args:
            query: What to look for, in plain words, e.g. "jury duty pay" or
                "carry over unused vacation days".
        """
        query = clean(query or "")
        if not query:
            raise ToolError("The query is empty. Describe the policy topic to search for.")
        result = collection.query(query_texts=[query], n_results=TOP_K)
        sources = []
        for rank, (text, meta, distance) in enumerate(
                zip(result["documents"][0], result["metadatas"][0], result["distances"][0]), start=1):
            status = f' status="{meta["status"]}"' if meta.get("status") else ""
            sources.append(
                f'<source rank="{rank}" file="{meta["source"]}" section="{meta["section"]}"'
                f'{status} similarity="{1 - distance:.3f}">\n{text}\n</source>'
            )
        return "\n".join(sources)

    if as_employee:
        def get_leave_balance() -> str:
            return balance_text(org_dir, as_employee)

        server.add_tool(
            get_leave_balance,
            annotations=READ_ONLY,
            description=(
                "Get the leave balances (such as vacation, PTO, sick, and personal days) of the "
                "employee this session belongs to. It takes no inputs and can only return that one "
                "employee's balance; balances of other employees are not available."
            ),
        )
    else:
        @server.tool(annotations=READ_ONLY)
        def get_leave_balance(email: str) -> str:
            """Get an employee's leave balances (such as vacation, PTO, sick, and personal days).

            For HR staff. Looks up the employee by their work email address (exact match,
            any capitalization).

            Args:
                email: The employee's work email address.
            """
            return balance_text(org_dir, email)

    return server


def main():
    parser = argparse.ArgumentParser(description="HR MCP server for one organization (stdio).")
    parser.add_argument("--org", metavar="FOLDER",
                        help="organization folder (path, or a folder name inside week-04); default: hartwell-hr")
    parser.add_argument("--as-employee", metavar="EMAIL",
                        help="serve only this employee's leave balance (get_leave_balance takes no inputs)")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild this organization's HR policy database and exit")
    args = parser.parse_args()

    try:
        org_dir = org_data.resolve_org(args.org)
        table = org_data.employee_table(org_dir)
    except org_data.OrgDataError as e:
        sys.exit(f"Cannot start: {e}")
    docs_dir, db_dir = org_dir / "hr-docs", db_dir_for(org_dir)
    log(f"Organization: {org_dir.name} ({table.describe()})")

    db = open_db(org_dir)
    if args.rebuild:
        build(db, docs_dir, db_dir)
        return

    as_employee = None
    if args.as_employee is not None:
        as_employee = args.as_employee.strip()
        if not as_employee:
            sys.exit("--as-employee needs an email address.")
        if table.find(as_employee) is None:
            log(f"Warning: no employee record matches {as_employee!r}; "
                "get_leave_balance will return an error.")
        log(f"Employee mode: get_leave_balance returns only {as_employee}'s balance.")

    collection = get_collection(db, docs_dir, db_dir)
    create_server(org_dir, collection, as_employee).run()  # stdio by default


if __name__ == "__main__":
    main()
