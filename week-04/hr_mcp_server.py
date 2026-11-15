"""MCP server for Hartwell Foods HR: policy search and leave balances.

Tools
    search_policy(query)      the 4 HR-document chunks closest to the query, with source files
    get_leave_balance(email)  an employee's vacation, sick, and personal days left (HR staff mode)
    get_leave_balance()       with --as-employee: only that one employee's balance, no inputs

The HR documents (hartwell-hr/hr-docs) are split into one chunk per <h2> section and
stored in a Chroma database in week-04/chroma-db/ (git-ignored). It is built on first
start, rebuilt automatically when the documents change, and can be rebuilt by hand.

Usage (from the top of the repository):
    .venv/bin/python week-04/hr_mcp_server.py                     # HR staff mode (stdio)
    .venv/bin/python week-04/hr_mcp_server.py --as-employee dana.whitfield@hartwellfoods.example
    .venv/bin/python week-04/hr_mcp_server.py --rebuild           # rebuild the database and exit

The server talks MCP over stdin/stdout, so it must never print to stdout:
messages go to stderr.
"""

import argparse
import contextlib
import csv
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

WEEK_DIR = Path(__file__).resolve().parent
DATA_DIR = WEEK_DIR / "hartwell-hr"
DOCS_DIR = DATA_DIR / "hr-docs"
EMPLOYEES_CSV = DATA_DIR / "employees.csv"
DB_DIR = WEEK_DIR / "chroma-db"  # git-ignored

COLLECTION = "hr_docs"
TOP_K = 4
MAX_CHUNK_WORDS = 400  # longer sections are split at paragraph breaks

BALANCE_FIELDS = {
    "vacation_days_left": "Vacation days left",
    "sick_days_left": "Sick days left",
    "personal_days_left": "Personal days left",
}


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


def make_chunks(path):
    title, status, sections = read_html(path)
    source = path.relative_to(DOCS_DIR).as_posix()
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


def document_files():
    return sorted(p for p in DOCS_DIR.rglob("*")
                  if p.is_file() and p.suffix.lower() in {".html", ".htm"})


def fingerprint(files):
    """Changes whenever a document is added, removed, or edited."""
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.relative_to(DOCS_DIR)}|{p.stat().st_size}|{p.stat().st_mtime_ns}".encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Chroma database
# ---------------------------------------------------------------------------

def open_db():
    return chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))


def build(client):
    files = document_files()
    if not files:
        sys.exit(f"No HTML documents found in {DOCS_DIR}")
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(
        COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},  # Chroma's default embedding model
        metadata={"fingerprint": fingerprint(files)},
    )
    log(f"Building the HR policy database from {DOCS_DIR} ...")
    total = 0
    # Anything a library prints while building goes to stderr, never into the MCP stream.
    with contextlib.redirect_stdout(sys.stderr):
        for path in files:
            chunks = make_chunks(path)
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
    log(f"Done: {total} chunks stored in {DB_DIR}")
    return collection


def get_collection(client):
    """Open the collection, building or rebuilding it if missing or out of date."""
    if COLLECTION not in [c.name for c in client.list_collections()]:
        log("No HR policy database yet, so building it.")
        return build(client)
    collection = client.get_collection(COLLECTION)
    if (collection.metadata or {}).get("fingerprint") != fingerprint(document_files()):
        log("The HR documents changed since the database was built, so rebuilding it.")
        return build(client)
    return collection


# ---------------------------------------------------------------------------
# Employee records
# ---------------------------------------------------------------------------

def normalize_email(email):
    return (email or "").strip().casefold()


def find_employee(email):
    """Return the employees.csv row for this email (exact match, any case), or None."""
    wanted = normalize_email(email)
    if not wanted:
        return None
    with EMPLOYEES_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if normalize_email(row.get("email")) == wanted:
                return row
    return None


def balance_text(email):
    row = find_employee(email)
    if row is None:
        raise ToolError(
            f"No employee record matches {email.strip()!r}. Do not state, guess, or estimate "
            "a leave balance for this person. Say that their balance could not be found and "
            "that someone from HR will check it."
        )
    lines = [f"Employee: {row['name']} <{row['email']}>"]
    for field, label in BALANCE_FIELDS.items():
        lines.append(f"{label}: {row[field].strip()}")
    lines.append(f"Source: {EMPLOYEES_CSV.name}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True)


def create_server(collection, as_employee=None):
    server = MCPServer(
        name="hartwell-hr",
        instructions=(
            "Tools for Hartwell Foods HR. Use search_policy for questions about HR policies "
            "and cite the source file of each policy fact. Use get_leave_balance for leave "
            "balances. Only state policies and numbers that these tools return."
        ),
    )

    @server.tool(annotations=READ_ONLY)
    def search_policy(query: str) -> str:
        """Search Hartwell Foods' HR policy documents.

        Returns the 4 document sections closest in meaning to the query. Each one is
        labeled with its source file (cite it, e.g. [leave-policy.html]), its section
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
            return balance_text(as_employee)

        server.add_tool(
            get_leave_balance,
            annotations=READ_ONLY,
            description=(
                "Get the leave balance (vacation, sick, and personal days left) of the employee "
                "this session belongs to. It takes no inputs and can only return that one "
                "employee's balance; balances of other employees are not available."
            ),
        )
    else:
        @server.tool(annotations=READ_ONLY)
        def get_leave_balance(email: str) -> str:
            """Get an employee's leave balance: vacation, sick, and personal days left.

            For HR staff. Looks up the employee by their work email address (exact match,
            any capitalization), e.g. dana.whitfield@hartwellfoods.example.

            Args:
                email: The employee's work email address.
            """
            return balance_text(email)

    return server


def main():
    parser = argparse.ArgumentParser(description="Hartwell Foods HR MCP server (stdio).")
    parser.add_argument("--as-employee", metavar="EMAIL",
                        help="serve only this employee's leave balance (get_leave_balance takes no inputs)")
    parser.add_argument("--rebuild", action="store_true",
                        help="rebuild the HR policy database and exit")
    args = parser.parse_args()

    db = open_db()
    if args.rebuild:
        build(db)
        return

    as_employee = None
    if args.as_employee is not None:
        as_employee = args.as_employee.strip()
        if not as_employee:
            sys.exit("--as-employee needs an email address.")
        if find_employee(as_employee) is None:
            log(f"Warning: no employee record matches {as_employee!r}; "
                "get_leave_balance will return an error.")
        log(f"Employee mode: get_leave_balance returns only {as_employee}'s balance.")

    collection = get_collection(db)
    create_server(collection, as_employee).run()  # stdio by default


if __name__ == "__main__":
    main()
