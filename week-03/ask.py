"""Ask questions about a folder of documents (RAG).

Documents in a folder (week-03/copperline-docs unless --docs names another) are
split into one chunk per section, stored in a local Chroma database (one per
folder, under week-03/chroma-dbs/), and the 6 most relevant chunks are sent to
Claude together with the question.

Usage (from the top of the repository):
    .venv/bin/python week-03/ask.py "How much can I spend on dinner in Boston?"
    .venv/bin/python week-03/ask.py              # ask several questions in a row
    .venv/bin/python week-03/ask.py --rebuild    # rebuild after adding/removing documents
    .venv/bin/python week-03/ask.py --show-chunks
    .venv/bin/python week-03/ask.py --docs week-03/larkfield-docs --rebuild
    .venv/bin/python week-03/ask.py --docs week-03/larkfield-docs "..." --save week-03/runs/q1.txt
"""

import argparse
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path

import anthropic
import chromadb
import pdfplumber
from bs4 import BeautifulSoup

WEEK_DIR = Path(__file__).resolve().parent
DEFAULT_DOCS_DIR = WEEK_DIR / "copperline-docs"
DBS_DIR = WEEK_DIR / "chroma-dbs"  # one database per document folder (git-ignored)
KEY_FILE = WEEK_DIR.parent.parent / "API-Key.txt"  # folder that contains the repo

COLLECTION = "docs"
MODEL = "claude-sonnet-5"
TOP_K = 6
PREVIEW_CHARS = 150
MAX_CHUNK_WORDS = 400  # longer sections are split at paragraph breaks

SUPPORTED = {".html", ".htm", ".pdf", ".md", ".markdown", ".txt"}

# Extra context added to chunks that would not make sense without it.
# Keyed by file name; the note is added only to chunks that contain `if_mentions`
# and do not already contain `unless_contains`.
CONTEXT_NOTES = {
    "month-end-close-calendar.html": {
        "if_mentions": "business day",
        "unless_contains": "business day 1 is the first business day",
        "note": "(Business days are counted from month end: business day 1 is the first "
                "business day of the new month, and business day minus 1 is the last "
                "business day of the month being closed.)",
    },
}

SYSTEM_PROMPT = """\
You answer questions about an organization's internal documents. \
Use only the sources provided in the user's message.

Rules:
- Answer only with facts stated in the sources. Do not use outside knowledge or make assumptions.
- When a source gives several rules, thresholds, or levels of approval, keep each condition \
with the rule the source attaches it to; do not move a condition from one rule to another. \
Before adding a side note the question did not ask about, check it against the source's \
exact wording, and leave it out if you are not sure it is accurate.
- If the sources give timing rules that could change the answer, such as cutoff times, \
what counts as a business day, or when something is treated as received, state them \
and explain how they affect the answer.
- Do not assume facts about the situation that the question does not state. If a rule \
depends on something the question leaves open, such as a total that could include other \
items, say what the answer depends on instead of deciding it.
- Cite the source file for every fact, in square brackets right after it, \
for example [policy-name.html].
- If the sources do not contain the answer, say clearly that the documents provided \
do not answer the question, instead of guessing. If they answer only part of it, \
answer that part and say what is missing.
- Write plain text for a terminal: short paragraphs or simple "-" lists, no Markdown \
headings, bold, or tables."""


# ---------------------------------------------------------------------------
# Reading documents: each reader returns (title, status, [(heading, [paragraphs])])
# ---------------------------------------------------------------------------

def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def read_html(path):
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


def read_pdf(path):
    lines = []  # (size, text, is_bold)
    with pdfplumber.open(path) as pdf:
        meta_title = (pdf.metadata or {}).get("Title", "")
        for page in pdf.pages:
            rows = {}
            for w in page.extract_words(extra_attrs=["fontname", "size"]):
                rows.setdefault(round(w["top"]), []).append(w)
            for top in sorted(rows):
                words = sorted(rows[top], key=lambda w: w["x0"])
                size = round(max(w["size"] for w in words), 1)
                bold = all("bold" in w["fontname"].lower() for w in words)
                lines.append((size, " ".join(w["text"] for w in words), bold))
    if not lines:
        return path.stem, "", []

    body_size = Counter(size for size, _, _ in lines).most_common(1)[0][0]
    heading_sizes = Counter(size for size, _, bold in lines if size > body_size)
    # Section headings: the most common size above body text. Larger = title/banner.
    section_size = heading_sizes.most_common(1)[0][0] if heading_sizes else None

    title = clean(meta_title.split("|")[0])
    if not title:
        big = [text for size, text, _ in lines if section_size and size > section_size]
        title = big[-1] if big else path.stem

    sections, heading, paras, preamble = [], None, [], []
    for size, text, _ in lines:
        if size < body_size:
            continue  # page headers/footers ("Page 1 of 1") are smaller than body text
        if size == section_size:
            if heading is not None and paras:
                sections.append((heading, [join_wrapped(paras)]))
            heading, paras = text, []
        elif section_size and size > section_size:
            continue  # banner and title lines
        elif heading is None:
            preamble.append(text)
        else:
            paras.append(text)
    if heading is not None and paras:
        sections.append((heading, [join_wrapped(paras)]))
    if not sections and preamble:  # no headings found: treat the whole PDF as one section
        sections, preamble = [("Overview", [join_wrapped(preamble)])], []

    # The status line is the body text right under the title (e.g. "Current version. ...").
    status = preamble[-1] if preamble else ""
    return title, status, sections


def join_wrapped(lines):
    """Join lines that the PDF wrapped mid-sentence back into one paragraph."""
    text = ""
    for line in lines:
        if text.endswith("-") and line[:1].islower():
            text = text[:-1] + line
        else:
            text = f"{text} {line}" if text else line
    return clean(text)


def read_markdown(path):
    title, sections, heading, paras = path.stem, [], "Overview", []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="replace")):
        block = block.strip()
        if not block:
            continue
        if block.startswith("# "):
            title = clean(block[2:].splitlines()[0])
            rest = "\n".join(block.splitlines()[1:]).strip()
            if rest:
                paras.append(clean(rest))
        elif re.match(r"#{2,6} ", block):
            if paras:
                sections.append((heading, paras))
            first, *rest = block.splitlines()
            heading, paras = clean(first.lstrip("#")), [clean(" ".join(rest))] if rest else []
        else:
            paras.append(clean(block) if not block.lstrip().startswith(("-", "*")) else block)
    if paras:
        sections.append((heading, paras))
    return title, "", sections


def read_text(path):
    paras = [clean(p) for p in re.split(r"\n\s*\n", path.read_text(encoding="utf-8", errors="replace"))]
    title = path.stem.replace("-", " ").replace("_", " ").title()
    return title, "", [("Overview", [p for p in paras if p])]


READERS = {".html": read_html, ".htm": read_html, ".pdf": read_pdf,
           ".md": read_markdown, ".markdown": read_markdown, ".txt": read_text}


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

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
    title, status, sections = READERS[path.suffix.lower()](path)
    source = path.relative_to(docs_dir).as_posix()
    context = CONTEXT_NOTES.get(source)
    chunks = []
    for heading, paras in sections:
        parts = split_long(paras)
        for n, part in enumerate(parts, start=1):
            label = heading if len(parts) == 1 else f"{heading} (part {n} of {len(parts)})"
            body = "\n".join(part)
            lowered = body.lower()
            if (context and context["if_mentions"] in lowered
                    and context.get("unless_contains", "\0") not in lowered):
                body = f"{context['note']}\n{body}"
            chunks.append({
                "id": f"{source}#{len(chunks) + 1:02d}",
                "text": f"{title} › {label}\n\n{body}",
                "metadata": {"source": source, "title": title, "section": label, "status": status},
            })
    return chunks


def document_files(docs_dir):
    return sorted(p for p in docs_dir.rglob("*") if p.is_file() and not p.name.startswith("."))


def fingerprint(files, docs_dir):
    """Changes whenever a document is added, removed, or edited."""
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.relative_to(docs_dir)}|{p.stat().st_size}|{p.stat().st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def shown(path):
    """Path relative to the repository when possible, for messages."""
    try:
        return path.relative_to(WEEK_DIR.parent).as_posix()
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# Chroma database
# ---------------------------------------------------------------------------

def db_dir_for(docs_dir):
    """Each document folder gets its own database, e.g. chroma-dbs/larkfield-docs-1a2b3c4d.
    The hash of the full path keeps two folders with the same name apart."""
    tag = hashlib.sha256(str(docs_dir).encode()).hexdigest()[:8]
    return DBS_DIR / f"{docs_dir.name}-{tag}"


def open_db(docs_dir):
    return chromadb.PersistentClient(path=str(db_dir_for(docs_dir)))


def build(client, docs_dir):
    files = document_files(docs_dir)
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(
        COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},  # Chroma's default embedding model
        metadata={"fingerprint": fingerprint(files, docs_dir)},
    )
    print(f"Building the database from {shown(docs_dir)}/ ...")
    total = 0
    for path in files:
        if path.suffix.lower() not in SUPPORTED:
            print(f"  skipped {path.name} (unsupported file type {path.suffix or 'none'})")
            continue
        chunks = make_chunks(path, docs_dir)
        if not chunks:
            print(f"  skipped {path.name} (no text found)")
            continue
        collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
        total += len(chunks)
        print(f"  {path.name}: {len(chunks)} chunks")
    print(f"Done: {total} chunks stored in {shown(db_dir_for(docs_dir))}/\n")
    return collection


def get_collection(client, docs_dir):
    if COLLECTION not in [c.name for c in client.list_collections()]:
        print("No database yet, so building it first.")
        return build(client, docs_dir)
    collection = client.get_collection(COLLECTION)
    files = document_files(docs_dir)
    if (collection.metadata or {}).get("fingerprint") != fingerprint(files, docs_dir):
        print("Note: documents have changed since the database was built. "
              "Run with --rebuild to include the changes.\n")
    return collection


def retrieve(collection, question):
    result = collection.query(query_texts=[question], n_results=TOP_K)
    hits = []
    for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({"text": text, "meta": meta, "similarity": 1 - distance})
    return hits


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------

def load_api_key():
    try:
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        sys.exit(f"API key file not found: {KEY_FILE}")
    if not key:
        sys.exit(f"API key file is empty: {KEY_FILE}")
    return key


def build_prompt(question, hits):
    sources = []
    for rank, hit in enumerate(hits, start=1):
        m = hit["meta"]
        status = f' status="{m["status"]}"' if m.get("status") else ""
        sources.append(f'<source rank="{rank}" file="{m["source"]}"{status}>\n{hit["text"]}\n</source>')
    return "<sources>\n" + "\n".join(sources) + f"\n</sources>\n\nQuestion: {question}"


def ask_claude(client, question, hits):
    """Stream the answer to the terminal and return its full text."""
    print(f"\nAnswer from {MODEL}:\n")
    parts = []

    def note(message):
        print(message)
        parts.append(message)

    try:
        with client.messages.stream(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(question, hits)}],
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                parts.append(text)
            final = stream.get_final_message()
    except anthropic.AuthenticationError:
        sys.exit(f"\nThe API key was rejected. Check the key in {KEY_FILE}.")
    except anthropic.RateLimitError:
        note("\n[Rate limit reached. Wait a moment and try again.]")
        return "".join(parts)
    except anthropic.APIStatusError as e:
        note(f"\n[API error {e.status_code}: {e.message}]")
        return "".join(parts)
    except anthropic.APIConnectionError:
        note("\n[Could not reach the Anthropic API. Check your internet connection.]")
        return "".join(parts)
    print()
    if final.stop_reason == "refusal":
        note("\n[Claude declined to answer this question.]")
    elif final.stop_reason == "max_tokens":
        note("\n[The answer was cut off at the token limit.]")
    return "".join(parts)


# ---------------------------------------------------------------------------

def save_run(path, hits, answer_text):
    """Write the retrieved chunks, a line with only ANSWER, then Claude's answer."""
    lines = []
    for rank, hit in enumerate(hits, start=1):
        lines.append(f"[{rank}] similarity {hit['similarity']:.3f} | source: {hit['meta']['source']}")
        lines.append(hit["text"])
        lines.append("")
    lines.append("ANSWER")
    lines.append(answer_text.strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved this run to {path}")


def answer(collection, claude, question, save_path=None):
    hits = retrieve(collection, question)
    print(f"\nTop {len(hits)} sources:")
    for rank, hit in enumerate(hits, start=1):
        preview = clean(hit["text"])[:PREVIEW_CHARS]
        print(f"  {rank}. similarity {hit['similarity']:.3f} | {hit['meta']['source']}\n     {preview}...")
    answer_text = ask_claude(claude, question, hits)
    if save_path:
        save_run(save_path, hits, answer_text)


def main():
    parser = argparse.ArgumentParser(description="Answer questions from a folder of documents.")
    parser.add_argument("question", nargs="*", help="your question (omit to ask several in a row)")
    parser.add_argument("--docs", type=Path, default=DEFAULT_DOCS_DIR,
                        help=f"folder of documents to answer from (default: {shown(DEFAULT_DOCS_DIR)})")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the database from the documents folder")
    parser.add_argument("--show-chunks", action="store_true", help="print every chunk in the database and exit")
    parser.add_argument("--save", type=Path, metavar="FILE",
                        help="also write the retrieved chunks and the answer to this text file")
    args = parser.parse_args()

    docs_dir = args.docs.resolve()
    if not docs_dir.is_dir():
        sys.exit(f"Document folder not found: {args.docs}")
    if args.save and not args.question:
        sys.exit("--save needs a question on the command line.")

    db = open_db(docs_dir)
    collection = build(db, docs_dir) if args.rebuild else get_collection(db, docs_dir)

    if args.show_chunks:
        data = collection.get()
        for chunk_id, text in sorted(zip(data["ids"], data["documents"])):
            print(f"===== {chunk_id}\n{text}\n")
        return
    if args.rebuild and not args.question:
        return

    claude = anthropic.Anthropic(api_key=load_api_key())
    if args.question:
        answer(collection, claude, " ".join(args.question), args.save)
        return
    print(f"Ask a question about the documents in {shown(docs_dir)}/ "
          "(press Enter on an empty line to quit).")
    while True:
        try:
            question = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        answer(collection, claude, question)


if __name__ == "__main__":
    main()
