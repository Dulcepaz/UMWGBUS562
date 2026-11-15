"""Organization folders for the week-04 HR agent and MCP server.

An organization folder holds:
    hr-docs/            HTML policy documents
    employees.csv       one row per employee (column names vary by organization)
    sample-emails/      .eml files to test with
    system_prompt.txt   the agent's instructions for this organization

The employee file's columns are worked out from its header row, so each
organization can name them differently (e.g. "email" / "Email Address",
"vacation_days_left" / "PTO Days Remaining").
"""

import csv
import re
from pathlib import Path

WEEK_DIR = Path(__file__).resolve().parent
DEFAULT_ORG = WEEK_DIR / "hartwell-hr"

# Words that mark a column as a leave balance (it must also hold only numbers).
BALANCE_WORDS = re.compile(
    r"\b(day|days|hour|hours|left|remaining|balance|pto|vacation|sick|personal|leave|holiday|floating)\b"
)
# Words that mark a column as being about someone else (a manager's email, a supervisor's name).
OTHER_PERSON_WORDS = re.compile(r"\b(manager|supervisor|boss|lead|director|approver|reports to)\b")


class OrgDataError(Exception):
    """An organization folder or employee file could not be used."""


def resolve_org(value=None):
    """Return the organization folder as an absolute Path.

    value can be a path (absolute or relative to the current folder) or the name of
    a folder inside week-04 (e.g. "ridgeline-hotels"). None means the default, hartwell-hr.
    """
    if value is None:
        org = DEFAULT_ORG
    else:
        candidates = [Path(value).expanduser(), WEEK_DIR / value]
        org = next((c.resolve() for c in candidates if c.is_dir()), None)
        if org is None:
            raise OrgDataError(f"organization folder not found: {value}")
    missing = [name for name in ("hr-docs", "employees.csv") if not (org / name).exists()]
    if missing:
        raise OrgDataError(f"{org} is missing {', '.join(missing)}")
    return org


def _normalize(header):
    return re.sub(r"[^a-z0-9]+", " ", header.lower()).strip()


def _is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def _label(header):
    """Display name for a column: 'vacation_days_left' -> 'Vacation days left'; others unchanged."""
    if header == header.lower() or "_" in header:
        text = re.sub(r"[_\s]+", " ", header).strip()
        return text[:1].upper() + text[1:]
    return header.strip()


class EmployeeTable:
    """employees.csv, with the email, name, and balance columns found from the header row."""

    def __init__(self, path):
        self.path = Path(path)
        try:
            with self.path.open(newline="", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                self.headers = [h for h in (reader.fieldnames or []) if h is not None]
                self.rows = [row for row in reader]
        except OSError as e:
            raise OrgDataError(f"could not read {self.path}: {e}") from None
        if not self.headers:
            raise OrgDataError(f"{self.path} has no header row")
        self.email_column = self._find_email_column()
        self.name_column = self._find_name_column()
        self.balance_columns = self._find_balance_columns()

    def _values(self, column):
        return [(row.get(column) or "").strip() for row in self.rows]

    def _find_email_column(self):
        def own(h):
            return not OTHER_PERSON_WORDS.search(_normalize(h))

        by_header = [h for h in self.headers if re.search(r"\be ?mail\b", _normalize(h))]
        by_values = []
        for h in self.headers:
            values = [v for v in self._values(h) if v]
            if values and sum("@" in v for v in values) / len(values) >= 0.8:
                by_values.append(h)
        for group in ([h for h in by_header if own(h)], [h for h in by_values if own(h)], by_header, by_values):
            if group:
                return group[0]
        raise OrgDataError(f"{self.path.name}: no column holds email addresses (columns: {', '.join(self.headers)})")

    def _find_name_column(self):
        names = [h for h in self.headers
                 if re.search(r"\bname\b", _normalize(h)) and not OTHER_PERSON_WORDS.search(_normalize(h))
                 and h != self.email_column]
        preferred = [h for h in names if re.search(r"\b(employee|team member|staff|full)\b", _normalize(h))
                     or _normalize(h) == "name"]
        return (preferred or names or [None])[0]

    def _find_balance_columns(self):
        found = []
        for h in self.headers:
            if h in (self.email_column, self.name_column):
                continue
            values = [v for v in self._values(h) if v]
            if values and all(_is_number(v) for v in values) and BALANCE_WORDS.search(_normalize(h)):
                found.append(h)
        if not found:
            raise OrgDataError(f"{self.path.name}: no leave-balance columns found (columns: {', '.join(self.headers)})")
        return found

    def describe(self):
        balances = ", ".join(f"{h!r}" for h in self.balance_columns)
        return (f"email column {self.email_column!r}, name column {self.name_column!r}, "
                f"balance columns {balances}")

    def find(self, email):
        """The row for this email (exact match, any capitalization), or None."""
        wanted = (email or "").strip().casefold()
        if not wanted:
            return None
        for row in self.rows:
            if (row.get(self.email_column) or "").strip().casefold() == wanted:
                return row
        return None

    def name_of(self, row):
        return (row.get(self.name_column) or "").strip() if self.name_column else ""

    def balances_of(self, row):
        """[(label, value)] for the row's balance columns."""
        return [(_label(h), (row.get(h) or "").strip()) for h in self.balance_columns]


def employee_table(org_dir):
    return EmployeeTable(Path(org_dir) / "employees.csv")
