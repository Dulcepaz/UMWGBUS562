"""A company's supply data and the coverage math, as plain functions.

A company folder (week-05/brookfield-bikes by default) holds five CSV files, a
prompts/ folder with the specialists' instructions, and supplier-emails/:
    parts.csv              part_id, description, on_hand, supplier
    inbound.csv            po_number, part_id, supplier, qty, arrives_in_days
    bill-of-materials.csv  part_id, plus a product column and a parts-per-unit column
    orders.csv             order_id, customer, contact_email, late_penalty, plus a
                           product column, a quantity column, and a ship-day column
    suppliers.csv          supplier, contact_name, contact_email, part_id

The product, quantity, parts-per-unit, and ship-day columns are named differently by
each company (Brookfield: model, qty, qty_per_bike, ship_in_days; Copperline: product,
units, per_unit, ship_by_day), so find_columns() works out from each header row which
column holds which. Inside the program they are always called product, qty, per_unit,
and ship_in_days.

The files are read fresh on every call. All days are counted from today (day 0).

How coverage is worked out (check_coverage)
    * The arrivals given replace ALL of the part's scheduled shipments.
    * Stock for a part = units on hand today + each arrival, counted from its day.
    * Open orders that need the part are served in ship-date order (earliest ship
      day first; ties by order number). An order ships complete or not at all.
    * An order is ready on the first day the stock reaches all the units needed by
      it and by every order that ships before it.
      So a short order holds up every order after it.
    * An order is covered when its ready day (the first day it can ship) is on or
      before its ship day; otherwise it is at risk.
      units_short = how many of its units are still missing on its ship day.

The MCP server (supply_mcp_server.py) gives these functions to the supply analyst.
Nothing here calls Claude.

Check a company folder without calling Claude (from the top of the repository):
    .venv/bin/python week-05/specialists.py --org week-05/copperline-coffee
"""

import csv
import math
import re
from pathlib import Path

WEEK_DIR = Path(__file__).resolve().parent
DEFAULT_ORG_DIR = WEEK_DIR / "brookfield-bikes"
DATA_DIR = DEFAULT_ORG_DIR  # kept for older code that imports it

CSV_FILES = ["parts.csv", "inbound.csv", "bill-of-materials.csv", "orders.csv", "suppliers.csv"]
PROMPT_FILES = ["analyst.txt", "supplier-writer.txt", "customer-writer.txt", "reviewer.txt", "orchestrator.txt"]


class SupplyDataError(Exception):
    """A problem with the data files or with a tool input, in words a person can act on."""


# ---------------------------------------------------------------------------
# The company folder
# ---------------------------------------------------------------------------

class Org:
    """One company folder: its data, its prompts, and its name.

    The company's name is taken from the "Sign as:" line of prompts/customer-writer.txt
    (e.g. "Sign as: Customer Service, Copperline Coffee Machines"); short is its first
    word ("Copperline"), used in phrases such as "Copperline's customer orders".
    """

    def __init__(self, folder=DEFAULT_ORG_DIR):
        self.dir = resolve_org_dir(folder)
        self.slug = self.dir.name                       # e.g. copperline-coffee
        self.prompts_dir = self.dir / "prompts"
        self.emails_dir = self.dir / "supplier-emails"
        self.name = self._company_name()
        self.short = self.name.split()[0]

    def _company_name(self):
        try:
            text = (self.prompts_dir / "customer-writer.txt").read_text(encoding="utf-8")
        except OSError:
            text = ""
        match = re.search(r"Sign as:\s*(.+)", text)
        if match:
            name = match.group(1).strip().rstrip(".").split(",")[-1].strip()
            if name:
                return name
        return self.slug.replace("-", " ").replace("_", " ").title()

    def prompt(self, name):
        path = self.prompts_dir / name
        try:
            return path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            raise SupplyDataError(f"Prompt file not found: {path}") from None

    def __repr__(self):
        return f"Org({self.name!r}, {self.dir})"


def resolve_org_dir(folder):
    """The company folder as an absolute path, after checking it has the expected layout.

    A bare name such as "copperline-coffee" is also looked up inside week-05/.
    """
    path = Path(folder).expanduser()
    if not path.is_dir() and not path.is_absolute() and (WEEK_DIR / path).is_dir():
        path = WEEK_DIR / path
    path = path.resolve()
    if not path.is_dir():
        raise SupplyDataError(f"Company folder not found: {folder}")

    missing = [n for n in CSV_FILES if not (path / n).is_file()]
    missing += [f"prompts/{n}" for n in PROMPT_FILES if not (path / "prompts" / n).is_file()]
    if missing:
        hint = ""
        inner = [d for d in path.iterdir() if d.is_dir() and (d / "parts.csv").is_file()]
        if inner:
            hint = f" It contains another company folder: did you mean --org {inner[0]}?"
        raise SupplyDataError(f"{path} is not a company folder: missing {', '.join(missing)}.{hint}")
    return path


# ---------------------------------------------------------------------------
# Reading the CSV files, and finding the columns whose names differ by company
# ---------------------------------------------------------------------------

def _read(data_dir, name):
    """(headers, rows) of one CSV file, with headers and values stripped."""
    path = Path(data_dir) / name
    try:
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = [{k.strip(): (v or "").strip() for k, v in row.items() if k} for row in reader]
            headers = [h.strip() for h in (reader.fieldnames or []) if h and h.strip()]
    except FileNotFoundError:
        raise SupplyDataError(f"Data file not found: {path}") from None
    return headers, rows


def _words(header):
    return set(re.findall(r"[a-z]+", header.lower()))


def _is_day(header):
    return any(w.startswith("day") for w in _words(header))


def _all_ints(rows, header):
    return bool(rows) and all(re.fullmatch(r"-?\d+", r.get(header, "")) for r in rows)


PRODUCT_WORDS = {"product", "products", "model", "models", "item", "items", "sku", "machine", "bike",
                 "assembly", "finished", "style", "variant"}
QTY_WORDS = {"qty", "quantity", "units", "unit", "count", "amount", "ordered", "pieces", "pcs", "volume"}
SHIP_WORDS = {"ship", "shipping", "ships", "shipment", "due", "deliver", "delivery", "dispatch", "needed"}


def _pick(headers, rows, file, what, rules):
    """The one header that the first matching rule picks.

    rules is a list of (description, test(header) -> bool), most specific first. A rule
    that matches more than one column is ambiguous, and that is an error: a person must
    rename a column rather than have the program guess.
    """
    for _, test in rules:
        found = [h for h in headers if test(h)]
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            raise SupplyDataError(f"{file}: can't tell which column holds {what}: "
                                  f"{', '.join(found)} all could. Columns: {', '.join(headers)}.")
    raise SupplyDataError(f"{file}: can't tell which column holds {what}. Columns: {', '.join(headers)}.")


def find_columns(data_dir):
    """Which column of bill-of-materials.csv and orders.csv holds the product, the quantity,
    the parts per unit, and the ship day. Returns {file: {role: header}}.
    """
    bom_h, bom_rows = _read(data_dir, "bill-of-materials.csv")
    ord_h, ord_rows = _read(data_dir, "orders.csv")
    for file, headers, needed in (("bill-of-materials.csv", bom_h, ["part_id"]),
                                  ("orders.csv", ord_h, ["order_id", "customer", "contact_email", "late_penalty"])):
        missing = [n for n in needed if n not in headers]
        if missing:
            raise SupplyDataError(f"{file}: missing column(s) {', '.join(missing)}. Columns: {', '.join(headers)}.")

    # bill-of-materials.csv: part_id, the product, and how many of the part one unit takes.
    bom_free = [h for h in bom_h if h != "part_id"]
    per_unit = _pick(bom_free, bom_rows, "bill-of-materials.csv", "the parts per unit", [
        ("a name with 'per'", lambda h: "per" in _words(h)),
        ("a quantity name", lambda h: bool(_words(h) & QTY_WORDS)),
        ("the only whole-number column", lambda h: _all_ints(bom_rows, h)),
    ])
    bom_free.remove(per_unit)
    bom_product = _pick(bom_free, bom_rows, "bill-of-materials.csv", "the product", [
        ("a product name", lambda h: bool(_words(h) & PRODUCT_WORDS)),
        ("the only column left", lambda h: True),
    ])
    products = {r.get(bom_product, "") for r in bom_rows}

    # orders.csv: the product, how many units, and the ship day.
    fixed = {"order_id", "customer", "contact_email", "late_penalty"}
    ord_free = [h for h in ord_h if h not in fixed]
    ship_day = _pick(ord_free, ord_rows, "orders.csv", "the ship day", [
        ("a ship/due name with 'day'", lambda h: bool(_words(h) & SHIP_WORDS) and _is_day(h)),
        ("a name with 'day'", lambda h: _is_day(h)),
        ("a ship/due name", lambda h: bool(_words(h) & SHIP_WORDS)),
    ])
    ord_free.remove(ship_day)
    qty = _pick(ord_free, ord_rows, "orders.csv", "the quantity", [
        ("a quantity name", lambda h: bool(_words(h) & QTY_WORDS) and "per" not in _words(h)),
        ("the only whole-number column", lambda h: _all_ints(ord_rows, h)),
    ])
    ord_free.remove(qty)
    ord_product = _pick(ord_free, ord_rows, "orders.csv", "the product", [
        ("a product name", lambda h: bool(_words(h) & PRODUCT_WORDS)),
        ("values that match the bill of materials",
         lambda h: bool(ord_rows) and all(r.get(h, "") in products for r in ord_rows)),
    ])
    if ord_rows and not any(r.get(ord_product, "") in products for r in ord_rows):
        raise SupplyDataError(f"orders.csv: no value in column {ord_product!r} matches a product in "
                              f"column {bom_product!r} of bill-of-materials.csv.")

    return {
        "bill-of-materials.csv": {"product": bom_product, "per_unit": per_unit},
        "orders.csv": {"product": ord_product, "qty": qty, "ship_in_days": ship_day},
    }


def describe_columns(columns):
    """One line per file, e.g. 'orders.csv: product = product, quantity = units, ship day = ship_by_day'."""
    labels = {"product": "product", "per_unit": "parts per unit", "qty": "quantity", "ship_in_days": "ship day"}
    return [f"{file}: " + ", ".join(f"{labels[role]} = {header}" for role, header in roles.items())
            for file, roles in columns.items()]


def _int(row, field, name):
    try:
        return int(row[field])
    except (KeyError, ValueError):
        raise SupplyDataError(f"{name}: {field} is not a whole number in row {row}") from None


def _need(row, field, name):
    if field not in row:
        raise SupplyDataError(f"{name}: missing column {field!r}.")
    return row[field]


# ---------------------------------------------------------------------------
# The data and the three tools
# ---------------------------------------------------------------------------

class SupplyData:
    """All five files, loaded once. Make a new one to pick up changes to the files."""

    def __init__(self, data_dir=DEFAULT_ORG_DIR):
        self.data_dir = Path(data_dir)
        self.columns = find_columns(data_dir)
        bom_col, ord_col = self.columns["bill-of-materials.csv"], self.columns["orders.csv"]

        self.parts = {}
        for row in _read(data_dir, "parts.csv")[1]:
            pid = _need(row, "part_id", "parts.csv").upper()
            self.parts[pid] = {
                "part_id": pid,
                "description": _need(row, "description", "parts.csv"),
                "on_hand": _int(row, "on_hand", "parts.csv"),
                "supplier": _need(row, "supplier", "parts.csv"),
            }
        self.inbound = [{
            "po_number": _need(row, "po_number", "inbound.csv").upper(),
            "part_id": _need(row, "part_id", "inbound.csv").upper(),
            "supplier": _need(row, "supplier", "inbound.csv"),
            "qty": _int(row, "qty", "inbound.csv"),
            "arrives_in_days": _int(row, "arrives_in_days", "inbound.csv"),
        } for row in _read(data_dir, "inbound.csv")[1]]
        self.bom = [{
            "product": row[bom_col["product"]],
            "part_id": row["part_id"].upper(),
            "per_unit": _int(row, bom_col["per_unit"], "bill-of-materials.csv"),
        } for row in _read(data_dir, "bill-of-materials.csv")[1]]
        self.orders = [{
            "order_id": row["order_id"].upper(),
            "customer": row["customer"],
            "contact_email": row["contact_email"],
            "product": row[ord_col["product"]],
            "qty": _int(row, ord_col["qty"], "orders.csv"),
            "ship_in_days": _int(row, ord_col["ship_in_days"], "orders.csv"),
            "late_penalty": row["late_penalty"].lower() in ("yes", "true", "y", "1"),
        } for row in _read(data_dir, "orders.csv")[1]]
        self.suppliers = _read(data_dir, "suppliers.csv")[1]

    # -- lookups -------------------------------------------------------------

    def part(self, part_id):
        key = str(part_id or "").strip().upper()
        if key not in self.parts:
            known = ", ".join(sorted(self.parts))
            raise SupplyDataError(f"Unknown part_id {part_id!r}. Known parts: {known}.")
        return self.parts[key]

    def shipments_for(self, part_id):
        key = self.part(part_id)["part_id"]
        return sorted((s for s in self.inbound if s["part_id"] == key),
                      key=lambda s: (s["arrives_in_days"], s["po_number"]))

    def per_unit(self, part_id):
        """{product: units of this part in one unit of the product}"""
        key = self.part(part_id)["part_id"]
        return {b["product"]: b["per_unit"] for b in self.bom if b["part_id"] == key}

    def orders_for(self, part_id):
        """Open orders whose product uses the part, in ship-date order, with units of the part needed."""
        per_unit = self.per_unit(part_id)
        rows = []
        for o in self.orders:
            if o["product"] in per_unit:
                rows.append({**o, "per_unit": per_unit[o["product"]],
                             "units_needed": o["qty"] * per_unit[o["product"]]})
        return sorted(rows, key=lambda o: (o["ship_in_days"], o["order_id"]))

    def order(self, order_id):
        key = str(order_id or "").strip().upper()
        for o in self.orders:
            if o["order_id"] == key:
                return o
        raise SupplyDataError(f"Unknown order_id {order_id!r}.")

    def supplier_names(self):
        names = {p["supplier"] for p in self.parts.values()} | {s["supplier"] for s in self.inbound}
        names |= {s["supplier"] for s in self.suppliers if s.get("supplier")}
        return sorted(n for n in names if n)

    def customer_names(self):
        return sorted({o["customer"] for o in self.orders})

    # -- the three tools ------------------------------------------------------

    def get_part(self, part_id):
        p = self.part(part_id)
        return {
            **p,
            "used_in": [{"product": m, "per_unit": q} for m, q in self.per_unit(p["part_id"]).items()],
            "scheduled_shipments": [{k: s[k] for k in ("po_number", "qty", "arrives_in_days")}
                                    for s in self.shipments_for(p["part_id"])],
        }

    def get_orders_for_part(self, part_id):
        p = self.part(part_id)
        return {
            "part_id": p["part_id"],
            "orders": [{k: o[k] for k in ("order_id", "customer", "product", "qty", "per_unit",
                                           "units_needed", "ship_in_days", "late_penalty")}
                       for o in self.orders_for(p["part_id"])],
        }

    def check_coverage(self, part_id, arrivals):
        """Coverage of every open order for the part, under a new list of arrivals.

        arrivals is a list of {"qty": units, "day": days from today}. It REPLACES all of
        the part's scheduled shipments (an empty list means nothing more arrives).
        """
        p = self.part(part_id)
        shipments = sorted(_check_arrivals(arrivals), key=lambda s: s["day"])

        def stock_on(day):
            return p["on_hand"] + sum(s["qty"] for s in shipments if s["day"] <= day)

        # Days on which stock can rise: today, then each arrival day.
        steps = sorted({0} | {s["day"] for s in shipments})
        total_supply = stock_on(steps[-1])

        results, needed_so_far = [], 0
        for o in self.orders_for(p["part_id"]):
            needed_so_far += o["units_needed"]
            ready_day = next((d for d in steps if stock_on(d) >= needed_so_far), None)
            on_ship_day = stock_on(o["ship_in_days"])
            short = min(o["units_needed"], max(0, needed_so_far - on_ship_day))
            at_risk = ready_day is None or ready_day > o["ship_in_days"]
            results.append({
                "order_id": o["order_id"],
                "customer": o["customer"],
                "product": o["product"],
                "qty": o["qty"],
                "units_needed": o["units_needed"],
                "ship_day": o["ship_in_days"],
                "covered": not at_risk,
                "ready_day": ready_day,  # first day it can ship; None: the arrivals never cover it
                "days_late": (None if ready_day is None else max(0, ready_day - o["ship_in_days"])),
                "at_risk": at_risk,
                "units_short": short if at_risk else 0,
                "products_short": math.ceil(short / o["per_unit"]) if at_risk else 0,
                "late_penalty": o["late_penalty"],
            })

        return {
            "part_id": p["part_id"],
            "on_hand": p["on_hand"],
            "arrivals_checked": shipments,
            "replaced_schedule": [{k: s[k] for k in ("po_number", "qty", "arrives_in_days")}
                                  for s in self.shipments_for(p["part_id"])],
            "total_supply": total_supply,
            "total_needed": needed_so_far,
            "orders": results,
            "orders_at_risk": [r["order_id"] for r in results if r["at_risk"]],
        }


def _check_arrivals(arrivals):
    if not isinstance(arrivals, list):
        raise SupplyDataError('arrivals must be a list of {"qty": <units>, "day": <days from today>} '
                              "(an empty list if nothing more arrives).")
    clean = []
    for i, s in enumerate(arrivals, start=1):
        if not isinstance(s, dict) or "qty" not in s or "day" not in s:
            raise SupplyDataError(f"arrivals item {i} needs both qty and day.")
        qty, day = s["qty"], s["day"]
        if isinstance(qty, bool) or isinstance(day, bool) or not all(
                isinstance(v, int) or (isinstance(v, float) and v.is_integer()) for v in (qty, day)):
            raise SupplyDataError(f"arrivals item {i}: qty and day must be whole numbers.")
        qty, day = int(qty), int(day)
        if qty <= 0:
            raise SupplyDataError(f"arrivals item {i}: qty must be more than 0.")
        if day < 0:
            raise SupplyDataError(f"arrivals item {i}: day can't be in the past (below 0).")
        clean.append({"qty": qty, "day": day})
    return clean
