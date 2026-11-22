"""MCP server with one company's supply tools, for the supply analyst.

Tools (all read-only; each returns JSON text)
    get_part(part_id)                  stock on hand, scheduled shipments, products that use it
    get_orders_for_part(part_id)       open orders that need it: units needed, ship day, late penalty
    check_coverage(part_id, arrivals)  arrivals [{qty, day}] replace the part's scheduled
                                       shipments; each order: covered, or units short and
                                       the first day it can ship

The math lives in supply_data.py; this file only exposes it over MCP. Customer
contact addresses are never returned: the analyst has no use for them.

--org names a company folder (five CSV files, prompts/, supplier-emails/); without it,
week-05/brookfield-bikes. The server is named after the folder's first word, e.g.
"brookfield-supply" or "copperline-supply".

Usage (from the top of the repository):
    .venv/bin/python week-05/supply_mcp_server.py                                  # Brookfield
    .venv/bin/python week-05/supply_mcp_server.py --org week-05/copperline-coffee  # Copperline

The server talks MCP over stdin/stdout, so it must never print to stdout:
messages go to stderr.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

import supply_data  # same folder

READ_ONLY = ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True)


class Arrival(TypedDict):
    """One shipment: how many units arrive, and on which day (days from today)."""
    qty: int
    day: int


def log(message):
    print(message, file=sys.stderr, flush=True)


def create_server(org):
    data_dir = org.dir
    server = MCPServer(
        name=f"{org.slug.split('-')[0]}-supply",
        instructions=(
            f"Supply tools for {org.name}. All days are counted from today (day 0). "
            "Use check_coverage for every coverage question: never work coverage out yourself."
        ),
    )

    def run(fn, *args):
        try:
            data = supply_data.SupplyData(data_dir)  # read fresh, so file edits are picked up
            return json.dumps(fn(data, *args), indent=2)
        except supply_data.SupplyDataError as e:
            raise ToolError(str(e)) from None

    @server.tool(annotations=READ_ONLY)
    def get_part(part_id: str) -> str:
        """Get one part's stock and scheduled shipments.

        Returns the description, units on hand today, the supplier, the products that
        use it (used_in: product, and per_unit = units of the part in one product), and
        each scheduled shipment: purchase order number, quantity, and arrives_in_days
        (days from today).

        Args:
            part_id: The part number, as written in the supplier's email.
        """
        return run(lambda d, p: d.get_part(p), part_id)

    @server.tool(annotations=READ_ONLY)
    def get_orders_for_part(part_id: str) -> str:
        """List the open customer orders that need a part, in ship-date order.

        Each order has its order_id, customer, product, qty (units of the product ordered),
        per_unit (units of the part in one product), units_needed (qty x per_unit),
        ship_in_days (days from today), and late_penalty (true if the contract has a
        penalty for shipping late).

        Args:
            part_id: The part number, as written in the supplier's email.
        """
        return run(lambda d, p: d.get_orders_for_part(p), part_id)

    @server.tool(annotations=READ_ONLY)
    def check_coverage(part_id: str, arrivals: list[Arrival]) -> str:
        """Work out which open orders for a part can ship on time, under a new list of arrivals.

        The arrivals REPLACE all of the part's scheduled shipments, so list every shipment
        still expected (get_part shows the current schedule; pass it unchanged to check the
        current plan). Check each option a supplier offers in its own call.

        Stock = units on hand + each arrival from its day. Orders are filled complete, in
        ship-day order, so a short order holds up every order after it. For each order the
        result gives ship_day, covered (true/false), ready_day (the first day it can ship;
        null if the arrivals never cover it), days_late, units_short (units of the part
        missing on its ship day), products_short (how many of the order's products that
        leaves unbuilt), and late_penalty. orders_at_risk lists the order_ids that are not
        covered.

        Args:
            part_id: The part number, as written in the supplier's email.
            arrivals: Every shipment expected, e.g. [{"qty": 120, "day": 5}, {"qty": 280, "day": 26}].
                An empty list means nothing more arrives.
        """
        return run(lambda d, p, a: d.check_coverage(p, [dict(x) for x in a]), part_id, arrivals)

    return server


def main():
    parser = argparse.ArgumentParser(description="A company's supply MCP server (stdio).")
    parser.add_argument("--org", "--data", dest="org", type=Path, default=supply_data.DEFAULT_ORG_DIR,
                        metavar="FOLDER",
                        help="company folder (five CSV files, prompts/, supplier-emails/); "
                             "default: week-05/brookfield-bikes")
    args = parser.parse_args()
    try:
        org = supply_data.Org(args.org)
        data = supply_data.SupplyData(org.dir)  # fail early if a file is missing or malformed
    except supply_data.SupplyDataError as e:
        sys.exit(f"Cannot start: {e}")
    log(f"Company: {org.name}    Data: {org.dir}")
    for line in supply_data.describe_columns(data.columns):
        log(f"  columns: {line}")
    create_server(org).run()  # stdio by default


if __name__ == "__main__":
    main()
