"""Coordinator 1: a fixed pipeline. Code decides every step; no Claude call coordinates.

For one supplier email, always in this order:
    1. supply analyst  -> brief (brief.json)
    2. customer writer -> one notice per order in the brief's orders_at_risk
                          (skipped if there are none); if the supplier offers options,
                          one notice per order at risk under each option, held until
                          a person chooses
    3. supplier writer -> the reply to the supplier (or NO REPLY NEEDED)

Every draft goes through the review step in specialists.py: reviewer with up to two
revisions, then the final name check in code. A draft is marked for a person if it fails
a check or its order has a late penalty. Drafts are saved in
week-05/drafts/<company folder>/<email>/pipeline-<time>/. Nothing is sent.

The analyst's brief is printed in full, and after each review step the final draft is
printed once (FINAL DRAFT TO SUPPLIER / FINAL DRAFT TO CUSTOMER <order_id>, the draft,
then "Marked for a person: yes" or "no").

--org names the company folder (five CSV files, prompts/, supplier-emails/); without it,
week-05/brookfield-bikes.

At the end it prints a summary: drafts saved, drafts marked for a person, decisions for a
person, and tokens (also saved there as summary.txt).

Usage (from the top of the repository):
    .venv/bin/python week-05/pipeline.py --email week-05/brookfield-bikes/supplier-emails/01-northgate-controller-delay.eml
    .venv/bin/python week-05/pipeline.py --email week-05/brookfield-bikes/supplier-emails/01-northgate-controller-delay.eml --save week-05/runs/pipeline-01.txt
    .venv/bin/python week-05/pipeline.py --org week-05/copperline-coffee --email week-05/copperline-coffee/supplier-emails/01-danforth-pump-delay.eml --save week-05/runs/copperline-01-pipeline.txt
"""

import argparse
import asyncio
import sys
from email.utils import formataddr
from pathlib import Path

import anthropic

import supply_data
from specialists import (NO_REPLY, Job, SpecialistError, Tee, add_org_argument, load_api_key, load_org,
                         parse_email, shown)


async def run_pipeline(email, path, org):
    claude = anthropic.AsyncAnthropic(api_key=load_api_key())
    job = Job(claude, email, path, "pipeline", org)
    stopped = None
    try:
        print("Step 1: supply analyst")
        brief = await job.analyst()

        needed = job.notices_needed()
        if needed and job.option_results():
            by_option = "; ".join(f"{name}: {', '.join(r['orders_at_risk']) or 'none'}"
                                  for name, r in job.option_results())
            print(f"\nStep 2: customer notices, one set per option, held until a person chooses ({by_option})")
            for order_id, option in needed:
                await job.customer_notice(order_id, option)
        elif needed:
            print(f"\nStep 2: customer notices ({len(needed)}: {', '.join(o for o, _ in needed)})")
            for order_id, _ in needed:
                await job.customer_notice(order_id)
        else:
            print("\nStep 2: customer notices: skipped (no orders at risk)")

        print("\nStep 3: reply to the supplier")
        await job.supplier_reply()
    except SpecialistError as e:
        stopped = str(e)
        print(f"\nStopped: {e}")
    summarize(job, stopped)


def summarize(job, stopped=None, finish=None):
    """finish: the orchestrator's (summary, needs_decision), shown first when given."""
    drafts = list(job.notices.values()) + ([job.supplier_draft] if job.supplier_draft else [])
    marked = [d for d in drafts if d["status"] == "NEEDS REVIEW"]
    decisions = job.decisions()
    needs = bool(decisions)  # decided by code: the analyst's decisions plus any schedule approval

    lines = ["", "=" * 72, "Summary", "=" * 72]
    if stopped:
        lines += [f"Stopped early: {stopped}", ""]
    if finish:
        summary, needs_decision = finish
        lines += ["Orchestrator's summary for the purchasing manager:",
                  *[f"  {line}" for line in summary.splitlines()], ""]
    if job.brief is not None or finish:
        line = f"Needs a decision by a person: {'yes' if needs else 'no'}"
        if finish and finish[1] != needs:
            line += (f" (the orchestrator said {'yes' if finish[1] else 'no'}; code decides from the decisions "
                     "listed below)")
        lines += [line, ""]
    lines.append(f"Drafts saved ({len(drafts)}), in {shown(job.out_dir)}/:")
    for d in drafts:
        label = {"APPROVED": "approved", "NEEDS REVIEW": "MARKED FOR A PERSON", NO_REPLY: NO_REPLY}[d["status"]]
        lines.append(f"  {d['path'].name:<30} {label}")
    if not drafts:
        lines.append("  none")

    lines += ["", f"Drafts marked for a person ({len(marked)}):"]
    for d in marked:
        lines.append(f"  {d['path'].name}")
        lines += [f"    - {r}" for r in d["reasons"]]
    if not marked:
        lines.append("  none")

    lines += ["", f"Decisions for a person ({len(decisions)}):"]
    lines += [f"  - {d}" for d in decisions] or ["  none"]

    lines += ["", "Tokens used by each specialist:", job.usage.report()]
    text = "\n".join(lines)
    print(text)
    if job.brief is not None or drafts or finish:
        job._write("summary.txt", text.lstrip("\n") + "\n")


def main(coordinator="pipeline", run=run_pipeline,
         description="Prepare a company's response to one supplier email with a fixed pipeline."):
    """The command line, shared with orchestrator.py. run(email, path, org) handles the email."""
    parser = argparse.ArgumentParser(description=description)
    add_org_argument(parser)
    parser.add_argument("--email", type=Path, required=True, metavar="PATH",
                        help="the supplier email (.eml file)")
    parser.add_argument("--save", type=Path, metavar="FILE",
                        help="also write everything printed to this text file (UTF-8)")
    args = parser.parse_args()

    out = None
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        out = args.save.open("w", encoding="utf-8", buffering=1)
        sys.stdout, sys.stderr = Tee(sys.__stdout__, out), Tee(sys.__stderr__, out)

    code = 0
    try:
        try:
            email = parse_email(args.email.read_bytes())
        except (OSError, ValueError) as e:
            print(f"Could not read the email {args.email}: {e}", file=sys.stderr)
            code = 1
        else:
            org = load_org(args.org)
            print(f"Company: {org.name}    Data: {shown(org.dir)}")
            for line in supply_data.describe_columns(supply_data.SupplyData(org.dir).columns):
                print(f"  columns: {line}")
            print(f"Email: {shown(args.email)}    Coordinator: {coordinator}")
            print(f"From: {formataddr((email['name'], email['address']))}")
            print(f"Subject: {email['subject']}\n")
            asyncio.run(run(email, args.email, org))
    except SystemExit as e:  # e.g. a missing or rejected API key
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            code = 1
        else:
            code = e.code or 0
    except KeyboardInterrupt:
        print("\nStopped by the user (Ctrl+C).", file=sys.stderr)
        code = 130
    finally:
        sys.stdout.flush()
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        if out:
            out.close()
            print(f"Output saved to {shown(args.save)}")
    sys.exit(code)


if __name__ == "__main__":
    main()
