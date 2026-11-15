"""Place the sample HR emails into my Gmail sandbox inbox as unread messages.

Each .eml file in week-04/hartwell-hr/sample-emails is added to INBOX with IMAP
APPEND, dated today. Nothing is sent: the messages are written straight into the
mailbox, the same way a mail client saves a draft.

The login is read at run time from Agent-Email.txt in the folder above the
repository, which must contain exactly these two lines:

    EMAIL_ADDRESS=<the sandbox Gmail address>
    APP_PASSWORD=<the 16-letter Gmail app password, no spaces>

Nothing read from that file is ever printed, logged, or saved.

Running it twice on the same day does not add duplicates: each message gets a
Message-ID made from its file name and today's date, and messages already in the
inbox are skipped.

Usage (from the top of the repository):
    .venv/bin/python week-04/load_inbox.py --check-login   # check the login file; does not contact Gmail
    .venv/bin/python week-04/load_inbox.py --dry-run       # list what would be added; does not contact Gmail
    .venv/bin/python week-04/load_inbox.py
"""

import argparse
import imaplib
import re
import sys
import time
from datetime import date
from email import message_from_bytes, policy
from email.utils import formatdate
from pathlib import Path

WEEK_DIR = Path(__file__).resolve().parent
EMAILS_DIR = WEEK_DIR / "hartwell-hr" / "sample-emails"
LOGIN_FILE = WEEK_DIR.parent.parent / "Agent-Email.txt"  # folder that contains the repository

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993
MAILBOX = "INBOX"
MESSAGE_ID_DOMAIN = "hartwell-sample.example"

ADDRESS_KEY = "EMAIL_ADDRESS"
PASSWORD_KEY = "APP_PASSWORD"
EMAIL_SHAPE = re.compile(r"[^\s@]+@[^\s@]+\.[A-Za-z]{2,}")
APP_PASSWORD_SHAPE = re.compile(r"[A-Za-z]{16}")  # Gmail app passwords are 16 letters


class LoginFileError(Exception):
    """The login file could not be used. Messages never include the file's contents."""


# ---------------------------------------------------------------------------
# Login file (values are never printed)
# ---------------------------------------------------------------------------

def check_login_text(text):
    """Check the login file's text against the required format.

    Returns (address, password, checks). Each check is (description, passed, required);
    descriptions refer to line numbers and rules only, never to the values.
    """
    checks = []
    values = {ADDRESS_KEY: [], PASSWORD_KEY: []}
    other_lines = []

    lines = text.lstrip("﻿").splitlines()
    for number, raw in enumerate(lines, start=1):
        line = raw.rstrip("\r")
        if not line.strip():
            continue  # blank lines are allowed
        key, sep, value = line.partition("=")
        if sep and key in values:
            values[key].append((number, value))
        elif sep and key.strip().upper() in values:
            checks.append((f"line {number}: the name must be written exactly "
                           f"{key.strip().upper()}= (no spaces, capital letters)", False, True))
        else:
            other_lines.append(number)

    checks.append(("no lines other than EMAIL_ADDRESS= and APP_PASSWORD= (blank lines are fine)"
                   + (f"; unrecognized line(s): {', '.join(map(str, other_lines))}" if other_lines else ""),
                   not other_lines, True))

    address = password = None
    for key in (ADDRESS_KEY, PASSWORD_KEY):
        found = values[key]
        checks.append((f"{key}= appears exactly once"
                       + (f" (found {len(found)} times)" if len(found) != 1 else ""),
                       len(found) == 1, True))
        if len(found) != 1:
            continue
        number, value = found[0]
        checks.append((f"{key}= value (line {number}) has no spaces before or after it",
                       value == value.strip(), True))
        checks.append((f"{key}= value (line {number}) is not in quotes",
                       not (len(value.strip()) >= 2 and value.strip()[0] in "'\"" and value.strip()[-1] in "'\""),
                       True))
        if key == ADDRESS_KEY:
            address = value
            checks.append(("email address looks like name@domain", bool(EMAIL_SHAPE.fullmatch(value)), True))
        else:
            password = value
            checks.append(("app password contains no spaces", " " not in value and "\t" not in value, True))
            checks.append(("app password is 16 letters, like Gmail app passwords",
                           bool(APP_PASSWORD_SHAPE.fullmatch(value)), False))
    return address, password, checks


def read_login_text():
    try:
        return LOGIN_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise LoginFileError(f"login file not found: {LOGIN_FILE}") from None
    except (OSError, UnicodeDecodeError):
        raise LoginFileError(f"could not read the login file: {LOGIN_FILE}") from None


def read_login():
    """Return (address, password), or raise LoginFileError if the file breaks a required rule."""
    address, password, checks = check_login_text(read_login_text())
    failed = [desc for desc, passed, required in checks if required and not passed]
    if failed:
        raise LoginFileError("the login file is not in the required format "
                             "(run with --check-login for details)")
    return address, password


def check_login():
    """Print a pass/fail report about the login file. Returns the exit code."""
    print(f"Checking {LOGIN_FILE.name} (values are never shown; Gmail is not contacted)")
    try:
        text = read_login_text()
    except LoginFileError as e:
        print(f"  FAIL  {e}")
        return 1
    print("  ok    file found and readable")
    _, _, checks = check_login_text(text)
    for desc, passed, required in checks:
        mark = "ok  " if passed else ("FAIL" if required else "WARN")
        print(f"  {mark}  {desc}")
    if any(required and not passed for _, passed, required in checks):
        print("Result: the file does not match the required format. Fix the FAIL lines above.")
        return 1
    if any(not passed for _, passed, _ in checks):
        print("Result: the format is right, but see the WARN line above.")
    else:
        print("Result: the format is right. (This does not prove Gmail will accept the login.)")
    return 0


def safe_reason(error, *secrets):
    """Gmail's sign-in error text, with the address and password blanked out just in case."""
    if isinstance(error, UnicodeEncodeError):
        return "the login contains characters IMAP cannot send (non-ASCII)"
    text = error.args[0] if error.args else ""
    text = text.decode("utf-8", "replace") if isinstance(text, bytes) else str(text)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[hidden]")
    return text or "(no reason given)"


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def prepare_message(path, today):
    """Return (Message-ID, message bytes) for one .eml file, dated now."""
    msg = message_from_bytes(path.read_bytes(), policy=policy.SMTP)  # SMTP policy: CRLF line endings
    message_id = f"<{path.stem}.{today:%Y%m%d}@{MESSAGE_ID_DOMAIN}>"
    del msg["Date"]
    del msg["Message-ID"]
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = message_id
    return message_id, msg.as_bytes()


def already_in_mailbox(imap, message_id):
    status, data = imap.search(None, "HEADER", "Message-ID", f'"{message_id}"')
    return status == "OK" and bool(data and data[0].split())


def email_files():
    files = sorted(EMAILS_DIR.glob("*.eml"))
    if not files:
        sys.exit(f"No .eml files found in {EMAILS_DIR}")
    return files


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Add the sample HR emails to the Gmail sandbox inbox as unread.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-login", action="store_true",
                      help="check the format of Agent-Email.txt without contacting Gmail")
    mode.add_argument("--dry-run", action="store_true",
                      help="list the messages that would be added, without signing in")
    args = parser.parse_args()

    if args.check_login:
        sys.exit(check_login())

    today = date.today()
    files = email_files()

    if args.dry_run:
        print(f"Dry run: these {len(files)} messages would be added to {MAILBOX}, unread, dated today:")
        for path in files:
            message_id, raw = prepare_message(path, today)
            subject = message_from_bytes(raw, policy=policy.default)["Subject"]
            print(f"  {path.name}  |  {subject}  |  {message_id}")
        return

    try:
        address, password = read_login()
    except LoginFileError as e:
        sys.exit(f"Cannot sign in: {e}.")

    try:
        imap = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    except OSError as e:
        sys.exit(f"Could not connect to {IMAP_HOST}: {e}")

    try:
        try:
            imap.login(address, password)
        except (imaplib.IMAP4.error, UnicodeEncodeError) as e:
            sys.exit("Sign-in failed. Gmail's reason: " + safe_reason(e, address, password) + "\n"
                     "Check Agent-Email.txt: APP_PASSWORD must be a Gmail app password "
                     "(not your normal password), and IMAP must be enabled for the account.")
        finally:
            del password
        print("Signed in to the Gmail sandbox.")

        status, _ = imap.select(MAILBOX)
        if status != "OK":
            sys.exit(f"Could not open the {MAILBOX} mailbox.")

        added = skipped = 0
        for path in files:
            message_id, raw = prepare_message(path, today)
            if already_in_mailbox(imap, message_id):
                print(f"  skipped {path.name} (already in {MAILBOX} today)")
                skipped += 1
                continue
            # No flags: the message arrives unread (no \Seen). Internal date = now.
            status, _ = imap.append(MAILBOX, "", imaplib.Time2Internaldate(time.time()), raw)
            if status != "OK":
                print(f"  FAILED  {path.name}")
                continue
            print(f"  added   {path.name}")
            added += 1
        print(f"Done: {added} added, {skipped} skipped. Nothing was sent.")
    finally:
        try:
            imap.logout()
        except (imaplib.IMAP4.error, OSError):
            pass


if __name__ == "__main__":
    main()
