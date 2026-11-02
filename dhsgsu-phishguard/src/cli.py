"""
Command-line interface for DHSGSU PhishGuard.

Usage:
    python src/cli.py --interactive          # type an email at the prompt
    python src/cli.py --file email.txt       # analyse a saved raw message
    python src/cli.py --subject "..." --sender "..." --body "..."

Exit codes:
    0  legitimate
    1  phishing
    2  usage or runtime error
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

# --- PATH BOOTSTRAP (idempotent) ---
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[1]
if not (_PROJECT_ROOT / "src").is_dir() or not (_PROJECT_ROOT / "config.py").exists():
    raise RuntimeError(f"Cannot resolve project root from {_HERE}")
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# --- END PATH BOOTSTRAP ---

from explain import ModelNotTrainedError, predict_with_explanation

PROJECT_ROOT = _PROJECT_ROOT

BAND_COLOURS = {
    "Low": "\033[92m",    # green
    "Medium": "\033[93m", # amber
    "High": "\033[91m",   # red
}
RESET = "\033[0m"


def _read_file(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise SystemExit(f"Error: no such file: {path}")
    except OSError as exc:
        raise SystemExit(f"Error: cannot read {path}: {exc}") from exc

    for encoding in ("utf-8-sig", "utf-16", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"Error: unsupported text encoding: {path}")


def _parse_raw(text: str) -> dict[str, Any]:
    """Best-effort extraction of fields from a free-form text file.

    Recognises lines of the form "Subject: ..." / "Sender: ..." / "From: ..."
    and treats the remainder as the body. A file with no headers is analysed
    as a body-only message.
    """
    fields: dict[str, Any] = {"subject": "", "sender_name": "",
                              "sender_email": "", "body": "", "url": ""}
    lines = text.splitlines()
    body_start = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            # A blank line terminates the header block; everything after it
            # is body text, so stop scanning for headers.
            body_start = index + 1
            break
        for key, aliases in (
            ("subject", ["subject", "subj"]),
            ("sender_name", ["sender name", "from name"]),
            ("sender_email", ["from", "sender", "sender email", "e-mail"]),
            ("url", ["url", "link", "website"]),
        ):
            if any(stripped.lower().startswith(a + ":") for a in aliases):
                value = stripped.split(":", 1)[1].strip()
                if key == "sender_email":
                    # Accept both "From: admin@x.in" and
                    # "From: Registrar <admin@x.in>".
                    address = re.search(r"[^@<\s]+@[^>\s]+", value)
                    fields["sender_email"] = address.group(0) if address else value
                    display = re.sub(r"\s*<[^>]+>\s*$", "", value).strip()
                    if display != value:
                        fields["sender_name"] = display
                else:
                    fields[key] = value
                body_start = index + 1
    fields["body"] = "\n".join(lines[body_start:]).strip()
    return fields


def _format_result(result: dict[str, Any]) -> str:
    band = result["risk_band"]
    colour = BAND_COLOURS.get(band, RESET)
    verdict = result["label_name"]
    lines = [
        "",
        f"{'=' * 60}",
        f"  DHSGSU PhishGuard -- Verdict",
        f"{'=' * 60}",
        f"  Verdict      : {colour}{verdict}{RESET}",
        f"  Risk score   : {result['risk_score']:.1f} / 100  "
        f"({band} risk)",
        f"  Confidence   : {result['phishing_probability']:.1%} "
        f"probability of phishing",
        "",
        "  Reasons:",
    ]
    for reason in result["reasons"]:
        lines.append(f"    - {reason}")
    if result["text_tokens"]:
        lines.append("")
        lines.append("  Contributing terms: "
                     + ", ".join(f"'{t}'" for t in result["text_tokens"]))
    lines.append(f"{'=' * 60}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phishguard",
        description="Classify a university email as LEGITIMATE or PHISHING "
                    "and explain the verdict.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--interactive", action="store_true",
                       help="prompt for email fields on the terminal")
    group.add_argument("--file", type=Path, metavar="EMAIL.TXT",
                       help="read a raw email from a text file")
    parser.add_argument("--subject", help="email subject line")
    parser.add_argument("--sender-name", help="display name of the sender")
    parser.add_argument("--sender", "--sender-email", dest="sender_email",
                        help="email address of the sender")
    parser.add_argument("--body", help="email body text")
    parser.add_argument("--url", help="URL contained in the email", default="")
    parser.add_argument(
        "--no-spf", action="store_true",
        help="simulate an SPF authentication failure",
    )
    parser.add_argument(
        "--no-dkim", action="store_true",
        help="simulate a DKIM authentication failure",
    )
    parser.add_argument(
        "--no-dmarc", action="store_true",
        help="simulate a DMARC authentication failure",
    )
    return parser


def _collect_interactive() -> dict[str, Any]:
    print("\nEnter the email details (press Enter to leave a field blank).")
    return {
        "subject": input("Subject: ").strip(),
        "sender_name": input("Sender Name: ").strip(),
        "sender_email": input("Sender Email: ").strip(),
        "body": input("Body: ").strip(),
        "url": input("URL: ").strip(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.interactive:
        email = _collect_interactive()
    elif args.file:
        email = _parse_raw(_read_file(args.file))
    elif args.subject or args.body or args.sender_email:
        email = {
            "subject": args.subject or "",
            "sender_name": args.sender_name or "",
            "sender_email": args.sender_email or "",
            "body": args.body or "",
            "url": args.url or "",
            "spf_pass": not args.no_spf,
            "dkim_pass": not args.no_dkim,
            "dmarc_pass": not args.no_dmarc,
        }
    else:
        parser.print_help()
        return 2

    if not any(email.get(key) for key in
               ("subject", "sender_name", "sender_email", "body", "url")):
        print("Error: no email content supplied. Use --interactive, --file, "
              "or at least --body.", file=sys.stderr)
        return 2
    email.update({
        "spf_pass": not args.no_spf,
        "dkim_pass": not args.no_dkim,
        "dmarc_pass": not args.no_dmarc,
    })

    try:
        result = predict_with_explanation(email)
    except (ModelNotTrainedError, ValueError, TypeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(_format_result(result))
    return 1 if result["label"] == 1 else 0


if __name__ == "__main__":
    sys.exit(main())