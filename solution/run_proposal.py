"""Turn one free-text client brief into the proposal JSON, printed to stdout.

Usage: python solution/run_proposal.py "Team dinner for 20 people in Madrid, max €50 each."

stdout is always exactly one ProposalOutput JSON document; the exit code is 1
when its status is "error" and 0 otherwise.
"""

import argparse
import sys

from pipeline.proposal_pipeline import build_proposal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an event proposal JSON from a free-text client brief.")
    parser.add_argument("brief", help="the client's request, in free text")
    args = parser.parse_args(argv)
    output = build_proposal(args.brief)
    print(output.model_dump_json(indent=2))
    return 1 if output.status == "error" else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # valid UTF-8 JSON even when redirected on Windows
    sys.exit(main())
