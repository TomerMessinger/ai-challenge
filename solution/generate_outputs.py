"""Run the official challenge briefs through the real pipeline and save each output.

This is challenge QA/output-generation tooling only; proposal behavior lives in
pipeline.proposal_pipeline.

Usage: python solution/generate_outputs.py

Reads data/briefs.json (never modified), calls build_proposal() for every
brief (one OpenAI call per non-blank brief, dates as of 2027-02-01) and writes
solution/outputs/<brief id>.json: the same JSON run_proposal.py prints.
Prints each brief's status and exits 1 if any of them is "error".
"""

import json
import sys
from pathlib import Path

from pipeline.proposal_pipeline import build_proposal

SOLUTION_DIR = Path(__file__).resolve().parent
BRIEFS_PATH = SOLUTION_DIR.parent / "data" / "briefs.json"
OUTPUT_DIR = SOLUTION_DIR / "outputs"


def main(briefs_path: Path = BRIEFS_PATH, output_dir: Path = OUTPUT_DIR) -> int:
    briefs = json.loads(briefs_path.read_text(encoding="utf-8"))["briefs"]
    output_dir.mkdir(exist_ok=True)
    failed = False
    for brief in briefs:
        output = build_proposal(brief["text"])
        path = output_dir / f"{brief['id']}.json"
        path.write_text(output.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"{brief['id']}: {output.status} -> {path.name}")
        failed |= output.status == "error"
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
