"""Command-line entry point for auditing an authorized local LST20 corpus."""

from __future__ import annotations

import argparse
import json

from .audit import audit_corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and summarize an extracted LST20Corpus directory."
    )
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    result = audit_corpus(args.data_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
