"""Command-line validation for configured raw datasets."""

from __future__ import annotations

import argparse
import json

from .registry import DatasetRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", choices=("all", "phm2010", "hannover", "nasa_milling"))
    parser.add_argument("--require-present", action="store_true")
    args = parser.parse_args()

    registry = DatasetRegistry()
    names = registry.names() if args.dataset == "all" else (args.dataset,)
    results = [registry.validate(name, require_present=args.require_present) for name in names]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
