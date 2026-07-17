"""Merge per-dataset Access aggregates without rerunning training or inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .access_revision_experiments import add_holm, write_outputs
from .experiments import ALL_KEYS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="results/cache_access_revision")
    parser.add_argument("--datasets", nargs="+", default=ALL_KEYS, choices=ALL_KEYS)
    parser.add_argument("--out-json", default="results/access_revision_experiments.json")
    parser.add_argument("--out-md", default="results/access_revision_experiments.md")
    args = parser.parse_args()
    root = Path(args.cache_dir)
    results = {}
    for dataset in args.datasets:
        path = root / dataset / "aggregate.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing completed aggregate: {path}")
        results[dataset] = json.loads(path.read_text(encoding="utf-8"))
    add_holm(results)
    write_outputs(results, args.out_json, args.out_md)
    print(f"Merged {len(results)} dataset aggregates into {args.out_json}")


if __name__ == "__main__":
    main()
