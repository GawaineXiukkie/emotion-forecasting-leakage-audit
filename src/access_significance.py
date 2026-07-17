"""Joint Holm correction with the corrected EFC-style variants.

This is a pure aggregation step: it reuses paired seed-0 bootstrap results for
the four core architectures and the isolated corrected-EFC cache.  No model is
trained here.

Run after ``python -m src.access_fixed_efc``:
    python -m src.access_significance
"""
from __future__ import annotations

import json
import os

from .experiments import ALL_KEYS
from .holm_correction import holm_bonferroni

CORE = ["gru", "dialoguernn", "dialoguegcn", "dagerc"]
FIXED = ["pec_fixed", "pseudofuture_fixed"]


def load_core(ds: str, model: str) -> dict:
    if model == "gru":
        with open(f"results/cache/{ds}.json", encoding="utf-8") as f:
            return json.load(f)["paired_vs_transition"]
    with open(f"results/cache_significance/{ds}__{model}.json", encoding="utf-8") as f:
        return json.load(f)


def main():
    results = {}
    for ds in ALL_KEYS:
        for model in CORE:
            results[(ds, model)] = load_core(ds, model)
        path = f"results/cache_access_fixed/{ds}.json"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}; run python -m src.access_fixed_efc")
        with open(path, encoding="utf-8") as f:
            fixed = json.load(f)
        for model in FIXED:
            results[(ds, model)] = fixed[model]["paired_vs_transition"]

    corrected = holm_bonferroni(
        {f"{ds}:{model}": row["p_value"] for (ds, model), row in results.items()}
    )
    rows = []
    lines = [
        "# IEEE Access joint significance tests",
        "",
        "Paired seed-0 dialogue-bootstrap tests against the transition baseline, with one "
        "Holm family across 6 corrected models × 6 dataset-configurations.",
        "",
        "| dataset | model | delta AUC | raw p | rank | Holm threshold | significant |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for key in sorted(corrected, key=lambda k: corrected[k]["rank"]):
        ds, model = key.split(":")
        c, r = corrected[key], results[(ds, model)]
        row = {"dataset": ds, "model": model, **r, **c}
        rows.append(row)
        lines.append(f"| {ds} | {model} | {r['delta_auc']:+.3f} | {c['p']:.4f} | "
                     f"{c['rank']} | {c['holm_threshold']:.4f} | "
                     f"{'yes' if c['significant_holm'] else 'no'} |")
    n_sig = sum(r["significant_holm"] for r in rows)
    lines += ["", f"{n_sig}/{len(rows)} comparisons remain significant after correction."]
    with open("results/access_significance_tests.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open("results/access_significance_tests.json", "w", encoding="utf-8") as f:
        json.dump({"n_significant": n_sig, "n_comparisons": len(rows), "rows": rows}, f, indent=2)
    print(f"{n_sig}/{len(rows)} significant; wrote results/access_significance_tests.{{md,json}}")


if __name__ == "__main__":
    main()
