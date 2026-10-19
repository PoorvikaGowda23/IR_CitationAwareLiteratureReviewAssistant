"""
tune.py
-------
Step 13 (Part 1): Grid search for hybrid graph re-ranking weights:
  ALPHA (embedding), BETA (PageRank), GAMMA (proximity), DELTA (co-citation).

Grid:
  - Each parameter in {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6}
  - Sum constrained to 1.0 (206 total combinations)

Evaluation:
  - Evaluates strictly on the 'val' split using evaluate() from evaluate.py.
  - Never reads or touches the 'test' split.
  - Saves all combinations to results/tuning_val.csv.
  - Prints the best 5 combinations.
  - Does NOT automatically overwrite config.py (waits for user confirmation).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import config
from evaluate import evaluate
from retrieval import retrieve, get_stage1_candidates, get_graph

RESULTS_CSV = config.RESULTS_DIR / "tuning_val.csv"


def generate_weight_grid(step: float = 0.1, max_val: float = 0.6) -> list[tuple[float, float, float, float]]:
    """Generate all (alpha, beta, gamma, delta) combinations summing to 1.0."""
    n_steps = int(round(max_val / step)) + 1
    values = [round(i * step, 1) for i in range(n_steps)]
    combos = [
        (a, b, c, d)
        for a in values
        for b in values
        for c in values
        for d in values
        if round(a + b + c + d, 2) == 1.0
    ]
    return combos


def run_tuning(output_csv: Path | str = RESULTS_CSV) -> pd.DataFrame:
    """Run grid search over the val split and save results to CSV."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    combos = generate_weight_grid(step=0.1, max_val=0.6)
    print(f"\nGenerated {len(combos)} weight combinations summing to 1.0.")
    print("Evaluating exclusively on 'val' split (never reading 'test') ...\n")

    # Warm up caches (graph and stage-1 embeddings for val queries)
    get_graph()

    records: list[dict[str, float]] = []

    # Run grid search
    for idx, (alpha, beta, gamma, delta) in enumerate(combos, start=1):
        w = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
        }

        # Evaluate on VAL split with current weights
        eval_res = evaluate(
            lambda q, k=50: retrieve(
                q,
                k=k,
                weights=w,
                surface_contradictions=False,
            ),
            split="val",
            k=50,
        )

        ndcg10 = float(eval_res["ndcg@10"])
        map_val = float(eval_res["map"])

        records.append({
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "delta": delta,
            "ndcg@10": round(ndcg10, 6),
            "map": round(map_val, 6),
        })

        if idx % 25 == 0 or idx == len(combos):
            print(f"  [{idx:>3}/{len(combos)}] a={alpha:.1f} b={beta:.1f} g={gamma:.1f} d={delta:.1f} -> nDCG@10: {ndcg10:.4f}, MAP: {map_val:.4f}")

    df = pd.DataFrame(records)

    # Sort descending by ndcg@10, then map
    df = df.sort_values(by=["ndcg@10", "map"], ascending=[False, False]).reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved all {len(df)} evaluations -> {output_csv}")

    # Print Top 5
    print("\n" + "=" * 65)
    print("TOP 5 WEIGHT COMBINATIONS ON VALIDATION SPLIT")
    print("=" * 65)
    print(f"{'Rank':<5} {'ALPHA':>6} {'BETA':>6} {'GAMMA':>6} {'DELTA':>6}  {'nDCG@10':>9} {'MAP':>8}")
    print("-" * 65)
    for rank in range(min(5, len(df))):
        row = df.iloc[rank]
        print(
            f"{rank + 1:<5} "
            f"{row['alpha']:>6.1f} {row['beta']:>6.1f} {row['gamma']:>6.1f} {row['delta']:>6.1f}  "
            f"{row['ndcg@10']:>9.4f} {row['map']:>8.4f}"
        )
    print("=" * 65)
    print("Note: Weights have NOT been written to config.py. Awaiting user confirmation.\n")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 13: Grid search tuning on validation split.")
    parser.add_argument("--output-csv", type=Path, default=RESULTS_CSV, help="Output path for CSV.")
    args = parser.parse_args()

    run_tuning(output_csv=args.output_csv)
