"""
Step 16: ablation study on the TEST split.

Runs evaluate() for each variant, adds 95% bootstrap CIs and a paired bootstrap
against the dense baseline, and writes:
    results/ablation_table.csv, results/ablation_table.md, results/ablation.png

Usage:
    python run_ablations.py --smoke   # VAL split, outputs get a 'smoke_' prefix
    python run_ablations.py           # final run on the TEST split
"""

import argparse
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")  # no display needed; we only save a PNG
import matplotlib.pyplot as plt
import pandas as pd

import config
import schema
from evaluate import bootstrap_ci, evaluate, paired_bootstrap
from graph_builder import annotate_graph, build_graph, load_graph
from retrieval import retrieve
from retrieval_baseline import retrieve_baseline, retrieve_bm25

METRICS = ["ndcg@10", "map", "p@10", "recall@50"]
BASELINE = "Dense baseline"   # every delta and p-value is measured against this row


def build_variants() -> dict[str, Callable]:
    """Create one retrieve function per ablation variant.

    Uses only existing switches: retrieve(weights=, graph=, surface_contradictions=)
    and build_graph(untyped=True).

    Returns:
        Ordered dict {variant name: fn(query, k=50) -> list of results}.
    """
    typed_graph = load_graph(config.DATA_DIR / "graph.gpickle")   # intent-weighted graph
    edges_pred = schema.load_edges_pred(config.DATA_DIR / "edges_pred.parquet")
    untyped_graph = build_graph(edges_pred, untyped=True)         # all edge weights = 1.0
    annotate_graph(untyped_graph, seed=config.SEED)               # PageRank etc., as in sanity_checks

    base_w = {"alpha": config.ALPHA, "beta": config.BETA,
              "gamma": config.GAMMA, "delta": config.DELTA}

    def reranker(weights: dict, graph, surface: bool) -> Callable:
        """Wrap retrieve() with fixed weights / graph / surfacing for evaluate()."""
        return lambda q, k=50: retrieve(q, k=k, weights=weights, graph=graph,
                                        surface_contradictions=surface)

    def without(**zeroed: float) -> dict:
        """Copy of the full weights with the named weights set to 0."""
        return {**base_w, **zeroed}

    return {
        "BM25": retrieve_bm25,
        BASELINE: retrieve_baseline,
        "Full graph re-ranker": reranker(base_w, typed_graph, True),
        "No graph signal (beta=gamma=delta=0)": reranker(without(beta=0, gamma=0, delta=0), typed_graph, True),
        "Untyped edges": reranker(base_w, untyped_graph, True),
        "No PageRank (beta=0)": reranker(without(beta=0), typed_graph, True),
        "No proximity (gamma=0)": reranker(without(gamma=0), typed_graph, True),
        "No co-citation (delta=0)": reranker(without(delta=0), typed_graph, True),
        "No contradiction surfacing": reranker(base_w, typed_graph, False),
    }


def run_ablations(split: str) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Evaluate every variant and build the results table.

    Args:
        split: 'test' for the final run, 'val' for the smoke test.

    Returns:
        (table, ndcg_ci): table with one row per variant; ndcg_ci maps variant
        name to the 95% CI of nDCG@10 (used for the chart error bars).
    """
    results = {name: evaluate(fn, split=split) for name, fn in build_variants().items()}
    base = results[BASELINE]

    rows, ndcg_ci = [], {}
    for name, res in results.items():
        row = {"variant": name}
        for m in METRICS:
            if m in res:                                           # recall@50 may be absent
                lo, hi = bootstrap_ci(res["scores"][m])
                row[m] = f"{res[m]:.4f} [{lo:.3f}, {hi:.3f}]"
                if m == "ndcg@10":
                    ndcg_ci[name] = (lo, hi)
            else:
                row[m] = "n/a"
        row["delta vs baseline (nDCG@10)"] = round(res["ndcg@10"] - base["ndcg@10"], 4)
        if name == BASELINE:
            row["p-value"] = ""                                    # baseline vs itself
        else:
            # paired_bootstrap returns P(variant > baseline); one-sided p = 1 - that
            p_better = paired_bootstrap(res["scores"]["ndcg@10"], base["scores"]["ndcg@10"])
            row["p-value"] = round(1 - p_better, 4)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.attrs["means"] = {n: r["ndcg@10"] for n, r in results.items()}   # for the chart
    return table, ndcg_ci


def write_outputs(table: pd.DataFrame, ndcg_ci: dict, out_dir: Path, prefix: str, split: str) -> None:
    """Write the CSV, a markdown copy, and the nDCG@10 bar chart.

    Args:
        table:   output of run_ablations().
        ndcg_ci: variant -> (low, high) 95% CI of nDCG@10.
        out_dir: results directory.
        prefix:  '' for the real run, 'smoke_' for the smoke test.
        split:   split name, shown in the chart title.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / f"{prefix}ablation_table.csv", index=False)

    header = "| " + " | ".join(table.columns) + " |"
    divider = "|" + "---|" * len(table.columns)
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in table.itertuples(index=False)]
    (out_dir / f"{prefix}ablation_table.md").write_text(
        "\n".join([header, divider, *body]) + "\n", encoding="utf-8")

    # Horizontal bars so the long variant names stay readable; error bars = 95% CI
    names = list(table["variant"])[::-1]                           # first variant on top
    means = [table.attrs["means"][n] for n in names]
    err = [[table.attrs["means"][n] - ndcg_ci[n][0] for n in names],
           [ndcg_ci[n][1] - table.attrs["means"][n] for n in names]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, means, xerr=err, color="#3b6ea8", height=0.6,
            error_kw={"elinewidth": 1.2, "capsize": 3, "ecolor": "#444444"})
    for y, (n, v) in enumerate(zip(names, means)):                 # value label just past the CI whisker
        ax.text(ndcg_ci[n][1] + 0.01, y, f"{v:.3f}", va="center", fontsize=8, color="#222222")
    ax.set_xlabel(f"nDCG@10 ({split} split, 95% bootstrap CI)")
    ax.set_title("Ablation: nDCG@10 by variant")
    ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}ablation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Parse flags, print the test-split warning, run the ablations, write outputs."""
    parser = argparse.ArgumentParser(description="Step 16: ablation study.")
    parser.add_argument("--smoke", action="store_true",
                        help="Use the VAL split and 'smoke_' output names (no test queries).")
    args = parser.parse_args()
    split = "val" if args.smoke else "test"

    if split == "test":
        print("=" * 70)
        print("WARNING: this is the FINAL run. TEST queries are being used.")
        print("Do not change weights or code based on these results.")
        print("=" * 70)

    table, ndcg_ci = run_ablations(split)
    write_outputs(table, ndcg_ci, config.RESULTS_DIR, "smoke_" if args.smoke else "", split)
    print(table.to_string(index=False))
    print(f"\nWrote {config.RESULTS_DIR}/{'smoke_' if args.smoke else ''}ablation_table.csv/.md and ablation.png")


if __name__ == "__main__":
    main()
