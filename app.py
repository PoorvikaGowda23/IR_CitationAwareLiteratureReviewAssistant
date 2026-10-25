"""Streamlit demo UI: citation-aware literature review assistant.

Run from the repo root with:  python -m streamlit run app.py
"""
from __future__ import annotations

import json
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as pgo
import streamlit as st

import config

st.set_page_config(page_title="Citation-Aware Literature Review", layout="wide")

RESULTS = config.RESULTS_DIR
PRESETS = [
    "Session-based recommendation with graph neural networks",  # both reviews cached
    "Graph neural networks for collaborative filtering in recommender systems",
    "Knowledge graph enhanced recommendation",
    "Self-supervised learning for graph-based recommendation",
]
TAG_RE = re.compile(r"\[S2:([^\]]+)\]")
INTENT_CLASSES = ["method", "background", "result"]  # confusion-matrix order


# ── loaders ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_papers() -> pd.DataFrame:
    df = pd.read_parquet(config.DATA_DIR / "papers.parquet")
    df["paper_id"] = df["paper_id"].astype(str)
    return df.set_index("paper_id")


@st.cache_data
def load_json(path: str):
    p = RESULTS / path
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@st.cache_data
def load_csv(path: str):
    p = RESULTS / path
    return pd.read_csv(p) if p.exists() else None


def read_text(path: str) -> str | None:
    p = RESULTS / path
    return p.read_text(encoding="utf-8") if p.exists() else None


@st.cache_data
def load_ground_truth() -> dict[str, dict]:
    p = config.DATA_DIR / "ground_truth.json"
    if not p.exists():
        return {}
    return {g["query_text"].strip().lower(): g for g in json.loads(p.read_text(encoding="utf-8"))}


@st.cache_data(show_spinner=False)
def run_retrieval(query: str, k: int, contradictions: bool):
    from retrieval import retrieve
    from retrieval_baseline import retrieve_baseline, retrieve_bm25

    return (
        retrieve_bm25(query, k=k),
        retrieve_baseline(query, k=k),
        retrieve(query, k=k, surface_contradictions=contradictions),
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def title_of(pid: str) -> str:
    papers = load_papers()
    return str(papers.at[pid, "title"]) if pid in papers.index else pid


def paper_rows(results: list[dict], relevant: set[str] | None = None) -> list[dict]:
    papers = load_papers()
    rows = []
    for r in results:
        pid = str(r["paper_id"])
        row = {
            "#": r["rank"],
            "Title": title_of(pid),
            "Year": int(papers.at[pid, "year"]) if pid in papers.index and pd.notna(papers.at[pid, "year"]) else None,
            "Score": round(float(r["score"]), 3),
            "Flags": ", ".join(map(str, r.get("flags", []))),
        }
        if relevant is not None:
            row["Relevant"] = "✓" if pid in relevant else ""
        for name, val in (r.get("components") or {}).items():
            row[name] = round(float(val), 3)
        rows.append(row)
    return rows


def resolve_tags(text: str) -> str:
    """Replace [S2:<id>] tags with short bracketed titles."""
    def sub(m: re.Match) -> str:
        return f" *[{title_of(m.group(1))[:50]}]*"
    return TAG_RE.sub(sub, text)


def show_review(rev: dict, with_clusters: bool = False) -> None:
    st.markdown(resolve_tags(rev["text"]))
    n_cited = sum(1 for c in rev["claims"] if c["paper_ids"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sentences", len(rev["claims"]))
    m2.metric("With citation", n_cited)
    m3.metric("Invalid tags", rev["n_invalid_tags"])
    m4.metric("From cache", "yes" if rev["from_cache"] else "no")
    if with_clusters and rev.get("clusters"):
        with st.expander("Paper clusters (Louvain communities)"):
            for cid, pids in rev["clusters"].items():
                st.markdown(f"**Cluster {cid}**")
                for p in pids:
                    st.markdown(f"- {title_of(p)}")
    recs = rev.get("paper_records")
    if recs:
        with st.expander("Evidence trace: what the generator saw for each paper"):
            rows = [
                {
                    "Paper": str(v.get("title", pid))[:70],
                    "Year": v.get("year"),
                    "Intent": v.get("intent"),
                    "Relative to": str(v.get("neighbour") or "")[:50],
                    "Context source": v.get("context_source"),
                    "Context": str(v.get("context") or "")[:220],
                }
                for pid, v in recs.items()
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


INTENT_COLORS = {
    "method": "#1f77b4", "background": "#9aa0a6", "result": "#2ca02c",
    "extension": "#ff7f0e", "contradiction": "#d62728",
}
COMM_PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#b07aa1", "#edc948", "#76b7b2",
                "#ff9da7", "#9c755f", "#bab0ac", "#e15759"]


@st.cache_resource
def get_citation_graph():
    from graph_builder import load_graph
    return load_graph()


def graph_html(G, highlight: set[str], height: int = 650) -> str:
    from pyvis.network import Network

    net = Network(height=f"{height}px", width="100%", directed=True, bgcolor="#ffffff")
    for n, d in G.nodes(data=True):
        hit = n in highlight
        net.add_node(
            n,
            label=title_of(n)[:30] if hit else " ",
            title=f"{title_of(n)}<br>PageRank: {d.get('pagerank', 0):.4f}<br>Community: {d.get('community')}",
            size=12 + 400 * d.get("pagerank", 0),
            color={"background": COMM_PALETTE[d.get("community", 0) % len(COMM_PALETTE)],
                   "border": "#000000" if hit else "#ffffff"},
            borderWidth=4 if hit else 1,
        )
    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, color=INTENT_COLORS.get(d.get("intent"), "#9aa0a6"),
                     width=1 + 2 * d.get("weight", 1.0), title=str(d.get("intent")))
    net.toggle_physics(True)
    return net.generate_html()


def confusion_df(d: dict) -> pd.DataFrame:
    labels = d.get("confusion_matrix_labels", INTENT_CLASSES)
    return pd.DataFrame(d["confusion_matrix"], index=[f"true {l}" for l in labels],
                        columns=[f"pred {l}" for l in labels])


def per_class_df(d: dict) -> pd.DataFrame:
    return pd.DataFrame(d["per_class"]).T.round(3)



# ── interactive chart helpers ─────────────────────────────────────────────────

CI_RE = re.compile(r"([\d.]+)\s*\[([\d.]+),\s*([\d.]+)\]")


def parse_ci(cell) -> tuple[float, float, float]:
    m = CI_RE.match(str(cell))
    return (float(m.group(1)), float(m.group(2)), float(m.group(3))) if m else (float("nan"),) * 3


def metric_bar(df: pd.DataFrame, systems: list[str], metrics: list[str], height: int = 380):
    """Grouped bar chart of 'mean [lo, hi]' cells, with CI error bars."""
    fig = pgo.Figure()
    for sysname in systems:
        row = df[df["System"] == sysname].iloc[0]
        vals = [parse_ci(row[m]) for m in metrics]
        fig.add_bar(
            name=sysname, x=metrics, y=[v[0] for v in vals],
            error_y=dict(type="data", symmetric=False,
                         array=[v[2] - v[0] for v in vals], arrayminus=[v[0] - v[1] for v in vals]),
        )
    fig.update_layout(barmode="group", height=height, yaxis_title="score (mean, 95% CI)",
                      margin=dict(t=30, b=10), legend=dict(orientation="h", y=1.1))
    return fig


@st.cache_data(show_spinner=False)
def graph_matrices():
    """Co-citation (A^T A) and bibliographic coupling (A A^T) over the full citation graph."""
    from graph_builder import cocitation_matrix, coupling_matrix
    CG = get_citation_graph()
    nodes = list(CG.nodes())
    return nodes, cocitation_matrix(CG, nodes).toarray(), coupling_matrix(CG, nodes).toarray()


def pair_table(M, nodes: list[str], top: int = 15) -> pd.DataFrame:
    import numpy as np
    U = np.triu(M, k=1)
    idx = np.dstack(np.unravel_index(np.argsort(-U, axis=None)[:top], U.shape))[0]
    return pd.DataFrame([
        {"Paper A": title_of(nodes[i])[:60], "Paper B": title_of(nodes[j])[:60], "Count": int(U[i, j])}
        for i, j in idx if U[i, j] > 0
    ])


# ── sidebar ───────────────────────────────────────────────────────────────────

st.title("Citation-Aware Literature Review Assistant")
st.caption("GNNs for recommender systems: SPECTER2 retrieval, intent-typed citation graph re-ranking, cited LLM review")

with st.sidebar:
    st.header("Query")
    preset = st.selectbox("Example queries", ["(custom)"] + PRESETS)
    query = st.text_area("Query", value=PRESETS[0] if preset == "(custom)" else preset, height=100)
    k = st.slider("Top-k papers", 5, 20, 10)
    contradictions = st.checkbox("Surface contradicting papers", value=True)
    use_cache = st.checkbox("Use cached review", value=True)
    gen_model = st.selectbox(
        "Generator model (Groq)",
        ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        help="Use another model if the daily limit of the default is exhausted. "
             "Non-default models cache to results/review_cache_<model> so the reported results stay untouched.",
    )
    go = st.button("Search", type="primary", width="stretch")

if go:
    st.session_state["query"] = query.strip()
    st.session_state.pop("review", None)
    st.session_state.pop("review_base", None)

q = st.session_state.get("query")
(tab_int, tab_graph, tab_ret, tab_abl, tab_rev, tab_faith) = st.tabs([
    "1. Intent classifiers", "2. Citation graph", "3. Retrieval & re-ranking",
    "4. Tuning & ablation", "5. LLM review", "6. Faithfulness",
])

# ── Live retrieval ────────────────────────────────────────────────────────────
baseline = graph = bm25 = None
if q:
    with st.spinner("Retrieving..."):
        bm25, baseline, graph = run_retrieval(q, k, contradictions)

with tab_ret:
    if not q:
        st.info("Pick a query in the sidebar and press Search.")
    else:
        st.subheader(f"Query: {q}")
        st.caption("Pipeline: BM25 / SPECTER2+FAISS candidates → citation-graph re-rank "
                   "(0.1·embedding + 0.1·PageRank + 0.3·proximity + 0.5·co-citation) → cited LLM review.")
        gt = load_ground_truth().get(q.strip().lower())
        relevant = {str(x) for x in gt["relevant_paper_ids"]} if gt else None
        if relevant is not None:
            hits_by = {
                name: sum(str(r["paper_id"]) in relevant for r in res[:k])
                for name, res in (("BM25", bm25), ("Dense", baseline), ("Hybrid (ours)", graph))
            }
            cols = st.columns(3)
            for col, (name, h) in zip(cols, hits_by.items()):
                col.metric(f"{name}: relevant in top-{k}", f"{h} / {k}")
            st.caption(f"Ground truth: survey reference list for this query ({len(relevant)} relevant papers, "
                       f"{gt['split']} split). ✓ marks a relevant paper.")
        else:
            st.caption("Custom query: no ground truth, so the ✓ relevance column is hidden. "
                       "Pick the first preset to see relevance marks.")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### BM25 (lexical)")
            st.dataframe(pd.DataFrame(paper_rows(bm25, relevant)), hide_index=True, width="stretch")
        with c2:
            st.markdown("#### Dense (SPECTER2 + FAISS)")
            st.dataframe(pd.DataFrame(paper_rows(baseline, relevant)), hide_index=True, width="stretch")
        with c3:
            st.markdown("#### Hybrid: citation-graph re-ranked (ours)")
            st.dataframe(pd.DataFrame(paper_rows(graph, relevant)), hide_index=True, width="stretch")
        st.caption("Re-ranked columns embedding / pagerank / proximity / cocitation are the min-max "
                   "normalised signals before weighting. Flags mark surfaced contradiction papers.")
        base_ids = {str(r["paper_id"]) for r in baseline}
        new = [title_of(str(r["paper_id"])) for r in graph if str(r["paper_id"]) not in base_ids]
        if new:
            st.info("Papers found only by the graph re-ranker (includes surfaced contradictions):\n\n"
                    + "\n".join(f"- {t}" for t in new))

    st.markdown("---")
    st.subheader("Retrieval metrics: BM25 vs Dense vs Hybrid (test split, 5 queries)")
    abl_main = load_csv("ablation_table.csv")
    if abl_main is not None:
        keep = abl_main[abl_main["variant"].isin(["BM25", "Dense baseline", "Full graph re-ranker"])].copy()
        keep["variant"] = keep["variant"].replace({"Full graph re-ranker": "Hybrid (ours)"})
        keep = keep.drop(columns=["delta vs baseline (nDCG@10)"]).rename(columns={"variant": "System"})
        st.dataframe(keep, hide_index=True, width="stretch")
        st.plotly_chart(metric_bar(keep, ["BM25", "Dense baseline", "Hybrid (ours)"],
                                   ["ndcg@10", "map", "p@10", "recall@50"]), width="stretch")
        st.caption("Mean [95% CI]. p-value is a paired test of nDCG@10 against the dense baseline.")

# ── Citation graph ────────────────────────────────────────────────────────────
with tab_graph:

    CG = get_citation_graph()
    hits = {str(r["paper_id"]) for r in graph} & set(CG.nodes) if graph else set()
    st.subheader("How the graph was built")
    import collections
    intents = collections.Counter(d.get("intent", "unknown") for _, _, d in CG.edges(data=True))
    comms = collections.Counter(d.get("community") for _, d in CG.nodes(data=True))
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Papers (nodes)", CG.number_of_nodes())
    g2.metric("Citation edges", CG.number_of_edges())
    g3.metric("Louvain communities", len(comms))
    g4.metric("Largest community", max(comms.values()))
    st.caption("Citation-context sentences from Semantic Scholar → intent classifier (SciBERT + extension / "
               "contradiction head) → typed, weighted edges → PageRank + Louvain communities.")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Edges by citation intent**")
        fig = px.pie(names=list(intents), values=list(intents.values()), hole=0.45,
                     color=list(intents), color_discrete_map=INTENT_COLORS)
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    with t2:
        st.markdown("**Top 10 papers by PageRank**")
        top10 = sorted(CG.nodes, key=lambda n: CG.nodes[n].get("pagerank", 0), reverse=True)[:10]
        st.dataframe(pd.DataFrame({
            "Title": [title_of(n)[:70] for n in top10],
            "PageRank": [round(CG.nodes[n].get("pagerank", 0), 4) for n in top10],
            "Community": [CG.nodes[n].get("community") for n in top10],
        }), hide_index=True, width="stretch")
    st.subheader("Co-citation and bibliographic coupling")
    nodes, CC, BC = graph_matrices()
    pos = {n: i for i, n in enumerate(nodes)}
    if hits:
        sel = [str(r["paper_id"]) for r in graph if str(r["paper_id"]) in pos]
    else:
        sel = sorted(nodes, key=lambda n: CG.nodes[n].get("pagerank", 0), reverse=True)[:40]
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Most co-cited pairs (full graph)**")
        st.dataframe(pair_table(CC, nodes), hide_index=True, width="stretch")
    with p2:
        st.markdown("**Most coupled pairs (full graph)**")
        st.dataframe(pair_table(BC, nodes), hide_index=True, width="stretch")
    focus = st.selectbox("Inspect one paper", sel, format_func=lambda n: title_of(n)[:90], key="focus_paper")
    fi = pos[focus]
    f1, f2 = st.columns(2)
    for col, M, name in ((f1, CC, "co-cited with"), (f2, BC, "coupled with")):
        row = M[fi].copy()
        row[fi] = 0
        top = [t for t in row.argsort()[::-1][:10] if row[t] > 0]
        with col:
            st.markdown(f"**Papers most often {name} this one**")
            if top:
                fig = px.bar(x=[int(row[t]) for t in top], y=[title_of(nodes[t])[:55] for t in top],
                             orientation="h", labels={"x": "count", "y": ""})
                fig.update_layout(height=340, margin=dict(t=10, b=10), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("None in this graph.")
    st.subheader("Interactive citation graph")
    mode = st.radio("View", ["Neighbourhood of query results", "Top papers by PageRank"], horizontal=True,
                    index=0 if hits else 1)
    if mode.startswith("Neighbourhood") and hits:
        nodes = set(hits)
        for h in hits:
            nodes |= set(CG.successors(h)) | set(CG.predecessors(h))
        sub = CG.subgraph(nodes)
    else:
        top_n = st.slider("Number of papers", 20, 150, 60)
        top = sorted(CG.nodes, key=lambda n: CG.nodes[n].get("pagerank", 0), reverse=True)[:top_n]
        sub = CG.subgraph(top)
        if not q:
            st.info("Press Search to highlight the papers retrieved for a query.")
    st.caption(f"Showing {sub.number_of_nodes()} papers and {sub.number_of_edges()} citation edges "
               f"(full graph: {CG.number_of_nodes()} / {CG.number_of_edges()}). Node colour = Louvain "
               "community, size = PageRank, black border = retrieved for the query, arrow = citing → cited.")
    st.markdown(" ".join(f"<span style='color:{c}'>■</span> {i}" for i, c in INTENT_COLORS.items()),
                unsafe_allow_html=True)
    st.iframe(graph_html(sub, hits), height=680)

# ── Reviews ───────────────────────────────────────────────────────────────────
with tab_rev:
    if not q:
        st.info("Pick a query in the sidebar and press Search.")
    else:
        if st.button("Generate both reviews"):
            from generate import generate_review as gen_aware
            from generate_baseline import generate_review as gen_base

            try:
                with st.spinner("Generating reviews (Groq, cached if available)..."):
                    kw = {"k": k, "use_cache": use_cache, "model": gen_model}
                    if gen_model != "openai/gpt-oss-120b":
                        kw["cache_dir"] = RESULTS / f"review_cache_{gen_model.replace('/', '_')}"
                    st.session_state["review_base"] = gen_base(q, baseline, **kw)
                    st.session_state["review"] = gen_aware(q, graph, **kw)
            except Exception as e:  # missing key, network, etc.
                st.error(f"Generation failed: {e}")
        left, right = st.columns(2)
        with left:
            st.markdown("#### Baseline RAG (dense top-k, plain prompt)")
            if st.session_state.get("review_base"):
                show_review(st.session_state["review_base"])
        with right:
            st.markdown("#### Citation-aware (graph re-ranked, clustered, intent-ordered)")
            if st.session_state.get("review"):
                show_review(st.session_state["review"], with_clusters=True)

# ── Intent classifiers ────────────────────────────────────────────────────────
with tab_int:
    tf, sb, fs, ex = (load_json(f"{n}_intent.json") for n in ("tfidf", "scibert", "fewshot", "extended"))
    if tf and sb and fs:
        st.subheader("SciCite: background / method / result")
        rows = []
        for name, d in (("TF-IDF + Logistic Regression", tf), ("SciBERT (fine-tuned)", sb),
                        ("Few-shot LLM (gpt-oss-120b)", fs)):
            pc = d["per_class"]
            rows.append({
                "Classifier": name,
                "Macro F1": round(d["macro_f1"], 3),
                "F1 background": round(pc["background"]["f1-score"], 3),
                "F1 method": round(pc["method"]["f1-score"], 3),
                "F1 result": round(pc["result"]["f1-score"], 3),
                "Result precision / recall": f"{pc['result']['precision']:.3f} / {pc['result']['recall']:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        summary = pd.DataFrame({
            "Model": ["TF-IDF + LogReg", "SciBERT (fine-tuned)", "Few-shot gpt-oss-120b (n=300)"],
            "Macro-F1": [tf["macro_f1"], sb["macro_f1"], fs["macro_f1"]],
        }).set_index("Model")
        fig = px.bar(summary.reset_index(), x="Model", y="Macro-F1", text_auto=".3f", color="Model")
        fig.update_layout(height=340, showlegend=False, margin=dict(t=10, b=10), yaxis_range=[0, 1])
        st.plotly_chart(fig, width="stretch")
        cols = st.columns(3)
        for col, name, d in zip(cols, ("TF-IDF", "SciBERT", "Few-shot LLM"), (tf, sb, fs)):
            with col:
                st.markdown(f"**{name}: per class**")
                st.dataframe(per_class_df(d), width="stretch")
    if ex:
        st.subheader("Extended classes: extension / contradiction / other")
        st.caption(ex.get("NOTE", ""))
        st.dataframe(per_class_df(ex), width="stretch")
        pc = per_class_df(ex).reset_index().rename(columns={"index": "class"})
        pcm = pc.melt(id_vars="class", value_vars=[c for c in ("precision", "recall", "f1-score", "f1") if c in pc.columns])
        st.plotly_chart(px.bar(pcm, x="class", y="value", color="variable", barmode="group", height=320),
                        width="stretch")
        st.metric("Macro-F1 (5-fold CV)", f"{ex['macro_f1']:.3f}")

# ── Tuning & ablation ─────────────────────────────────────────────────────────
with tab_abl:
    st.subheader("Weight tuning on the validation split (3 queries)")
    tune = load_csv("tuning_val.csv")
    if tune is not None:
        st.dataframe(tune.head(10).round(4), hide_index=True, width="stretch")
        fig = px.parallel_coordinates(tune, dimensions=["alpha", "beta", "gamma", "delta", "ndcg@10", "map"],
                                      color="ndcg@10", color_continuous_scale="Viridis")
        fig.update_layout(height=380, margin=dict(t=40, b=10))
        st.caption("Each line is one weight combination from the grid; drag on an axis to filter.")
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            f"**Chosen weights:** alpha={config.ALPHA}, beta={config.BETA}, gamma={config.GAMMA}, "
            f"delta={config.DELTA}. The top grid rows set alpha=0 (no embedding term); we picked a "
            "near-top row that keeps the hybrid. Validation nDCG@10 ≈ 0.44 but test ≈ 0.21, so tuning "
            "on 3 queries overfits."
        )
    st.subheader("Ablation study on the test split (5 queries)")
    abl = load_csv("ablation_table.csv")
    if abl is not None:
        st.dataframe(abl, hide_index=True, width="stretch")
        a2 = abl.rename(columns={"variant": "System"})
        st.plotly_chart(metric_bar(a2, list(a2["System"]), ["ndcg@10", "p@10"], 420), width="stretch")
    img = RESULTS / "ablation.png"
    if img.exists():
        st.image(str(img))

# ── Faithfulness ──────────────────────────────────────────────────────────────
with tab_faith:
    st.subheader("Faithfulness of generated claims (40 claims, 20 per system)")
    jr = load_json("faithfulness/judge_results.json")
    sheet = load_csv("faithfulness/judging_sheet.csv")
    if jr:
        m = jr["metrics"]
        rows = []
        for key, name in (("baseline", "Baseline RAG"), ("citation_aware", "Citation-aware")):
            rows.append({
                "System": name,
                "LLM judge: strict": f"{m[key]['faithfulness_rate']:.0%}",
                "LLM judge: lenient": f"{m[key]['lenient_rate']:.0%}",
                "Supported / partial / unsupported":
                    f"{m[key]['n_supported']} / {m[key]['n_partial']} / {m[key]['n_unsupported']}",
            })
        if sheet is not None and "label" in sheet.columns:
            sys_of = {c["claim_id"]: c["system"] for c in jr["labelled_claims"]}
            sheet = sheet.assign(system=sheet["claim_id"].map(sys_of))
            for r in rows:
                key = "baseline" if r["System"] == "Baseline RAG" else "citation_aware"
                sub = sheet[sheet["system"] == key]["label"]
                if len(sub):
                    r["Manual labels: strict"] = f"{(sub == 'supported').mean():.0%}"
                    r["Manual labels: lenient"] = f"{sub.isin(['supported', 'partially_supported']).mean():.0%}"
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        lab = pd.DataFrame([
            {"System": n, "Label": l, "Claims": m[k][f"n_{l}"]}
            for k, n in (("baseline", "Baseline RAG"), ("citation_aware", "Citation-aware"))
            for l in ("supported", "partial", "unsupported")
        ])
        st.plotly_chart(px.bar(lab, x="System", y="Claims", color="Label", barmode="stack", height=340,
                               color_discrete_map={"supported": "#2ca02c", "partial": "#edc948",
                                                   "unsupported": "#d62728"}), width="stretch")
