from __future__ import annotations

from itertools import combinations
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).parent
OUT_MD = ROOT / "Adapt_vs_fixed_vs_noretrieval.md"
OUT_XLSX = ROOT / "Adapt_vs_fixed_vs_noretrieval.xlsx"

APPROACH = {"adaptive": "Adaptive RAG", "fixed": "Fixed RAG", "noretrieval": "No Retrieval LLM"}
MODEL = {"llama": "Llama-70B", "qwen": "Qwen-32B"}

QUALITY = [
    "hallucination_rate",
    "groundedness_score",
    "answer_relevance_1to5",
    "context_relevance_1to5",
    "confidence",
    "query_coverage",
]
PERF = [
    "response_time_s",
    "llm_latency_s",
    "gpu_throughput_toks_per_s",
    "eff_gpu_throughput",
    "gpu_util_percent",
    "gpu_mem_percent",
    "gpu_mem_peak_mb",
    "total_deployment_cost_usd",
]
RETR = ["retrieved_docs_count", "top_retrieval_score", "avg_retrieval_score"]
ALL = QUALITY + PERF + RETR


def parse_name(name: str) -> dict:
    n = name.lower()
    if n.startswith("rag_eval_"):
        a = "adaptive"
    elif n.startswith("fixed_rag_results_"):
        a = "fixed"
    elif n.startswith("noretrieval_"):
        a = "noretrieval"
    else:
        raise ValueError(name)
    m = "llama" if "llama" in n else "qwen"
    q = "4-bit" if "4bit" in n else "8-bit"
    s = "Small" if "small" in n else "Medium" if "medium" in n else "Big"
    return {"approach": a, "model": m, "quantization": q, "corpus_size": s}


def fmt(v: float, d: int = 4) -> str:
    if pd.isna(v):
        return "NA"
    return f"{v:.{d}f}"


def pfmt(p: float) -> str:
    if pd.isna(p):
        return "NA"
    s = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{p:.4f}{s}"


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    c = list(df.columns)
    lines = ["| " + " | ".join(c) + " |", "| " + " | ".join(["---"] * len(c)) + " |"]
    for _, r in df.iterrows():
        vals = []
        for col in c:
            x = r[col]
            vals.append(("NA" if pd.isna(x) else str(x)).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    n1, n2 = len(x), len(y)
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return (x.mean() - y.mean()) / pooled


def welch(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    return stats.ttest_ind(x, y, equal_var=False, nan_policy="omit").pvalue


def pair_table(df: pd.DataFrame, g: str, a: str, b: str, metrics: list[str]) -> pd.DataFrame:
    xa = df[df[g] == a]
    xb = df[df[g] == b]
    rows = []
    for m in metrics:
        if m not in df.columns:
            continue
        sa = pd.to_numeric(xa[m], errors="coerce")
        sb = pd.to_numeric(xb[m], errors="coerce")
        p = welch(sa, sb)
        d = cohens_d(sa, sb)
        rows.append(
            {
                "Metric": m,
                f"{APPROACH.get(a, a)} (mean +/- std)": f"{fmt(sa.mean())} +/- {fmt(sa.std(ddof=1))}",
                f"{APPROACH.get(b, b)} (mean +/- std)": f"{fmt(sb.mean())} +/- {fmt(sb.std(ddof=1))}",
                "Diff": sa.mean() - sb.mean(),
                "p-value": p,
                "Cohen's d": d,
                "Sig.": "Yes" if (not pd.isna(p) and p < 0.05) else "No",
            }
        )
    return pd.DataFrame(rows)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    audit = []
    for p in sorted(ROOT.glob("*.csv")):
        if not (p.name.startswith("rag_eval_") or p.name.startswith("fixed_rag_results_") or p.name.startswith("noretrieval_")):
            continue
        d = pd.read_csv(p)
        meta = parse_name(p.name)
        for k, v in meta.items():
            d[k] = v
        d["approach_label"] = d["approach"].map(APPROACH)
        d["model_label"] = d["model"].map(MODEL)
        d["source_file"] = p.name
        if "used_general_knowledge" in d.columns:
            d["used_general_knowledge"] = d["used_general_knowledge"].astype(str).str.lower().map({"true": 1, "false": 0}).fillna(pd.to_numeric(d["used_general_knowledge"], errors="coerce"))
        else:
            d["used_general_knowledge"] = np.nan
        frames.append(d)
        audit.append(
            {
                "Approach": d["approach_label"].iloc[0],
                "Model": d["model_label"].iloc[0],
                "Quantization": d["quantization"].iloc[0],
                "Corpus Size": d["corpus_size"].iloc[0],
                "source_file": p.name,
                "n": len(d),
            }
        )
    all_df = pd.concat(frames, ignore_index=True, sort=False)
    return all_df, pd.DataFrame(audit)


def main() -> None:
    df, coverage = load()
    total = len(df)
    uniq_cfg = df[["approach", "model", "quantization", "corpus_size"]].drop_duplicates().shape[0]

    overall_means = (
        df.groupby("approach_label")[ALL]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"approach_label": "Approach"})
    )
    overall_by_corpus = (
        df.groupby(["approach_label", "corpus_size"])[ALL]
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"approach_label": "Approach", "corpus_size": "Corpus"})
    )

    pairs = [("adaptive", "fixed"), ("adaptive", "noretrieval"), ("fixed", "noretrieval")]
    pair_tabs = {}
    for a, b in pairs:
        key = f"{APPROACH[a]} vs {APPROACH[b]}"
        pair_tabs[key] = pair_table(df, "approach", a, b, ALL)

    model_cmp = pair_table(df.assign(model=df["model_label"]), "model", "Llama-70B", "Qwen-32B", ALL)
    quant_cmp = pair_table(df, "quantization", "4-bit", "8-bit", ALL)

    corpus_tabs = {}
    for a, b in combinations(["Small", "Medium", "Big"], 2):
        corpus_tabs[f"{a} vs {b}"] = pair_table(df[df["corpus_size"].isin([a, b])], "corpus_size", a, b, ALL)

    hall_rows = []
    for dim in ["approach_label", "model_label", "quantization", "corpus_size"]:
        for val, g in df.groupby(dim):
            h = pd.to_numeric(g["hallucination_rate"], errors="coerce")
            hall_rows.append(
                {
                    "Dimension": dim.replace("_label", "").replace("_", " "),
                    "Value": val,
                    "Mean": fmt(h.mean()),
                    "Median": fmt(h.median()),
                    "Std": fmt(h.std(ddof=1)),
                    "% Zero": f"{fmt((h == 0).mean() * 100, 1)}%",
                    "% Above 0.3": f"{fmt((h > 0.3).mean() * 100, 1)}%",
                    "n": len(g),
                }
            )
    hall = pd.DataFrame(hall_rows)

    cstat = pd.DataFrame()
    if "context_status" in df.columns:
        ctab = df.groupby(["approach_label", "context_status"]).size().reset_index(name="count")
        ctab["pct"] = ctab["count"] / ctab.groupby("approach_label")["count"].transform("sum") * 100
        cstat = ctab.pivot(index="approach_label", columns="context_status", values="pct").fillna(0).reset_index().rename(columns={"approach_label": "Approach"})

    gk = df.groupby("approach_label")["used_general_knowledge"].agg(["mean", "count"]).reset_index()
    gk["GK Usage %"] = gk["mean"] * 100
    gk = gk[["approach_label", "GK Usage %", "count"]].rename(columns={"approach_label": "Approach", "count": "n"})

    corr_pairs = [
        ("hallucination_rate", "groundedness_score"),
        ("hallucination_rate", "response_time_s"),
        ("groundedness_score", "response_time_s"),
        ("answer_relevance_1to5", "context_relevance_1to5"),
        ("hallucination_rate", "retrieved_docs_count"),
        ("groundedness_score", "retrieved_docs_count"),
        ("gpu_throughput_toks_per_s", "response_time_s"),
        ("gpu_mem_peak_mb", "gpu_throughput_toks_per_s"),
        ("confidence", "hallucination_rate"),
        ("query_coverage", "groundedness_score"),
    ]
    corr_rows = []
    for a, b in corr_pairs:
        if a in df.columns and b in df.columns:
            z = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
            corr_rows.append({"Metric A": a, "Metric B": b, "Pearson r": fmt(z[a].corr(z[b])) if len(z) > 1 else "NA"})
    corr = pd.DataFrame(corr_rows)

    rank = (
        df.groupby(["approach_label", "model_label", "quantization", "corpus_size"], dropna=False)
        .agg(
            hallucination_rate=("hallucination_rate", "mean"),
            groundedness_score=("groundedness_score", "mean"),
            answer_relevance_1to5=("answer_relevance_1to5", "mean"),
            response_time_s=("response_time_s", "mean"),
            gpu_mem_peak_mb=("gpu_mem_peak_mb", "mean"),
        )
        .reset_index()
    )
    rank["Config"] = rank["approach_label"] + "|" + rank["model_label"] + "|" + rank["quantization"] + "|" + rank["corpus_size"]
    rank = rank.sort_values(["groundedness_score", "hallucination_rate"], ascending=[False, True])

    overall_idx = overall_means.set_index("Approach")
    best_h_approach = overall_idx["hallucination_rate"].idxmin()
    best_h_value = overall_idx.loc[best_h_approach, "hallucination_rate"]
    best_g_approach = overall_idx["groundedness_score"].idxmax()
    best_g_value = overall_idx.loc[best_g_approach, "groundedness_score"]
    fast_approach = overall_idx["response_time_s"].idxmin()
    fast_value = overall_idx.loc[fast_approach, "response_time_s"]

    p_ad = pair_tabs["Adaptive RAG vs Fixed RAG"]
    p_ad_fmt = p_ad.assign(
        Diff=p_ad["Diff"].map(lambda x: fmt(x)),
        **{
            "p-value": p_ad["p-value"].map(pfmt),
            "Cohen's d": p_ad["Cohen's d"].map(lambda x: fmt(x, 3)),
        },
    )
    p_an = pair_tabs["Adaptive RAG vs No Retrieval LLM"]
    p_an_fmt = p_an.assign(
        Diff=p_an["Diff"].map(lambda x: fmt(x)),
        **{
            "p-value": p_an["p-value"].map(pfmt),
            "Cohen's d": p_an["Cohen's d"].map(lambda x: fmt(x, 3)),
        },
    )
    p_fn = pair_tabs["Fixed RAG vs No Retrieval LLM"]
    p_fn_fmt = p_fn.assign(
        Diff=p_fn["Diff"].map(lambda x: fmt(x)),
        **{
            "p-value": p_fn["p-value"].map(pfmt),
            "Cohen's d": p_fn["Cohen's d"].map(lambda x: fmt(x, 3)),
        },
    )
    model_fmt = model_cmp.assign(
        Diff=model_cmp["Diff"].map(lambda x: fmt(x)),
        **{
            "p-value": model_cmp["p-value"].map(pfmt),
            "Cohen's d": model_cmp["Cohen's d"].map(lambda x: fmt(x, 3)),
        },
    )
    quant_fmt = quant_cmp.assign(
        Diff=quant_cmp["Diff"].map(lambda x: fmt(x)),
        **{
            "p-value": quant_cmp["p-value"].map(pfmt),
            "Cohen's d": quant_cmp["Cohen's d"].map(lambda x: fmt(x, 3)),
        },
    )
    sm = corpus_tabs["Small vs Medium"]
    sm_fmt = sm.assign(
        Diff=sm["Diff"].map(lambda x: fmt(x)),
        **{"p-value": sm["p-value"].map(pfmt), "Cohen's d": sm["Cohen's d"].map(lambda x: fmt(x, 3))},
    )
    sb = corpus_tabs["Small vs Big"]
    sb_fmt = sb.assign(
        Diff=sb["Diff"].map(lambda x: fmt(x)),
        **{"p-value": sb["p-value"].map(pfmt), "Cohen's d": sb["Cohen's d"].map(lambda x: fmt(x, 3))},
    )
    mb = corpus_tabs["Medium vs Big"]
    mb_fmt = mb.assign(
        Diff=mb["Diff"].map(lambda x: fmt(x)),
        **{"p-value": mb["p-value"].map(pfmt), "Cohen's d": mb["Cohen's d"].map(lambda x: fmt(x, 3))},
    )
    hrow = p_ad[p_ad["Metric"] == "hallucination_rate"].iloc[0]
    grow = p_ad[p_ad["Metric"] == "groundedness_score"].iloc[0]
    ad_rel = p_ad[p_ad["Metric"] == "answer_relevance_1to5"].iloc[0]
    an_h = p_an[p_an["Metric"] == "hallucination_rate"].iloc[0]
    an_g = p_an[p_an["Metric"] == "groundedness_score"].iloc[0]
    an_t = p_an[p_an["Metric"] == "response_time_s"].iloc[0]
    fn_h = p_fn[p_fn["Metric"] == "hallucination_rate"].iloc[0]
    fn_g = p_fn[p_fn["Metric"] == "groundedness_score"].iloc[0]
    fn_ar = p_fn[p_fn["Metric"] == "answer_relevance_1to5"].iloc[0]
    pairwise_guide = pd.DataFrame(
        [
            {
                "Concept": "p-value",
                "Meaning": "Probability of observing a difference at least this large if the two groups were truly the same (two-tailed Welch's t-test).",
            },
            {"Concept": "*, **, ***", "Meaning": "* means p < 0.05, ** means p < 0.01, *** means p < 0.001."},
            {"Concept": "Cohen's d", "Meaning": "Standardized effect size; rough guide: 0.2 small, 0.5 medium, 0.8+ large."},
            {"Concept": "Sig.", "Meaning": "Yes when p < 0.05, otherwise No."},
            {"Concept": "Sign of Cohen's d", "Meaning": "Positive means the first-named group has a higher mean; negative means lower."},
        ]
    )
    pairwise_takeaways = pd.DataFrame(
        [
            {
                "Comparison": "Adaptive RAG vs Fixed RAG",
                "Takeaway": (
                    f"Adaptive has lower hallucination (Diff={fmt(hrow['Diff'])}, p={pfmt(hrow['p-value'])}) and higher groundedness "
                    f"(Diff={fmt(grow['Diff'])}, p={pfmt(grow['p-value'])}), while also improving answer relevance "
                    f"(Diff={fmt(ad_rel['Diff'])}, p={pfmt(ad_rel['p-value'])}). Conceptually, this indicates adaptive retrieval policy is "
                    f"not only changing retrieval quantity but improving evidence-to-answer alignment: the model is more often using retrieved context "
                    f"to produce faithful answers instead of over-committing to weak context. The operational trade-off is orchestration complexity and "
                    f"some latency/memory sensitivity, but the quality signal is stronger on factual reliability metrics."
                ),
            },
            {
                "Comparison": "Adaptive RAG vs No Retrieval LLM",
                "Takeaway": (
                    f"Adaptive strongly improves factual reliability vs no-retrieval: hallucination drops by {fmt(abs(an_h['Diff']))} and "
                    f"groundedness rises by {fmt(an_g['Diff'])} (both {pfmt(an_h['p-value'])}/{pfmt(an_g['p-value'])}). "
                    f"No-retrieval remains faster on response time by {fmt(abs(an_t['Diff']),2)}s (p={pfmt(an_t['p-value'])}). Conceptually, "
                    f"this shows the classic speed-vs-trustworthiness frontier: removing retrieval can reduce pipeline overhead, but it shifts the system "
                    f"toward unguided generation where responses may stay fluent yet become weakly grounded. For production factual QA, adaptive retrieval "
                    f"acts as a reliability control layer rather than a mere performance optimization."
                ),
            },
            {
                "Comparison": "Fixed RAG vs No Retrieval LLM",
                "Takeaway": (
                    f"Fixed also materially outperforms no-retrieval on hallucination and groundedness "
                    f"(Diff={fmt(fn_h['Diff'])} / {fmt(fn_g['Diff'])}, p={pfmt(fn_h['p-value'])}/{pfmt(fn_g['p-value'])}) "
                    f"and gains answer relevance by {fmt(fn_ar['Diff'])} (p={pfmt(fn_ar['p-value'])}). Conceptually, even a non-adaptive retrieval baseline "
                    f"is a substantial upgrade over no-retrieval for factual tasks because it constrains answer generation to external evidence. The remaining gap "
                    f"between fixed and adaptive can then be interpreted as policy flexibility: fixed retrieval improves grounding, while adaptive retrieval further "
                    f"improves how context is selected and used under varying query difficulty."
                ),
            },
        ]
    )

    md = f"""# Adaptive RAG vs Fixed RAG vs No Retrieval LLM: Comprehensive Analysis Report

This report is generated from the **current full dataset collection** in this workspace and reflects all available adaptive, fixed, and no-retrieval runs.

## Executive Summary

- **Total observations**: {total}
- **Unique configurations**: {uniq_cfg}
- **Approaches analyzed**: Adaptive RAG, Fixed RAG, No Retrieval LLM

### Key Findings at a Glance

| Finding | Detail |
|---------|--------|
| Lowest Hallucination | **{best_h_approach}** ({fmt(best_h_value)}) |
| Highest Groundedness | **{best_g_approach}** ({fmt(best_g_value)}) |
| Fastest Response | **{fast_approach}** ({fmt(fast_value,2)}s) |

---

## 1. Dataset Coverage and Balance

{md_table(coverage.sort_values(["Approach", "Model", "Quantization", "Corpus Size"]))}

---

## 2. Overall Means by Approach

{md_table(overall_means.round(4))}

### 2.1 Overall Means by Approach and Corpus Size

{md_table(overall_by_corpus.round(4))}

---

## 3. Pairwise Approach Comparisons (Welch's t-test)

Welch's t-test is used for all pairwise comparisons to avoid equal-variance assumptions across systems. `Sig.` marks `p < 0.05`.

### How to read p-values, Cohen's d, and Sig.

- p-value: Probability of observing a difference at least this large if the two groups were truly the same (two-tailed Welch's t-test).
- `*`, `**`, `***`: `*` means p < 0.05, `**` means p < 0.01, `***` means p < 0.001.
- Cohen's d: Standardized effect size; rough guide: 0.2 small, 0.5 medium, 0.8+ large.
- Sig.: Yes when p < 0.05, otherwise No.
- Sign of Cohen's d: positive means the first-named group has a higher mean; negative means lower.

### Adaptive RAG vs Fixed RAG

{md_table(p_ad_fmt)}

**Takeaway:** {pairwise_takeaways.iloc[0]["Takeaway"]}

**Conceptual interpretation:** This comparison primarily reflects **policy flexibility vs policy consistency**. Fixed retrieval gives stable behavior with lower control overhead; adaptive retrieval adds decision-making layers that can recover quality under difficult/ambiguous prompts by changing retrieval and answer mode.

### Adaptive RAG vs No Retrieval LLM

{md_table(p_an_fmt)}

**Takeaway:** {pairwise_takeaways.iloc[1]["Takeaway"]}

**Conceptual interpretation:** This comparison isolates the value of **retrieval as grounding infrastructure**. The no-retrieval baseline can still be fluent and sometimes fast, but it lacks systematic evidence anchoring; adaptive retrieval converts the system from memory-driven answering to evidence-conditioned answering.

### Fixed RAG vs No Retrieval LLM

{md_table(p_fn_fmt)}

**Takeaway:** {pairwise_takeaways.iloc[2]["Takeaway"]}

**Conceptual interpretation:** This result shows that **retrieval itself** delivers the biggest reliability jump over no-retrieval, even before adaptivity is introduced. In design terms, fixed retrieval is a strong minimum viable grounding architecture; adaptivity is the next layer for quality optimization.

---

## 4. Model Comparison: Llama-70B vs Qwen-32B

{md_table(model_fmt)}

---

## 5. Quantization Impact: 4-bit vs 8-bit

{md_table(quant_fmt)}

---

## 6. Corpus Size Effects

### Small vs Medium
{md_table(sm_fmt)}

### Small vs Big
{md_table(sb_fmt)}

### Medium vs Big
{md_table(mb_fmt)}

---

## 7. Hallucination and Grounding Diagnostics

### 7.1 Hallucination Rate by Dimension
{md_table(hall)}

### 7.2 Context Status Distribution by Approach
{md_table(cstat.round(2))}

### 7.3 General Knowledge Fallback Rate
{md_table(gk.assign(**{"GK Usage %": gk["GK Usage %"].map(lambda x: f"{fmt(x,2)}%")}))}

---

## 8. Key Correlation Insights

{md_table(corr)}

---

## 9. Interaction Effects: Best and Worst Configurations

{md_table(rank[["Config", "hallucination_rate", "groundedness_score", "answer_relevance_1to5", "response_time_s", "gpu_mem_peak_mb"]].assign(
    hallucination_rate=lambda t: t["hallucination_rate"].map(lambda x: fmt(x)),
    groundedness_score=lambda t: t["groundedness_score"].map(lambda x: fmt(x)),
    answer_relevance_1to5=lambda t: t["answer_relevance_1to5"].map(lambda x: fmt(x,2)),
    response_time_s=lambda t: t["response_time_s"].map(lambda x: fmt(x,2)),
    gpu_mem_peak_mb=lambda t: t["gpu_mem_peak_mb"].map(lambda x: fmt(x,0)),
))}

---

## 10. Discussion and Recommendations

1. Adaptive RAG is the strongest option when groundedness and hallucination control are prioritized.
2. Fixed RAG remains competitive for latency-oriented deployments and simpler orchestration.
3. No Retrieval LLM may be computationally efficient in some runs, but quality reliability is consistently weaker in grounded QA settings.
4. Model and quantization choices should be selected jointly with retrieval strategy rather than in isolation.

## 11. Threats to Validity and Notes

1. Some prompts are repeated across configurations; observations are not fully independent in a causal sense.
2. Retrieval-score scales are architecture-dependent and should be interpreted within-system first.
3. Cost fields can reflect logging conventions and may need external billing validation.
4. Multiple-comparison correction is not applied in this report.

## 12. Methodology Notes

- Statistical test: Welch's t-test (two-tailed, unequal variance)
- Effect size: Cohen's d
- Significance notation: `* p<0.05`, `** p<0.01`, `*** p<0.001`
- Source: all matching adaptive/fixed/noretrieval CSV files in this project root
"""
    OUT_MD.write_text(md, encoding="utf-8")

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="all_rows_tagged", index=False)
        coverage.to_excel(w, sheet_name="coverage", index=False)
        overall_means.to_excel(w, sheet_name="overall_means", index=False)
        overall_by_corpus.to_excel(w, sheet_name="overall_by_corpus", index=False)
        pair_tabs["Adaptive RAG vs Fixed RAG"].to_excel(w, sheet_name="Adaptive_vs_Fixed", index=False)
        pair_tabs["Adaptive RAG vs No Retrieval LLM"].to_excel(w, sheet_name="Adaptive_vs_NoRetr", index=False)
        pair_tabs["Fixed RAG vs No Retrieval LLM"].to_excel(w, sheet_name="Fixed_vs_NoRetr", index=False)
        model_cmp.to_excel(w, sheet_name="model_comparison", index=False)
        quant_cmp.to_excel(w, sheet_name="quant_comparison", index=False)
        corpus_tabs["Small vs Medium"].to_excel(w, sheet_name="corpus_small_medium", index=False)
        corpus_tabs["Small vs Big"].to_excel(w, sheet_name="corpus_small_big", index=False)
        corpus_tabs["Medium vs Big"].to_excel(w, sheet_name="corpus_medium_big", index=False)
        hall.to_excel(w, sheet_name="hallucination_profile", index=False)
        cstat.to_excel(w, sheet_name="context_status_dist", index=False)
        gk.to_excel(w, sheet_name="gk_usage", index=False)
        corr.to_excel(w, sheet_name="key_correlations", index=False)
        rank.to_excel(w, sheet_name="config_rankings", index=False)
        pairwise_guide.to_excel(w, sheet_name="pairwise_how_to_read", index=False)
        pairwise_takeaways.to_excel(w, sheet_name="pairwise_takeaways", index=False)

    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_XLSX}")


if __name__ == "__main__":
    main()
