from __future__ import annotations

from itertools import combinations
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).parent
MD_OUTPUT = ROOT / "RAG_vs_SSFR_Analysis_Report_All_Datasets.md"
XLSX_OUTPUT = ROOT / "RAG_vs_SSFR_Comprehensive_Analysis_All_Datasets.xlsx"

APPROACH_DISPLAY = {
    "adaptive": "Adaptive RAG",
    "fixed": "Fixed RAG",
    "noretrieval": "No Retrieval",
}
MODEL_DISPLAY = {
    "qwen": "Qwen-32B",
    "llama": "Llama",
}
METRICS_QUALITY = [
    "hallucination_rate",
    "groundedness_score",
    "answer_relevance_1to5",
    "context_relevance_1to5",
    "confidence",
    "query_coverage",
]
METRICS_PERF = [
    "response_time_s",
    "llm_latency_s",
    "gpu_throughput_toks_per_s",
    "eff_gpu_throughput",
    "gpu_util_percent",
    "gpu_mem_percent",
    "gpu_mem_peak_mb",
    "total_deployment_cost_usd",
]
ALL_METRICS = METRICS_QUALITY + METRICS_PERF
HIGHER_IS_BETTER = {
    "hallucination_rate": False,
    "groundedness_score": True,
    "answer_relevance_1to5": True,
    "context_relevance_1to5": True,
    "confidence": True,
    "query_coverage": True,
    "response_time_s": False,
    "llm_latency_s": False,
    "gpu_throughput_toks_per_s": True,
    "eff_gpu_throughput": True,
    "gpu_util_percent": True,
    "gpu_mem_percent": False,
    "gpu_mem_peak_mb": False,
    "total_deployment_cost_usd": False,
}


def parse_metadata(file_name: str) -> dict:
    lower = file_name.lower()

    if lower.startswith("rag_eval_"):
        approach = "adaptive"
        model = "llama" if "llama" in lower else "qwen"
    elif lower.startswith("fixed_rag_results_"):
        approach = "fixed"
        model = "llama" if "llama" in lower else "qwen"
    elif lower.startswith("noretrieval_"):
        approach = "noretrieval"
        model = "llama" if "llama" in lower else "qwen"
    else:
        raise ValueError(f"Could not parse approach/model from: {file_name}")

    quant_match = re.search(r"([48])bit", lower)
    size_match = re.search(r"(small|medium|big)", lower)
    if not quant_match or not size_match:
        raise ValueError(f"Could not parse quantization/corpus size from: {file_name}")

    return {
        "approach": approach,
        "model_family": model,
        "quantization": f"{quant_match.group(1)}-bit",
        "corpus_size": size_match.group(1).capitalize(),
    }


def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    file_audit_rows: list[dict] = []

    for csv_path in sorted(ROOT.glob("*.csv")):
        if not (
            csv_path.name.startswith("rag_eval_")
            or csv_path.name.startswith("fixed_rag_results_")
            or csv_path.name.startswith("noretrieval_")
        ):
            continue

        meta = parse_metadata(csv_path.name)
        df = pd.read_csv(csv_path)
        for k, v in meta.items():
            df[k] = v
        df["source_file"] = csv_path.name
        df["query_norm"] = (
            df.get("query", pd.Series([""] * len(df))).astype(str).str.strip().str.lower()
        )
        if "used_general_knowledge" not in df.columns:
            df["used_general_knowledge"] = np.nan
        df["used_general_knowledge"] = (
            df["used_general_knowledge"]
            .astype(str)
            .str.lower()
            .map({"true": 1, "false": 0})
            .fillna(pd.to_numeric(df["used_general_knowledge"], errors="coerce"))
        )

        rows.append(df)
        file_audit_rows.append(
            {
                "source_file": csv_path.name,
                "rows": len(df),
                "approach": meta["approach"],
                "model_family": meta["model_family"],
                "quantization": meta["quantization"],
                "corpus_size": meta["corpus_size"],
            }
        )

    if not rows:
        raise RuntimeError("No matching CSV files found.")

    data = pd.concat(rows, ignore_index=True, sort=False)
    file_audit = pd.DataFrame(file_audit_rows)
    return data, file_audit


def fmt_num(x: float, digits: int = 4) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{digits}f}"


def fmt_pm(mean: float, std: float, digits: int = 4) -> str:
    return f"{fmt_num(mean, digits)} +/- {fmt_num(std, digits)}"


def fmt_p_value(p: float) -> str:
    if pd.isna(p):
        return "NA"
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{p:.4f}{stars}"


def cohen_d(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    s1 = x.std(ddof=1)
    s2 = y.std(ddof=1)
    n1 = len(x)
    n2 = len(y)
    pooled = np.sqrt(((n1 - 1) * (s1 ** 2) + (n2 - 1) * (s2 ** 2)) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return (x.mean() - y.mean()) / pooled


def welch_test(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan
    t, p = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return t, p


def pairwise_metric_table(data: pd.DataFrame, group_col: str, a: str, b: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    ga = data[data[group_col] == a]
    gb = data[data[group_col] == b]
    for metric in metrics:
        if metric not in data.columns:
            continue
        x = pd.to_numeric(ga[metric], errors="coerce")
        y = pd.to_numeric(gb[metric], errors="coerce")
        t, p = welch_test(x, y)
        d = cohen_d(x, y)
        diff = x.mean() - y.mean()
        rows.append(
            {
                "metric": metric,
                f"{a}_mean_std": fmt_pm(x.mean(), x.std(ddof=1)),
                f"{b}_mean_std": fmt_pm(y.mean(), y.std(ddof=1)),
                "diff": diff,
                "p_value": p,
                "cohens_d": d,
                "significant": "Yes" if (not pd.isna(p) and p < 0.05) else "No",
            }
        )
    return pd.DataFrame(rows)


def one_factor_table(data: pd.DataFrame, factor: str, metrics: list[str]) -> pd.DataFrame:
    values = list(data[factor].dropna().unique())
    if len(values) != 2:
        return pd.DataFrame()
    a, b = values[0], values[1]
    ga = data[data[factor] == a]
    gb = data[data[factor] == b]
    rows = []
    for metric in metrics:
        if metric not in data.columns:
            continue
        x = pd.to_numeric(ga[metric], errors="coerce")
        y = pd.to_numeric(gb[metric], errors="coerce")
        _, p = welch_test(x, y)
        d = cohen_d(x, y)
        rows.append(
            {
                "metric": metric,
                f"{a}_mean": x.mean(),
                f"{b}_mean": y.mean(),
                "diff": x.mean() - y.mean(),
                "p_value": p,
                "cohens_d": d,
                "significant": "Yes" if (not pd.isna(p) and p < 0.05) else "No",
            }
        )
    return pd.DataFrame(rows)


def hallucination_by_dimension(data: pd.DataFrame) -> pd.DataFrame:
    out = []
    dims = ["approach", "model_family", "quantization", "corpus_size"]
    for dim in dims:
        for val, g in data.groupby(dim):
            h = pd.to_numeric(g["hallucination_rate"], errors="coerce")
            out.append(
                {
                    "dimension": dim,
                    "value": val,
                    "mean": h.mean(),
                    "median": h.median(),
                    "std": h.std(ddof=1),
                    "pct_zero": (h == 0).mean() * 100,
                    "pct_above_0_3": (h > 0.3).mean() * 100,
                    "n": len(g),
                }
            )
    return pd.DataFrame(out)


def context_status_distribution(data: pd.DataFrame) -> pd.DataFrame:
    if "context_status" not in data.columns:
        return pd.DataFrame()
    tab = (
        data.groupby(["approach", "context_status"])
        .size()
        .reset_index(name="count")
    )
    totals = tab.groupby("approach")["count"].transform("sum")
    tab["percent"] = (tab["count"] / totals) * 100
    pivot = tab.pivot(index="approach", columns="context_status", values="percent").fillna(0)
    pivot = pivot.reset_index()
    return pivot


def gk_usage_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dims = ["approach", "model_family", "quantization", "corpus_size"]
    for dim in dims:
        for val, g in data.groupby(dim):
            gk = pd.to_numeric(g["used_general_knowledge"], errors="coerce")
            rows.append(
                {
                    "dimension": dim,
                    "value": val,
                    "gk_usage_percent": gk.mean() * 100,
                    "n": len(g),
                }
            )
    return pd.DataFrame(rows)


def correlation_table(data: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("hallucination_rate", "groundedness_score"),
        ("hallucination_rate", "response_time_s"),
        ("groundedness_score", "response_time_s"),
        ("groundedness_score", "confidence"),
        ("answer_relevance_1to5", "context_relevance_1to5"),
        ("hallucination_rate", "retrieved_docs_count"),
        ("groundedness_score", "retrieved_docs_count"),
        ("gpu_throughput_toks_per_s", "response_time_s"),
        ("gpu_mem_peak_mb", "gpu_throughput_toks_per_s"),
        ("confidence", "hallucination_rate"),
        ("query_coverage", "groundedness_score"),
        ("query_coverage", "hallucination_rate"),
    ]
    rows = []
    for a, b in pairs:
        if a not in data.columns or b not in data.columns:
            continue
        df = data[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
        r = df[a].corr(df[b]) if len(df) > 1 else np.nan
        rows.append({"metric_a": a, "metric_b": b, "pearson_r": r})
    return pd.DataFrame(rows)


def ranked_configurations(data: pd.DataFrame) -> pd.DataFrame:
    g = (
        data.groupby(["approach", "model_family", "quantization", "corpus_size"], dropna=False)
        .agg(
            hallucination_rate=("hallucination_rate", "mean"),
            groundedness_score=("groundedness_score", "mean"),
            answer_relevance_1to5=("answer_relevance_1to5", "mean"),
            response_time_s=("response_time_s", "mean"),
            gpu_mem_peak_mb=("gpu_mem_peak_mb", "mean"),
        )
        .reset_index()
    )
    g["quality_index"] = (
        g["groundedness_score"] * 0.45
        + g["answer_relevance_1to5"] / 5.0 * 0.35
        + (1 - g["hallucination_rate"]) * 0.20
    )
    g = g.sort_values(["quality_index", "groundedness_score"], ascending=[False, False])
    g["config"] = (
        g["approach"].map(APPROACH_DISPLAY)
        + "|"
        + g["model_family"].map(MODEL_DISPLAY)
        + "|"
        + g["quantization"]
        + "|"
        + g["corpus_size"]
    )
    return g


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No data available._"
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                s = "NA"
            else:
                s = str(v)
            vals.append(s.replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_outputs(
    data: pd.DataFrame,
    file_audit: pd.DataFrame,
    pairwise_tables: dict[str, pd.DataFrame],
    model_table: pd.DataFrame,
    quant_table: pd.DataFrame,
    corpus_table: pd.DataFrame,
    hall_dim: pd.DataFrame,
    context_dist: pd.DataFrame,
    gk_table: pd.DataFrame,
    corr_table: pd.DataFrame,
    ranked: pd.DataFrame,
) -> None:
    approach_counts = data["approach"].value_counts().to_dict()
    total_obs = len(data)
    unique_cfg = (
        data[["approach", "model_family", "quantization", "corpus_size"]]
        .drop_duplicates()
        .shape[0]
    )

    pair_names = list(pairwise_tables.keys())
    pair_sections = []
    for key in pair_names:
        qa = pairwise_tables[key]["quality"]
        pe = pairwise_tables[key]["performance"]
        pair_sections.append(
            f"""### {key}: Answer Quality Metrics

{df_to_markdown(qa.assign(diff=qa["diff"].map(lambda x: fmt_num(x, 4)),
                           p_value=qa["p_value"].map(fmt_p_value),
                           cohens_d=qa["cohens_d"].map(lambda x: fmt_num(x, 3))))}

### {key}: Performance Metrics

{df_to_markdown(pe.assign(diff=pe["diff"].map(lambda x: fmt_num(x, 2)),
                           p_value=pe["p_value"].map(fmt_p_value),
                           cohens_d=pe["cohens_d"].map(lambda x: fmt_num(x, 3))))}
"""
        )

    best_ground = ranked.loc[ranked["groundedness_score"].idxmax()]
    best_hall = ranked.loc[ranked["hallucination_rate"].idxmin()]
    fastest = ranked.loc[ranked["response_time_s"].idxmin()]
    worst_ground = ranked.loc[ranked["groundedness_score"].idxmin()]

    md = f"""# Adaptive vs Fixed vs No-Retrieval Comparative Evaluation: Comprehensive Analysis Report

## Executive Summary

This report presents a comparative evaluation across **{total_obs} observations** and **{unique_cfg} unique configurations** spanning adaptive RAG, fixed RAG, and no-retrieval baselines.

| Variable | Levels |
|----------|--------|
| Retrieval Approach | Adaptive RAG, Fixed RAG, No Retrieval |
| Base Model | Qwen-32B, Llama |
| Quantization | 4-bit, 8-bit |
| Corpus Size | Small, Medium, Big |

- **Adaptive observations**: {approach_counts.get("adaptive", 0)}
- **Fixed observations**: {approach_counts.get("fixed", 0)}
- **No-retrieval observations**: {approach_counts.get("noretrieval", 0)}

## 1. Approach Head-to-Head Comparisons

{''.join(pair_sections)}

## 2. Model Comparison: Qwen-32B vs. Llama

{df_to_markdown(model_table.assign(
    diff=model_table["diff"].map(lambda x: fmt_num(x, 4)),
    p_value=model_table["p_value"].map(fmt_p_value),
    cohens_d=model_table["cohens_d"].map(lambda x: fmt_num(x, 3))
))}

## 3. Quantization Impact: 4-bit vs. 8-bit

{df_to_markdown(quant_table.assign(
    diff=quant_table["diff"].map(lambda x: fmt_num(x, 4)),
    p_value=quant_table["p_value"].map(fmt_p_value),
    cohens_d=quant_table["cohens_d"].map(lambda x: fmt_num(x, 3))
))}

## 4. Corpus Size Impact: Small vs. Medium

{df_to_markdown(corpus_table.assign(
    diff=corpus_table["diff"].map(lambda x: fmt_num(x, 4)),
    p_value=corpus_table["p_value"].map(fmt_p_value),
    cohens_d=corpus_table["cohens_d"].map(lambda x: fmt_num(x, 3))
))}

## 5. Hallucination Analysis

### 5.1 Hallucination Rate by Dimension

{df_to_markdown(hall_dim.assign(
    mean=hall_dim["mean"].map(lambda x: fmt_num(x, 4)),
    median=hall_dim["median"].map(lambda x: fmt_num(x, 4)),
    std=hall_dim["std"].map(lambda x: fmt_num(x, 4)),
    pct_zero=hall_dim["pct_zero"].map(lambda x: f"{fmt_num(x, 1)}%"),
    pct_above_0_3=hall_dim["pct_above_0_3"].map(lambda x: f"{fmt_num(x, 1)}%"),
))}

### 5.2 Context Grounding Status Distribution

{df_to_markdown(context_dist.round(2))}

### 5.3 General Knowledge Fallback Rate

{df_to_markdown(gk_table.assign(gk_usage_percent=gk_table["gk_usage_percent"].map(lambda x: f"{fmt_num(x, 2)}%")))}

## 6. Key Correlation Insights

{df_to_markdown(corr_table.assign(pearson_r=corr_table["pearson_r"].map(lambda x: fmt_num(x, 4))))}

## 7. Interaction Effects: Best & Worst Configurations

{df_to_markdown(ranked[["config", "hallucination_rate", "groundedness_score", "answer_relevance_1to5", "response_time_s", "gpu_mem_peak_mb"]].assign(
    hallucination_rate=lambda x: x["hallucination_rate"].map(lambda y: fmt_num(y, 4)),
    groundedness_score=lambda x: x["groundedness_score"].map(lambda y: fmt_num(y, 4)),
    answer_relevance_1to5=lambda x: x["answer_relevance_1to5"].map(lambda y: fmt_num(y, 2)),
    response_time_s=lambda x: x["response_time_s"].map(lambda y: fmt_num(y, 2)),
    gpu_mem_peak_mb=lambda x: x["gpu_mem_peak_mb"].map(lambda y: fmt_num(y, 0)),
))}

- **Highest Groundedness**: `{best_ground["config"]}` (groundedness={fmt_num(best_ground["groundedness_score"], 4)})
- **Lowest Hallucination**: `{best_hall["config"]}` (hallucination={fmt_num(best_hall["hallucination_rate"], 4)})
- **Fastest Response**: `{fastest["config"]}` (response_time={fmt_num(fastest["response_time_s"], 2)}s)
- **Lowest Groundedness**: `{worst_ground["config"]}` (groundedness={fmt_num(worst_ground["groundedness_score"], 4)})

## 8. Methodology Notes

- Total observations: {total_obs}
- Source CSV files: {len(file_audit)}
- Unique configurations: {unique_cfg}
- Statistical tests: Welch's t-test (unequal variances), two-tailed
- Effect sizes: Cohen's d
- Significance levels: * p<0.05, ** p<0.01, *** p<0.001

---

*Report generated automatically from all adaptive, fixed, and no-retrieval CSV datasets found in this project directory.*
"""

    MD_OUTPUT.write_text(md, encoding="utf-8")

    with pd.ExcelWriter(XLSX_OUTPUT, engine="openpyxl") as writer:
        data.to_excel(writer, sheet_name="raw_combined", index=False)
        file_audit.to_excel(writer, sheet_name="file_audit", index=False)
        for key, tabs in pairwise_tables.items():
            q_name = (f"{key}_quality"[:31]).replace(" ", "_")
            p_name = (f"{key}_perf"[:31]).replace(" ", "_")
            tabs["quality"].to_excel(writer, sheet_name=q_name, index=False)
            tabs["performance"].to_excel(writer, sheet_name=p_name, index=False)
        model_table.to_excel(writer, sheet_name="model_comparison", index=False)
        quant_table.to_excel(writer, sheet_name="quantization", index=False)
        corpus_table.to_excel(writer, sheet_name="corpus_small_medium", index=False)
        hall_dim.to_excel(writer, sheet_name="hallucination_by_dim", index=False)
        context_dist.to_excel(writer, sheet_name="context_status", index=False)
        gk_table.to_excel(writer, sheet_name="gk_usage", index=False)
        corr_table.to_excel(writer, sheet_name="correlations", index=False)
        ranked.to_excel(writer, sheet_name="config_ranking", index=False)


def main() -> None:
    data, file_audit = load_all_data()

    data["approach_label"] = data["approach"].map(APPROACH_DISPLAY)

    pairwise_tables: dict[str, dict[str, pd.DataFrame]] = {}
    for a, b in combinations(sorted(data["approach"].unique()), 2):
        pair_name = f"{APPROACH_DISPLAY[a]} vs {APPROACH_DISPLAY[b]}"
        pairwise_tables[pair_name] = {
            "quality": pairwise_metric_table(data, "approach", a, b, METRICS_QUALITY),
            "performance": pairwise_metric_table(data, "approach", a, b, METRICS_PERF),
        }

    model_table = one_factor_table(data.assign(model_family=data["model_family"].map(MODEL_DISPLAY)), "model_family", ALL_METRICS)
    quant_table = one_factor_table(data, "quantization", ALL_METRICS)
    # Keep this aligned with the reference section shape (two-level t-test table).
    corpus_table = one_factor_table(data[data["corpus_size"].isin(["Small", "Medium"])], "corpus_size", ALL_METRICS)

    hall_dim = hallucination_by_dimension(data)
    context_dist = context_status_distribution(data)
    gk_table = gk_usage_table(data)
    corr_table = correlation_table(data)
    ranked = ranked_configurations(data)

    write_outputs(
        data=data,
        file_audit=file_audit,
        pairwise_tables=pairwise_tables,
        model_table=model_table,
        quant_table=quant_table,
        corpus_table=corpus_table,
        hall_dim=hall_dim,
        context_dist=context_dist,
        gk_table=gk_table,
        corr_table=corr_table,
        ranked=ranked,
    )

    print(f"Wrote markdown: {MD_OUTPUT}")
    print(f"Wrote workbook: {XLSX_OUTPUT}")


if __name__ == "__main__":
    main()
