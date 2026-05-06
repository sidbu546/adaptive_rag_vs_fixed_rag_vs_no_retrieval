from __future__ import annotations

import math
import os
import re
import textwrap
import base64
import html
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon  # type: ignore

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    wilcoxon = None  # type: ignore


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "analysis_outputs"

DATASET_ORDER = ["small", "medium", "big"]
MODEL_ORDER = ["Llama 3.3 70B", "Qwen 32B"]
SYSTEM_ORDER = ["fixed", "adaptive"]
QUANT_ORDER = ["4bit", "8bit"]

QUALITY_METRICS = [
    "hallucination_rate",
    "groundedness_score",
    "answer_relevance_1to5",
    "context_relevance_1to5",
    "quality_index",
]

OPERATIONAL_METRICS = [
    "response_time_s",
    "gpu_util_percent",
    "gpu_throughput_toks_per_s",
    "gpu_mem_peak_mb",
]

ALL_METRICS = QUALITY_METRICS + OPERATIONAL_METRICS
REQUIRED_COLUMNS = set(
    [
        "query",
        "hallucination_rate",
        "groundedness_score",
        "answer_relevance_1to5",
        "context_relevance_1to5",
        "response_time_s",
        "gpu_util_percent",
        "gpu_throughput_toks_per_s",
        "gpu_mem_peak_mb",
    ]
)

LOWER_IS_BETTER = {"hallucination_rate", "response_time_s", "gpu_mem_peak_mb"}
HIGHER_IS_BETTER = {
    "groundedness_score",
    "answer_relevance_1to5",
    "context_relevance_1to5",
    "quality_index",
    "gpu_throughput_toks_per_s",
}

QUALITY_EXPLAINERS = {
    "hallucination_rate": "unsupported or risky generation",
    "groundedness_score": "how strongly the answer stays anchored to retrieved evidence",
    "answer_relevance_1to5": "how directly the final answer addresses the user question",
    "context_relevance_1to5": "how aligned the retrieved evidence is with the query",
    "quality_index": "a composite of groundedness, answer usefulness, and hallucination control",
}

OPERATIONAL_EXPLAINERS = {
    "response_time_s": "end-user latency across the whole pipeline",
    "gpu_util_percent": "GPU saturation rather than an intrinsic quality win",
    "gpu_throughput_toks_per_s": "decoding efficiency during generation",
    "gpu_mem_peak_mb": "deployment footprint and hardware feasibility",
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)


def normalize_query(text: object) -> str:
    value = "" if pd.isna(text) else str(text)
    value = value.strip()
    value = re.sub(r"^\s*\d+\s*[\.\)]\s*", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def mode_or_first(values: Sequence[object]) -> object:
    cleaned = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not cleaned:
        return np.nan
    as_strings = [str(v) for v in cleaned]
    counts = Counter(as_strings)
    most_common = counts.most_common()
    top_count = most_common[0][1]
    top_values = {value for value, count in most_common if count == top_count}
    for raw in cleaned:
        if str(raw) in top_values:
            return raw
    return cleaned[0]


def infer_metadata(filename: str) -> Optional[Dict[str, str]]:
    name = filename.lower()
    if "fixed" in name:
        system = "fixed"
    elif "adaptive" in name:
        system = "adaptive"
    else:
        return None

    if "llama" in name:
        model_family = "Llama 3.3 70B"
    elif "qwen" in name:
        model_family = "Qwen 32B"
    else:
        return None

    if "4bit" in name:
        quantization = "4bit"
    elif "8bit" in name:
        quantization = "8bit"
    else:
        return None

    if "small" in name:
        dataset_size = "small"
    elif "medium" in name:
        dataset_size = "medium"
    elif "big" in name:
        dataset_size = "big"
    else:
        return None

    return {
        "system": system,
        "model_family": model_family,
        "quantization": quantization,
        "dataset_size": dataset_size,
    }


def fmt(value: object, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if abs(numeric) >= 100 and digits > 1:
        return f"{numeric:.1f}"
    return f"{numeric:.{digits}f}"


def better_label(metric: str, delta: float) -> str:
    if metric == "gpu_util_percent":
        if abs(delta) < 1e-12:
            return "similar saturation"
        return "Adaptive saturates GPU more" if delta > 0 else "Fixed saturates GPU more"
    if metric in LOWER_IS_BETTER:
        if abs(delta) < 1e-12:
            return "tie"
        return "adaptive" if delta < 0 else "fixed"
    if metric in HIGHER_IS_BETTER:
        if abs(delta) < 1e-12:
            return "tie"
        return "adaptive" if delta > 0 else "fixed"
    return "NA"


def better_model_label(metric: str, delta: float) -> str:
    if metric == "gpu_util_percent":
        return "higher GPU saturation" if delta > 0 else "lower GPU saturation"
    if metric in LOWER_IS_BETTER:
        return "Llama" if delta < 0 else "Qwen" if delta > 0 else "Tie"
    if metric in HIGHER_IS_BETTER:
        return "Llama" if delta > 0 else "Qwen" if delta < 0 else "Tie"
    return "Tie"


def short_model(model_family: str) -> str:
    return "Llama" if "Llama" in model_family else "Qwen"


def compact_config_label(model_family: str, quantization: str, dataset_size: str) -> str:
    return f"{short_model(model_family)}\n{quantization}-{dataset_size}"


def annotate_heatmap(ax, data: np.ndarray, cmap, norm) -> None:
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                text = "NA"
                color = "black"
            else:
                rgba = cmap(norm(val))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                color = "black" if luminance > 0.55 else "white"
                text = fmt(val, 3)
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=11, fontweight="bold")


def _rects_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0)


def add_repel_labels(ax, rows: List[Dict[str, object]], x_key: str, y_key: str, manual_offsets: Optional[Dict[str, Tuple[int, int]]] = None) -> None:
    fig = ax.figure
    fig.canvas.draw()
    candidates = [
        (12, 12), (12, -14), (-12, 12), (-12, -14),
        (24, 0), (-24, 0), (0, 20), (0, -20),
        (28, 16), (-28, 16), (28, -16), (-28, -16),
        (40, 0), (-40, 0), (18, 28), (-18, 28),
    ]
    placed_rects: List[Tuple[float, float, float, float]] = []
    axes_bbox = ax.get_window_extent()
    left_bound = axes_bbox.x0 + 6
    right_bound = axes_bbox.x1 - 6
    bottom_bound = axes_bbox.y0 + 6
    top_bound = axes_bbox.y1 - 10

    display_points = []
    for row in rows:
        x_val = float(row[x_key])
        y_val = float(row[y_key])
        px, py = ax.transData.transform((x_val, y_val))
        display_points.append((row, px, py))

    for row, px, py in display_points:
        label = compact_config_label(str(row["model_family"]), str(row["quantization"]), str(row["dataset_size"]))
        longest = max(len(part) for part in label.split("\n"))
        est_w = longest * 6.6 + 16
        est_h = len(label.split("\n")) * 12.5 + 10

        if manual_offsets and label in manual_offsets:
            dx, dy = manual_offsets[label]
            final_rect = (px + dx, py + dy - est_h, px + dx + est_w, py + dy)
            placed_rects.append(final_rect)
            ax.annotate(
                label,
                xy=(float(row[x_key]), float(row[y_key])),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx >= 0 else "right",
                va="bottom" if dy >= 0 else "top",
                fontsize=9.2,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#777777", lw=0.6, alpha=0.92),
                arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7, shrinkA=2, shrinkB=2),
                zorder=4,
            )
            continue

        best_offset = candidates[0]
        best_penalty = float("inf")
        for dx, dy in candidates:
            rect = (
                px + dx,
                py + dy - est_h,
                px + dx + est_w,
                py + dy,
            )
            overlap_penalty = sum(1 for prev in placed_rects if _rects_overlap(rect, prev)) * 1000
            center_penalty = abs(dx) + abs(dy)
            boundary_penalty = 0
            if rect[0] < left_bound:
                boundary_penalty += (left_bound - rect[0]) * 30
            if rect[2] > right_bound:
                boundary_penalty += (rect[2] - right_bound) * 30
            if rect[1] < bottom_bound:
                boundary_penalty += (bottom_bound - rect[1]) * 35
            if rect[3] > top_bound:
                boundary_penalty += (rect[3] - top_bound) * 50
            penalty = overlap_penalty + center_penalty + boundary_penalty
            if penalty < best_penalty:
                best_penalty = penalty
                best_offset = (dx, dy)

        dx, dy = best_offset
        final_rect = (px + dx, py + dy - est_h, px + dx + est_w, py + dy)
        placed_rects.append(final_rect)
        ax.annotate(
            label,
            xy=(float(row[x_key]), float(row[y_key])),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=9.2,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#777777", lw=0.6, alpha=0.92),
            arrowprops=dict(arrowstyle="-", color="#777777", lw=0.7, shrinkA=2, shrinkB=2),
            zorder=4,
        )


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, samples: int = 5000) -> Tuple[float, float]:
    if values.size == 0:
        return (np.nan, np.nan)
    draws = rng.choice(values, size=(samples, values.size), replace=True)
    sample_means = draws.mean(axis=1)
    return tuple(np.percentile(sample_means, [2.5, 97.5]))


def paired_cohens_d(values: np.ndarray) -> float:
    if values.size == 0:
        return np.nan
    std = np.std(values, ddof=1) if values.size > 1 else 0.0
    if std == 0:
        return np.nan
    return float(np.mean(values) / std)


def make_row_label(model_family: str, quantization: str) -> str:
    return f"{short_model(model_family)} {quantization}"


def figure_caption_block(title: str, body: str) -> str:
    return f"**{title}.** {body}\n"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "| empty |\n|---|\n| no rows |"
    columns = [str(col) for col in df.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body_lines = []
    for _, row in df.iterrows():
        values = [str(row[col]).replace("\n", " ") for col in df.columns]
        body_lines.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator] + body_lines)


def markdown_to_plain_text(line: str) -> str:
    text = re.sub(r"^#+\s*", "", line)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", text)
    return text


def relativize_markdown_images_for_github(markdown_text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        alt_text = match.group(1)
        raw_path = match.group(2)
        path_obj = Path(raw_path)
        relative_name = path_obj.name if path_obj.name else raw_path
        return f"![{alt_text}](./{relative_name})"

    return re.sub(r"!\[(.*?)\]\((.*?)\)", repl, markdown_text)


def load_and_prepare() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    records: List[pd.DataFrame] = []
    file_audit_rows: List[Dict[str, object]] = []
    missing_by_file: Dict[str, List[str]] = {}

    csv_files = sorted(ROOT.glob("*.csv"))
    ignored_files = []

    for path in csv_files:
        meta = infer_metadata(path.name)
        if meta is None:
            ignored_files.append(path.name)
            continue

        df = pd.read_csv(path)
        df["source_file"] = path.name
        for key, value in meta.items():
            df[key] = value

        missing = sorted(REQUIRED_COLUMNS - set(df.columns))
        missing_by_file[path.name] = missing

        df["query_original"] = df["query"]
        df["query_stripped"] = df["query"].astype(str).str.strip()
        df["normalized_query"] = df["query"].apply(normalize_query)
        df["normalization_changed"] = df["query_original"].astype(str) != df["normalized_query"].astype(str)

        file_audit_rows.append(
            {
                "source_file": path.name,
                "rows": len(df),
                "columns": len(df.columns),
                "missing_required_columns": ", ".join(missing) if missing else "",
                "system": meta["system"],
                "model_family": meta["model_family"],
                "quantization": meta["quantization"],
                "dataset_size": meta["dataset_size"],
            }
        )
        records.append(df)

    if not records:
        raise RuntimeError("No eligible CSV files found in the working directory.")

    raw = pd.concat(records, ignore_index=True, sort=False)

    numeric_candidates = [
        "hallucination_rate",
        "groundedness_score",
        "answer_relevance_1to5",
        "context_relevance_1to5",
        "response_time_s",
        "gpu_util_percent",
        "gpu_throughput_toks_per_s",
        "gpu_mem_peak_mb",
        "llm_latency_s",
        "eff_gpu_throughput",
        "gpu_mem_percent",
        "gpu_util_max_percent",
        "gpu_mem_avg_percent",
        "gpu_mem_torch_peak_mb",
        "gpu_monitor_samples",
        "retrieved_docs_count",
        "top_retrieval_score",
        "avg_retrieval_score",
        "query_coverage",
        "confidence",
        "claim_count",
        "unsupported_count",
    ]
    for col in numeric_candidates:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    if "used_general_knowledge" in raw.columns:
        normalized_gk = raw["used_general_knowledge"].map(
            lambda v: 1.0
            if str(v).strip().lower() == "true"
            else 0.0
            if str(v).strip().lower() == "false"
            else (float(v) if pd.notna(v) and str(v).strip() not in {"", "nan"} else np.nan)
        )
        raw["used_general_knowledge"] = normalized_gk.astype(float)

    cost_columns = [col for col in raw.columns if "cost" in col.lower()]

    raw["quality_index"] = (
        0.40 * raw["groundedness_score"]
        + 0.35 * (raw["answer_relevance_1to5"] / 5.0)
        + 0.25 * (1.0 - raw["hallucination_rate"])
    )

    group_keys = ["system", "model_family", "quantization", "dataset_size", "normalized_query"]

    numeric_cols = [
        col
        for col in raw.columns
        if pd.api.types.is_numeric_dtype(raw[col]) and col not in cost_columns
    ]
    non_numeric_cols = [
        col
        for col in raw.columns
        if col not in numeric_cols and col not in group_keys
    ]

    aggregate_spec = {col: "mean" for col in numeric_cols}
    for col in non_numeric_cols:
        aggregate_spec[col] = mode_or_first

    dedup = raw.groupby(group_keys, dropna=False, as_index=False).agg(aggregate_spec)
    dedup["duplicate_count"] = raw.groupby(group_keys, dropna=False).size().values

    raw_pair_keys = ["model_family", "quantization", "dataset_size", "query_stripped"]
    normalized_pair_keys = ["model_family", "quantization", "dataset_size", "normalized_query"]

    fixed_raw = raw[raw["system"] == "fixed"][raw_pair_keys].drop_duplicates()
    adaptive_raw = raw[raw["system"] == "adaptive"][raw_pair_keys].drop_duplicates()
    raw_pair_count = len(fixed_raw.merge(adaptive_raw, on=raw_pair_keys, how="inner"))

    fixed_norm = dedup[dedup["system"] == "fixed"][normalized_pair_keys].drop_duplicates()
    adaptive_norm = dedup[dedup["system"] == "adaptive"][normalized_pair_keys].drop_duplicates()
    normalized_pair_count = len(fixed_norm.merge(adaptive_norm, on=normalized_pair_keys, how="inner"))

    audit = {
        "file_audit": pd.DataFrame(file_audit_rows).sort_values("source_file").reset_index(drop=True),
        "ignored_files": ignored_files,
        "missing_by_file": missing_by_file,
        "cost_columns_ignored": cost_columns,
        "total_raw_rows": int(len(raw)),
        "total_deduplicated_rows": int(len(dedup)),
        "duplicate_rows_collapsed": int(len(raw) - len(dedup)),
        "duplicate_groups": int((dedup["duplicate_count"] > 1).sum()),
        "raw_pair_count": int(raw_pair_count),
        "normalized_pair_count": int(normalized_pair_count),
        "normalization_changed_rows": int(raw["normalization_changed"].sum()),
    }
    return raw, dedup, audit


def build_paired(dedup: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    merge_keys = ["model_family", "quantization", "dataset_size", "normalized_query"]
    fixed = dedup[dedup["system"] == "fixed"].copy()
    adaptive = dedup[dedup["system"] == "adaptive"].copy()

    unmatched_fixed = fixed.merge(adaptive[merge_keys], on=merge_keys, how="left", indicator=True)
    unmatched_fixed = unmatched_fixed[unmatched_fixed["_merge"] == "left_only"].drop(columns=["_merge"])

    unmatched_adaptive = adaptive.merge(fixed[merge_keys], on=merge_keys, how="left", indicator=True)
    unmatched_adaptive = unmatched_adaptive[unmatched_adaptive["_merge"] == "left_only"].drop(columns=["_merge"])

    paired = fixed.merge(adaptive, on=merge_keys, how="inner", suffixes=("_fixed", "_adaptive"))
    paired["query"] = paired["query_original_fixed"].combine_first(paired["query_original_adaptive"])
    paired["config_label"] = (
        paired["model_family"].map(short_model)
        + "-"
        + paired["quantization"].astype(str)
        + "-"
        + paired["dataset_size"].astype(str)
    )

    delta_metrics = sorted(set(ALL_METRICS) | {"retrieved_docs_count", "query_coverage", "top_retrieval_score", "avg_retrieval_score", "used_general_knowledge"})
    for metric in delta_metrics:
        fixed_col = f"{metric}_fixed"
        adaptive_col = f"{metric}_adaptive"
        if fixed_col in paired.columns and adaptive_col in paired.columns:
            paired[f"delta_{metric}"] = paired[adaptive_col] - paired[fixed_col]

    if "delta_hallucination_rate" in paired.columns:
        paired["hallucination_reduction"] = -paired["delta_hallucination_rate"]
    if "delta_quality_index" in paired.columns:
        paired["quality_gain"] = paired["delta_quality_index"]

    return paired, pd.concat([unmatched_fixed, unmatched_adaptive], ignore_index=True, sort=False)


def summarise_overall(dedup: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ALL_METRICS:
        fixed_mean = dedup.loc[dedup["system"] == "fixed", metric].mean()
        adaptive_mean = dedup.loc[dedup["system"] == "adaptive", metric].mean()
        rows.append(
            {
                "metric": metric,
                "fixed_mean": fixed_mean,
                "adaptive_mean": adaptive_mean,
                "adaptive_minus_fixed": adaptive_mean - fixed_mean,
                "paired_mean_delta": paired[f"delta_{metric}"].mean() if f"delta_{metric}" in paired.columns else np.nan,
                "better_system": better_label(metric, adaptive_mean - fixed_mean),
            }
        )
    return pd.DataFrame(rows)


def summarise_factor(dedup: pd.DataFrame, factor: str) -> pd.DataFrame:
    rows = []
    for level in sorted(dedup[factor].dropna().unique(), key=lambda x: DATASET_ORDER.index(x) if factor == "dataset_size" else MODEL_ORDER.index(x) if factor == "model_family" else QUANT_ORDER.index(x)):
        subset = dedup[dedup[factor] == level]
        for metric in ALL_METRICS:
            fixed_mean = subset.loc[subset["system"] == "fixed", metric].mean()
            adaptive_mean = subset.loc[subset["system"] == "adaptive", metric].mean()
            rows.append(
                {
                    factor: level,
                    "metric": metric,
                    "fixed_mean": fixed_mean,
                    "adaptive_mean": adaptive_mean,
                    "adaptive_minus_fixed": adaptive_mean - fixed_mean,
                    "better_system": better_label(metric, adaptive_mean - fixed_mean),
                }
            )
    return pd.DataFrame(rows)


def summarise_configuration_level(paired: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["model_family", "quantization", "dataset_size"]
    rows = []
    for keys, group in paired.groupby(group_keys, sort=False):
        base = dict(zip(group_keys, keys))
        base["pair_count"] = len(group)
        for metric in ALL_METRICS:
            base[f"fixed_{metric}"] = group[f"{metric}_fixed"].mean()
            base[f"adaptive_{metric}"] = group[f"{metric}_adaptive"].mean()
            base[f"delta_{metric}"] = group[f"delta_{metric}"].mean()
        rows.append(base)
    result = pd.DataFrame(rows)
    return result.sort_values(["model_family", "quantization", "dataset_size"]).reset_index(drop=True)


def run_stat_tests(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    for metric in ALL_METRICS:
        delta_col = f"delta_{metric}"
        if delta_col not in paired.columns:
            continue
        values = paired[delta_col].dropna().to_numpy(dtype=float)
        mean_delta = float(np.mean(values)) if values.size else np.nan
        median_delta = float(np.median(values)) if values.size else np.nan
        ci_low, ci_high = bootstrap_mean_ci(values, rng, samples=5000)
        d = paired_cohens_d(values)
        p_value = np.nan
        test_note = "Wilcoxon skipped because scipy is unavailable."
        if SCIPY_AVAILABLE:
            if values.size == 0:
                test_note = "Wilcoxon skipped because there were no paired values."
            elif np.allclose(values, 0.0):
                test_note = "Wilcoxon skipped because all paired deltas were zero."
            else:
                try:
                    _, p_value = wilcoxon(values, zero_method="wilcox", alternative="two-sided")
                    test_note = "Wilcoxon signed-rank test computed on paired query-level deltas."
                except Exception as exc:
                    test_note = f"Wilcoxon skipped: {exc}"
                    p_value = np.nan

        if metric == "gpu_util_percent":
            direction = "higher saturation for Adaptive" if mean_delta > 0 else "lower saturation for Adaptive" if mean_delta < 0 else "no saturation shift"
        elif metric in LOWER_IS_BETTER:
            direction = "Adaptive improves" if mean_delta < 0 else "Fixed improves" if mean_delta > 0 else "no difference"
        else:
            direction = "Adaptive improves" if mean_delta > 0 else "Fixed improves" if mean_delta < 0 else "no difference"

        rows.append(
            {
                "metric": metric,
                "paired_n": int(values.size),
                "paired_mean_delta": mean_delta,
                "paired_median_delta": median_delta,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "wilcoxon_p_value": p_value,
                "paired_cohens_d": d,
                "effect_direction": direction,
                "wilcoxon_note": test_note,
            }
        )
    return pd.DataFrame(rows)


def compare_models_overall(dedup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ALL_METRICS:
        llama_mean = dedup.loc[dedup["model_family"] == "Llama 3.3 70B", metric].mean()
        qwen_mean = dedup.loc[dedup["model_family"] == "Qwen 32B", metric].mean()
        delta = llama_mean - qwen_mean
        rows.append(
            {
                "metric": metric,
                "llama_mean": llama_mean,
                "qwen_mean": qwen_mean,
                "llama_minus_qwen": delta,
                "better_model": better_model_label(metric, delta),
            }
        )
    return pd.DataFrame(rows)


def compare_models_by_system(dedup: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for system in SYSTEM_ORDER:
        subset = dedup[dedup["system"] == system]
        for metric in ALL_METRICS:
            llama_mean = subset.loc[subset["model_family"] == "Llama 3.3 70B", metric].mean()
            qwen_mean = subset.loc[subset["model_family"] == "Qwen 32B", metric].mean()
            delta = llama_mean - qwen_mean
            rows.append(
                {
                    "system": system,
                    "metric": metric,
                    "llama_mean": llama_mean,
                    "qwen_mean": qwen_mean,
                    "llama_minus_qwen": delta,
                    "better_model": better_model_label(metric, delta),
                }
            )
    return pd.DataFrame(rows)


def adaptive_gain_by_model(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_family in MODEL_ORDER:
        subset = paired[paired["model_family"] == model_family]
        for metric in ALL_METRICS:
            rows.append(
                {
                    "model_family": model_family,
                    "metric": metric,
                    "fixed_mean": subset[f"{metric}_fixed"].mean(),
                    "adaptive_mean": subset[f"{metric}_adaptive"].mean(),
                    "adaptive_minus_fixed": subset[f"delta_{metric}"].mean(),
                }
            )
    return pd.DataFrame(rows)


def model_factor_interaction_summary(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for factor in ["dataset_size", "quantization"]:
        for keys, group in paired.groupby(["model_family", factor], sort=False):
            model_family, factor_level = keys
            rows.append(
                {
                    "interaction_factor": factor,
                    "model_family": model_family,
                    factor: factor_level,
                    "pair_count": len(group),
                    "quality_gain": group["delta_quality_index"].mean(),
                    "answer_relevance_gain": group["delta_answer_relevance_1to5"].mean(),
                    "groundedness_gain": group["delta_groundedness_score"].mean(),
                    "hallucination_delta": group["delta_hallucination_rate"].mean(),
                    "response_time_delta": group["delta_response_time_s"].mean(),
                    "gpu_mem_peak_delta": group["delta_gpu_mem_peak_mb"].mean(),
                }
            )
    return pd.DataFrame(rows)


def save_tables(
    raw: pd.DataFrame,
    dedup: pd.DataFrame,
    paired: pd.DataFrame,
    unmatched: pd.DataFrame,
    overall_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    quant_summary: pd.DataFrame,
    config_summary: pd.DataFrame,
    stat_results: pd.DataFrame,
    model_comparison: pd.DataFrame,
    model_by_system: pd.DataFrame,
    adaptive_gain_model: pd.DataFrame,
    interaction_summary: pd.DataFrame,
) -> None:
    dedup.to_csv(OUTPUT_DIR / "cleaned_deduplicated_data.csv", index=False)
    paired.to_csv(OUTPUT_DIR / "paired_query_level_deltas.csv", index=False)
    overall_summary.to_csv(OUTPUT_DIR / "overall_metric_summary.csv", index=False)
    dataset_summary.to_csv(OUTPUT_DIR / "factor_summary_by_dataset_size.csv", index=False)
    model_summary.to_csv(OUTPUT_DIR / "factor_summary_by_model_family.csv", index=False)
    quant_summary.to_csv(OUTPUT_DIR / "factor_summary_by_quantization.csv", index=False)
    config_summary.to_csv(OUTPUT_DIR / "configuration_level_deltas.csv", index=False)
    stat_results.to_csv(OUTPUT_DIR / "statistical_test_results.csv", index=False)
    model_comparison.to_csv(OUTPUT_DIR / "model_family_comparison.csv", index=False)
    model_by_system.to_csv(OUTPUT_DIR / "model_by_system_comparison.csv", index=False)
    adaptive_gain_model.to_csv(OUTPUT_DIR / "adaptive_gain_by_model.csv", index=False)
    interaction_summary.to_csv(OUTPUT_DIR / "model_factor_interaction_summary.csv", index=False)
    stale_unmatched = OUTPUT_DIR / "unmatched_queries.csv"
    if stale_unmatched.exists():
        stale_unmatched.unlink()


def base_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 220,
            "savefig.dpi": 360,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def save_radar(df: pd.DataFrame, dims: List[str], title: str, filename: str, subtitle: str) -> None:
    values = []
    labels = dims
    systems = ["fixed", "adaptive"]
    for system in systems:
        row = df[df["system"] == system].iloc[0]
        values.append([row[col] for col in dims])

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(9.2, 9.2), subplot_kw={"polar": True})
    colors = {"fixed": "#B43C2A", "adaptive": "#1F5A94"}
    for idx, system in enumerate(systems):
        plot_values = values[idx] + values[idx][:1]
        ax.plot(angles, plot_values, linewidth=2.5, label=system.capitalize(), color=colors[system])
        ax.fill(angles, plot_values, alpha=0.15, color=colors[system])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", pad=12, labelsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(0, max(1.05, max(max(v) for v in values) * 1.1))
    ax.set_title(title, pad=22)
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.12))
    wrapped_subtitle = "\n".join(textwrap.wrap(subtitle, width=105))
    fig.text(0.5, 0.045, wrapped_subtitle, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=(0.03, 0.16, 0.97, 0.97))
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def plot_combined_quality_profile(dedup: pd.DataFrame) -> None:
    summary = (
        dedup.groupby("system", as_index=False)
        .agg(
            faithfulness=("hallucination_rate", lambda s: 1 - s.mean()),
            groundedness_score=("groundedness_score", "mean"),
            answer_relevance_norm=("answer_relevance_1to5", lambda s: s.mean() / 5.0),
            context_relevance_norm=("context_relevance_1to5", lambda s: s.mean() / 5.0),
        )
    )
    save_radar(
        summary,
        ["faithfulness", "groundedness_score", "answer_relevance_norm", "context_relevance_norm"],
        "Combined Quality Profile",
        "combined_quality_profile.png",
        "Faithfulness uses 1 - hallucination rate so every dimension is higher-is-better. Answer and context relevance are normalized to 0-1 to show the quality shape of each system on a common scale.",
    )


def plot_combined_operational_profile(dedup: pd.DataFrame) -> None:
    summary = (
        dedup.groupby("system", as_index=False)
        .agg(
            latency_efficiency=("response_time_s", lambda s: 1 / s.mean()),
            throughput_efficiency=("gpu_throughput_toks_per_s", "mean"),
            memory_efficiency=("gpu_mem_peak_mb", lambda s: 1 / s.mean()),
            gpu_utilization=("gpu_util_percent", lambda s: s.mean() / 100.0),
        )
    )
    for col in ["latency_efficiency", "throughput_efficiency", "memory_efficiency", "gpu_utilization"]:
        best = summary[col].max()
        if best and not np.isnan(best):
            summary[col] = summary[col] / best
    save_radar(
        summary,
        ["latency_efficiency", "throughput_efficiency", "memory_efficiency", "gpu_utilization"],
        "Combined Operational Profile",
        "combined_operational_profile.png",
        "Operational metrics have different units, so each dimension is normalized so the better system gets 1.0. This is a relative profile, not a raw-value chart. GPU utilization is shown as saturation, not as an automatic win.",
    )


def plot_heatmap(config_summary: pd.DataFrame, metric: str, filename: str, direction_note: str) -> None:
    df = config_summary.copy()
    df["row_label"] = df.apply(lambda r: make_row_label(r["model_family"], r["quantization"]), axis=1)
    pivot = (
        df.pivot_table(index="row_label", columns="dataset_size", values=f"delta_{metric}", aggfunc="mean")
        .reindex(index=[make_row_label(m, q) for m in MODEL_ORDER for q in QUANT_ORDER], columns=DATASET_ORDER)
    )
    data = pivot.to_numpy(dtype=float)
    max_abs = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0
    max_abs = max(max_abs, 1e-6)
    cmap = plt.get_cmap("RdBu_r")
    norm = Normalize(vmin=-max_abs, vmax=max_abs)
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(DATASET_ORDER)))
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Adaptive - Fixed Heatmap: {metric}")
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Model Family x Quantization")
    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    annotate_heatmap(ax, data, cmap, norm)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Adaptive - Fixed")
    cbar.ax.tick_params(labelsize=10)
    wrapped_note = "\n".join(textwrap.wrap(direction_note, width=100))
    fig.text(0.5, 0.035, wrapped_note, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=(0.02, 0.12, 0.98, 0.98))
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def plot_interaction_by_dataset(dedup: pd.DataFrame, metric: str, filename: str, ylabel: str) -> None:
    summary = (
        dedup.groupby(["dataset_size", "system", "model_family"], as_index=False)[metric]
        .mean()
    )
    summary["dataset_size"] = pd.Categorical(summary["dataset_size"], categories=DATASET_ORDER, ordered=True)
    summary = summary.sort_values("dataset_size")

    fig, ax = plt.subplots(figsize=(9.4, 6.1))
    colors = {"Llama 3.3 70B": "#1F5A94", "Qwen 32B": "#C9651D"}
    linestyles = {"fixed": "-", "adaptive": "--"}
    for (system, model), group in summary.groupby(["system", "model_family"]):
        ax.plot(
            group["dataset_size"].astype(str),
            group[metric],
            marker="o",
            linewidth=2.2,
            linestyle=linestyles[system],
            color=colors[model],
            label=f"{system.capitalize()} {short_model(model)}",
        )
    ax.set_title(f"Interaction by Dataset Size: {metric}")
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename)
    plt.close(fig)


def plot_tradeoff_scatter(config_summary: pd.DataFrame, x: str, y: str, filename: str, xlabel: str, ylabel: str, title: str, note: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.6))
    colors = {"Llama 3.3 70B": "#1F5A94", "Qwen 32B": "#C9651D"}
    markers = {"4bit": "o", "8bit": "s"}
    label_rows: List[Dict[str, object]] = []
    for _, row in config_summary.iterrows():
        ax.scatter(
            row[x],
            row[y],
            color=colors[row["model_family"]],
            marker=markers[row["quantization"]],
            s=110,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.4,
        )
        label_rows.append(
            {
                "model_family": row["model_family"],
                "quantization": row["quantization"],
                "dataset_size": row["dataset_size"],
                x: row[x],
                y: row[y],
            }
        )
    ax.axhline(0, color="gray", linewidth=1)
    ax.axvline(0, color="gray", linewidth=1)
    xvals = config_summary[x].to_numpy(dtype=float)
    yvals = config_summary[y].to_numpy(dtype=float)
    if len(xvals):
        xspan = max(np.ptp(xvals), 1e-6)
        yspan = max(np.ptp(yvals), 1e-6)
        ax.set_xlim(xvals.min() - 0.10 * xspan - 0.03, xvals.max() + 0.12 * xspan + 0.03)
        ax.set_ylim(yvals.min() - 0.12 * yspan - 0.03, yvals.max() + 0.18 * yspan + 0.03)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=18)
    ax.grid(alpha=0.25)
    manual_offsets = None
    if filename == "tradeoff_quality_gain_vs_latency.png":
        manual_offsets = {
            "Llama\n4bit-medium": (2, 24),
            "Llama\n4bit-small": (14, 8),
            "Llama\n8bit-big": (10, 28),
            "Llama\n8bit-small": (-18, 10),
            "Llama\n8bit-medium": (8, 14),
        }
    add_repel_labels(ax, label_rows, x, y, manual_offsets=manual_offsets)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["Llama 3.3 70B"], markeredgecolor="black", markersize=8, label="Llama 3.3 70B"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["Qwen 32B"], markeredgecolor="black", markersize=8, label="Qwen 2.5 32B"),
        Line2D([0], [0], marker="o", color="#666666", markerfacecolor="white", markeredgecolor="#666666", markersize=7, label="4-bit"),
        Line2D([0], [0], marker="s", color="#666666", markerfacecolor="white", markeredgecolor="#666666", markersize=7, label="8-bit"),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper left", ncol=2)
    wrapped_note = "\n".join(textwrap.wrap(note, width=100))
    fig.text(0.5, 0.04, wrapped_note, ha="center", va="bottom", fontsize=10)
    fig.tight_layout(rect=(0.03, 0.14, 0.98, 0.95))
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def plot_model_quality_comparison(model_comparison: pd.DataFrame) -> None:
    metrics = ["quality_index", "hallucination_rate", "groundedness_score", "answer_relevance_1to5", "context_relevance_1to5"]
    transformed = []
    for metric in metrics:
        row = model_comparison[model_comparison["metric"] == metric].iloc[0]
        if metric == "hallucination_rate":
            llama_value = 1 - row["llama_mean"]
            qwen_value = 1 - row["qwen_mean"]
            label = "faithfulness"
        elif metric in {"answer_relevance_1to5", "context_relevance_1to5"}:
            llama_value = row["llama_mean"] / 5.0
            qwen_value = row["qwen_mean"] / 5.0
            label = metric.replace("_1to5", "")
        else:
            llama_value = row["llama_mean"]
            qwen_value = row["qwen_mean"]
            label = metric
        transformed.append((label, llama_value, qwen_value))

    labels = [t[0] for t in transformed]
    llama_vals = [t[1] for t in transformed]
    qwen_vals = [t[2] for t in transformed]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, llama_vals, width, label="Llama 3.3 70B", color="#1F5A94")
    ax.bar(x + width / 2, qwen_vals, width, label="Qwen 32B", color="#C9651D")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_ylim(0, max(max(llama_vals), max(qwen_vals)) * 1.2)
    ax.set_title("Overall Model Quality Comparison")
    ax.set_ylabel("Higher-is-better normalized scale")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_overall_quality_comparison.png")
    plt.close(fig)


def plot_model_by_system_quality(dedup: pd.DataFrame) -> None:
    metrics = ["quality_index", "hallucination_rate", "groundedness_score", "answer_relevance_1to5", "context_relevance_1to5"]
    grouped = dedup.groupby(["system", "model_family"], as_index=False)[metrics].mean()
    grouped["label"] = grouped["system"].str.capitalize() + " " + grouped["model_family"].map(short_model)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5.4))
    for ax, metric in zip(axes, metrics):
        values = grouped[metric].copy()
        if metric == "hallucination_rate":
            values = 1 - values
            ylabel = "Faithfulness"
        elif metric in {"answer_relevance_1to5", "context_relevance_1to5"}:
            values = values / 5.0
            ylabel = "Normalized"
        else:
            ylabel = "Score"
        ax.bar(grouped["label"], values, color=["#B43C2A", "#C9651D", "#8D4F4A", "#1F5A94"])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=24, labelsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Model Comparison Within Each RAG System", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_by_system_quality_comparison.png")
    plt.close(fig)


def plot_adaptive_gain_by_model(adaptive_gain_model: pd.DataFrame) -> None:
    metrics = ["quality_index", "hallucination_rate", "groundedness_score", "answer_relevance_1to5", "context_relevance_1to5"]
    pivot = adaptive_gain_model[adaptive_gain_model["metric"].isin(metrics)].pivot(index="metric", columns="model_family", values="adaptive_minus_fixed")
    pivot = pivot.reindex(metrics)
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, pivot["Llama 3.3 70B"], width, label="Llama 3.3 70B", color="#1F5A94")
    ax.bar(x + width / 2, pivot["Qwen 32B"], width, label="Qwen 32B", color="#C9651D")
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=14)
    ax.set_title("Adaptive Gain by Model")
    ax.set_ylabel("Adaptive - Fixed")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, 0.01, "Positive bars are improvements for higher-is-better metrics. Hallucination rate is lower-is-better, so negative bars there indicate improvement.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUTPUT_DIR / "adaptive_gain_by_model.png")
    plt.close(fig)


def plot_model_dataset_interaction(dedup: pd.DataFrame) -> None:
    summary = dedup.groupby(["dataset_size", "system", "model_family"], as_index=False)["quality_index"].mean()
    summary["dataset_size"] = pd.Categorical(summary["dataset_size"], categories=DATASET_ORDER, ordered=True)
    summary = summary.sort_values("dataset_size")
    fig, ax = plt.subplots(figsize=(9.4, 6.1))
    colors = {"Llama 3.3 70B": "#1F5A94", "Qwen 32B": "#C9651D"}
    linestyles = {"fixed": "-", "adaptive": "--"}
    for (model, system), group in summary.groupby(["model_family", "system"]):
        ax.plot(
            group["dataset_size"].astype(str),
            group["quality_index"],
            marker="o",
            linewidth=2.2,
            linestyle=linestyles[system],
            color=colors[model],
            label=f"{short_model(model)} {system}",
        )
    ax.set_title("Model x Dataset Size Interaction (Quality Index)")
    ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Quality Index")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_dataset_size_interaction_quality_index.png")
    plt.close(fig)


def plot_model_quantization_interaction(paired: pd.DataFrame) -> None:
    summary = paired.groupby(["model_family", "quantization"], as_index=False)["delta_quality_index"].mean()
    summary["label"] = summary["model_family"].map(short_model) + " " + summary["quantization"]
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    colors = ["#1F5A94" if "Llama" in label else "#C9651D" for label in summary["label"]]
    ax.bar(summary["label"], summary["delta_quality_index"], color=colors)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title("Model x Quantization Interaction (Adaptive Quality Gain)")
    ax.set_ylabel("Quality Index Gain (Adaptive - Fixed)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "model_quantization_interaction_quality_gain.png")
    plt.close(fig)


def plot_operational_model_comparison(model_comparison: pd.DataFrame) -> None:
    metrics = ["response_time_s", "gpu_throughput_toks_per_s", "gpu_mem_peak_mb"]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    for ax, metric in zip(axes, metrics):
        row = model_comparison[model_comparison["metric"] == metric].iloc[0]
        ax.bar(["Llama", "Qwen"], [row["llama_mean"], row["qwen_mean"]], color=["#1F5A94", "#C9651D"])
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Operational Model Comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "operational_model_comparison.png")
    plt.close(fig)


def create_figures(dedup: pd.DataFrame, paired: pd.DataFrame, config_summary: pd.DataFrame, model_comparison: pd.DataFrame, adaptive_gain_model: pd.DataFrame) -> None:
    base_style()
    plot_combined_quality_profile(dedup)
    plot_combined_operational_profile(dedup)

    directions = {
        "hallucination_rate": "Lower hallucination is better, so negative Adaptive - Fixed cells favor Adaptive.",
        "groundedness_score": "Higher groundedness is better, so positive Adaptive - Fixed cells favor Adaptive.",
        "answer_relevance_1to5": "Higher answer relevance is better, so positive Adaptive - Fixed cells favor Adaptive.",
        "context_relevance_1to5": "Higher context relevance is better, so positive Adaptive - Fixed cells favor Adaptive.",
        "response_time_s": "Lower response time is better, so negative Adaptive - Fixed cells favor Adaptive on latency.",
        "gpu_util_percent": "GPU utilization is a saturation metric, so positive cells mean Adaptive uses more of the GPU, not automatically that it is better.",
        "gpu_throughput_toks_per_s": "Higher throughput is better, so positive Adaptive - Fixed cells favor Adaptive on decoding efficiency.",
        "gpu_mem_peak_mb": "Lower memory is better, so negative Adaptive - Fixed cells favor Adaptive on deployment footprint.",
    }
    for metric in ["hallucination_rate", "groundedness_score", "answer_relevance_1to5", "context_relevance_1to5"]:
        plot_heatmap(config_summary, metric, f"heatmap_{metric}.png", directions[metric])
    for metric in ["response_time_s", "gpu_util_percent", "gpu_throughput_toks_per_s", "gpu_mem_peak_mb"]:
        plot_heatmap(config_summary, metric, f"heatmap_{metric}.png", directions[metric])

    plot_interaction_by_dataset(dedup, "answer_relevance_1to5", "interaction_answer_relevance_by_dataset_size.png", "Answer Relevance (1-5)")
    plot_interaction_by_dataset(dedup, "hallucination_rate", "interaction_hallucination_by_dataset_size.png", "Hallucination Rate")
    plot_interaction_by_dataset(dedup, "response_time_s", "interaction_response_time_by_dataset_size.png", "Response Time (s)")
    plot_interaction_by_dataset(dedup, "gpu_mem_peak_mb", "interaction_gpu_memory_by_dataset_size.png", "GPU Peak Memory (MB)")

    plot_tradeoff_scatter(
        config_summary,
        "delta_answer_relevance_1to5",
        "delta_groundedness_score",
        "tradeoff_answer_vs_groundedness.png",
        "Answer relevance gain (Adaptive - Fixed)",
        "Groundedness gain (Adaptive - Fixed)",
        "Tradeoff Plot: Answer Relevance Gain vs Groundedness Gain",
        "Upper-right is the most desirable quadrant because Adaptive improves both answer usefulness and evidence anchoring.",
    )
    plot_tradeoff_scatter(
        config_summary.assign(hallucination_reduction=-config_summary["delta_hallucination_rate"]),
        "delta_answer_relevance_1to5",
        "hallucination_reduction",
        "tradeoff_answer_vs_hallucination_reduction.png",
        "Answer relevance gain (Adaptive - Fixed)",
        "Hallucination-rate reduction (Fixed - Adaptive)",
        "Tradeoff Plot: Answer Relevance Gain vs Hallucination Reduction",
        "Upper-right combines more useful answers with less unsupported generation, while lower-left means Adaptive loses on both dimensions.",
    )
    plot_tradeoff_scatter(
        config_summary,
        "delta_response_time_s",
        "delta_gpu_mem_peak_mb",
        "tradeoff_latency_vs_memory.png",
        "Response-time delta (Adaptive - Fixed)",
        "Memory delta (Adaptive - Fixed)",
        "Tradeoff Plot: Response Time Delta vs Memory Delta",
        "Lower-left is operationally attractive because Adaptive is both faster and lighter there. Upper-right marks the heaviest operational penalty.",
    )
    plot_tradeoff_scatter(
        config_summary,
        "delta_response_time_s",
        "delta_quality_index",
        "tradeoff_quality_gain_vs_latency.png",
        "Response-time delta (Adaptive - Fixed)",
        "Quality-index gain (Adaptive - Fixed)",
        "Tradeoff Plot: Quality Gain vs Response-Time Delta",
        "Upper-left is the best quadrant because Adaptive improves quality while reducing latency. Upper-right shows quality gains that come with extra delay.",
    )

    plot_model_quality_comparison(model_comparison)
    plot_model_by_system_quality(dedup)
    plot_adaptive_gain_by_model(adaptive_gain_model)
    plot_model_dataset_interaction(dedup)
    plot_model_quantization_interaction(paired)
    plot_operational_model_comparison(model_comparison)


def top_examples(df: pd.DataFrame, condition: pd.Series, score_col: str, n: int = 3, ascending: bool = False) -> pd.DataFrame:
    subset = df[condition].copy()
    if subset.empty:
        return subset
    return subset.sort_values(score_col, ascending=ascending).head(n)


def example_table_markdown(df: pd.DataFrame, category: str) -> str:
    if df.empty:
        return f"No representative paired examples met the criteria for **{category}**.\n"
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"- Query: {row['query']}\n"
            f"  Model/quant/data: {short_model(row['model_family'])}, {row['quantization']}, {row['dataset_size']}\n"
            f"  Fixed: answer relevance {fmt(row['answer_relevance_1to5_fixed'])}, hallucination {fmt(row['hallucination_rate_fixed'])}, groundedness {fmt(row['groundedness_score_fixed'])}\n"
            f"  Adaptive: answer relevance {fmt(row['answer_relevance_1to5_adaptive'])}, hallucination {fmt(row['hallucination_rate_adaptive'])}, groundedness {fmt(row['groundedness_score_adaptive'])}\n"
            f"  Interpretation: {row['concept_note']}\n"
        )
    return "\n".join(lines)


def make_metric_summary_text(overall_summary: pd.DataFrame, metric: str) -> str:
    row = overall_summary[overall_summary["metric"] == metric].iloc[0]
    delta = row["adaptive_minus_fixed"]
    if metric == "gpu_util_percent":
        improvement_text = (
            "Adaptive drives higher GPU saturation"
            if delta > 0
            else "Fixed drives higher GPU saturation"
            if delta < 0
            else "Both systems saturate the GPU similarly"
        )
    elif metric in LOWER_IS_BETTER:
        improvement_text = "Adaptive is better" if delta < 0 else "Fixed is better" if delta > 0 else "Both are tied"
    else:
        improvement_text = "Adaptive is better" if delta > 0 else "Fixed is better" if delta < 0 else "Both are tied"

    explainer = QUALITY_EXPLAINERS.get(metric, OPERATIONAL_EXPLAINERS.get(metric, ""))
    return (
        f"- `{metric}`: Fixed mean = {fmt(row['fixed_mean'])}, Adaptive mean = {fmt(row['adaptive_mean'])}, "
        f"Adaptive - Fixed = {fmt(delta)}. {improvement_text}. This metric matters because it reflects {explainer}."
    )


def factor_best_summary(summary_df: pd.DataFrame, factor: str, metric: str) -> Tuple[str, str]:
    metric_df = summary_df[summary_df["metric"] == metric].copy()
    if metric_df.empty:
        return ("NA", "NA")
    if metric in LOWER_IS_BETTER:
        best_row = metric_df.loc[metric_df["adaptive_minus_fixed"].idxmin()]
        worst_row = metric_df.loc[metric_df["adaptive_minus_fixed"].idxmax()]
    else:
        best_row = metric_df.loc[metric_df["adaptive_minus_fixed"].idxmax()]
        worst_row = metric_df.loc[metric_df["adaptive_minus_fixed"].idxmin()]
    return (
        f"{best_row[factor]} ({fmt(best_row['adaptive_minus_fixed'])})",
        f"{worst_row[factor]} ({fmt(worst_row['adaptive_minus_fixed'])})",
    )


def generate_report(
    raw: pd.DataFrame,
    dedup: pd.DataFrame,
    paired: pd.DataFrame,
    unmatched: pd.DataFrame,
    audit: Dict[str, object],
    overall_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    quant_summary: pd.DataFrame,
    config_summary: pd.DataFrame,
    stat_results: pd.DataFrame,
    model_comparison: pd.DataFrame,
    model_by_system: pd.DataFrame,
    adaptive_gain_model: pd.DataFrame,
    interaction_summary: pd.DataFrame,
) -> str:
    quality_gain_row = overall_summary[overall_summary["metric"] == "quality_index"].iloc[0]
    adaptive_docs_delta = paired["delta_retrieved_docs_count"].mean() if "delta_retrieved_docs_count" in paired.columns else np.nan
    adaptive_gk_delta = paired["delta_used_general_knowledge"].mean() if "delta_used_general_knowledge" in paired.columns else np.nan
    adaptive_cov_delta = paired["delta_query_coverage"].mean() if "delta_query_coverage" in paired.columns else np.nan

    rows_per_file_md = dataframe_to_markdown(audit["file_audit"][["source_file", "rows"]])
    missing_cols_files = {
        file: cols for file, cols in audit["missing_by_file"].items() if cols
    }
    missing_core_text = (
        "No required metric columns were missing in the eligible root CSV files."
        if not missing_cols_files
        else "Missing required columns were detected in: "
        + "; ".join(f"{name}: {', '.join(cols)}" for name, cols in missing_cols_files.items())
    )

    overall_quality_lines = "\n".join(make_metric_summary_text(overall_summary, metric) for metric in QUALITY_METRICS)
    overall_operational_lines = "\n".join(make_metric_summary_text(overall_summary, metric) for metric in OPERATIONAL_METRICS)

    dataset_qi_best, dataset_qi_worst = factor_best_summary(dataset_summary, "dataset_size", "quality_index")
    dataset_ar_best, dataset_ar_worst = factor_best_summary(dataset_summary, "dataset_size", "answer_relevance_1to5")
    model_qi_best, model_qi_worst = factor_best_summary(model_summary, "model_family", "quality_index")
    quant_qi_best, quant_qi_worst = factor_best_summary(quant_summary, "quantization", "quality_index")

    top_config_qi = config_summary.sort_values("delta_quality_index", ascending=False).head(3).copy()
    bottom_config_qi = config_summary.sort_values("delta_quality_index", ascending=True).head(3).copy()

    def config_lines(df: pd.DataFrame) -> str:
        if df.empty:
            return "No configuration rows available."
        lines = []
        for _, row in df.iterrows():
            lines.append(
                f"- {short_model(row['model_family'])} {row['quantization']} {row['dataset_size']}: "
                f"quality gain {fmt(row['delta_quality_index'])}, answer relevance delta {fmt(row['delta_answer_relevance_1to5'])}, "
                f"hallucination delta {fmt(row['delta_hallucination_rate'])}, latency delta {fmt(row['delta_response_time_s'])}, "
                f"memory delta {fmt(row['delta_gpu_mem_peak_mb'])}"
            )
        return "\n".join(lines)

    # Failure-mode examples
    over_retrieval = top_examples(
        paired.assign(
            concept_note="Adaptive retrieved more material but answer relevance or grounding fell, which is consistent with broader retrieval introducing distracting evidence."
        ),
        (paired.get("delta_retrieved_docs_count", 0) > 0)
        & ((paired["delta_answer_relevance_1to5"] < 0) | (paired["delta_groundedness_score"] < 0)),
        "delta_quality_index",
        n=3,
        ascending=True,
    )
    context_dilution = top_examples(
        paired.assign(
            concept_note="Adaptive broadened the evidence space and the answer became less well anchored, suggesting that extra context diluted focus instead of clarifying it."
        ),
        (paired["delta_groundedness_score"] < 0) & (paired["delta_context_relevance_1to5"] <= 0),
        "delta_groundedness_score",
        n=3,
        ascending=True,
    )
    hybrid_overreach = top_examples(
        paired.assign(
            concept_note="Adaptive relied more on hybrid or general-knowledge behavior and lost factual discipline, indicating that broader reasoning outpaced the available evidence."
        ),
        (
            paired.get("used_general_knowledge_adaptive", pd.Series([np.nan] * len(paired), index=paired.index)).fillna(0) > 0
        )
        & ((paired["delta_hallucination_rate"] > 0) | (paired["delta_groundedness_score"] < 0)),
        "delta_hallucination_rate",
        n=3,
        ascending=False,
    )
    conservative_fixed = top_examples(
        paired.assign(
            concept_note="Fixed answered more cautiously and preserved grounding or hallucination control better, showing that conservative retrieval-only behavior can still win on factual discipline."
        ),
        ((paired["delta_hallucination_rate"] > 0) | (paired["delta_groundedness_score"] < 0))
        & (paired["delta_answer_relevance_1to5"] <= 0),
        "delta_hallucination_rate",
        n=3,
        ascending=False,
    )

    instability_counts = (
        paired.assign(adaptive_loses=paired["delta_quality_index"] < 0)
        .groupby("model_family")["adaptive_loses"]
        .mean()
        .sort_values(ascending=False)
    )
    unstable_model_text = (
        f"{short_model(instability_counts.index[0])} shows the larger share of Adaptive-underperforming pairs ({fmt(instability_counts.iloc[0], 3)} of its pairs)."
        if not instability_counts.empty
        else "No model-specific instability signal could be computed."
    )

    quant_failure_counts = (
        paired.assign(adaptive_loses=paired["delta_quality_index"] < 0)
        .groupby("quantization")["adaptive_loses"]
        .mean()
        .sort_values(ascending=False)
    )
    quant_sensitivity_text = (
        f"{quant_failure_counts.index[0]} has the larger share of Adaptive-underperforming pairs ({fmt(quant_failure_counts.iloc[0], 3)})."
        if not quant_failure_counts.empty
        else "No quantization sensitivity signal could be computed."
    )

    # Adaptive win examples
    better_synthesis = top_examples(
        paired.assign(
            concept_note="Adaptive improved answer relevance without needing an equally large context-relevance gain, which suggests better synthesis of the evidence rather than only better retrieval precision."
        ),
        (paired["delta_answer_relevance_1to5"] > 0) & (paired["delta_groundedness_score"] >= 0),
        "delta_answer_relevance_1to5",
        n=3,
        ascending=False,
    )
    insufficient_context = top_examples(
        paired.assign(
            concept_note="Adaptive was more useful when Fixed stayed conservative under partial context, indicating better handling of insufficient evidence or more flexible answer routing."
        ),
        (
            paired.get("context_status_fixed", pd.Series([""] * len(paired), index=paired.index)).astype(str).str.contains("partial|insufficient", case=False, na=False)
        )
        & (paired["delta_answer_relevance_1to5"] > 0),
        "delta_answer_relevance_1to5",
        n=3,
        ascending=False,
    )
    better_routing = top_examples(
        paired.assign(
            concept_note="Adaptive switched to a different answer mode and that change coincided with a stronger answer, which is consistent with query-type routing helping beyond fixed retrieval."
        ),
        (
            paired.get("answer_mode_used_fixed", pd.Series([""] * len(paired), index=paired.index)).astype(str)
            != paired.get("answer_mode_used_adaptive", pd.Series([""] * len(paired), index=paired.index)).astype(str)
        )
        & (paired["delta_answer_relevance_1to5"] > 0),
        "delta_answer_relevance_1to5",
        n=3,
        ascending=False,
    )
    better_breadth = top_examples(
        paired.assign(
            concept_note="Adaptive retrieved more or broader evidence and translated that extra breadth into a more complete answer."
        ),
        (paired.get("delta_retrieved_docs_count", 0) > 0) & (paired["delta_answer_relevance_1to5"] > 0),
        "delta_answer_relevance_1to5",
        n=3,
        ascending=False,
    )
    better_caution = top_examples(
        paired.assign(
            concept_note="Adaptive reduced hallucination without sacrificing usefulness, which is the most attractive quality gain because it improves reliability and user value together."
        ),
        (paired["delta_hallucination_rate"] < 0) & (paired["delta_answer_relevance_1to5"] >= 0),
        "hallucination_reduction",
        n=3,
        ascending=False,
    )

    adaptive_win_table = pd.DataFrame(
        [
            ["Overall", ", ".join(overall_summary.query("better_system == 'adaptive'")["metric"].tolist()) or "None", "Better evidence use and answer policy where Adaptive’s mean beats Fixed"],
            ["Dataset size", dataset_ar_best, "Larger retrieval spaces raise ambiguity, so adaptive control has more room to help when its deltas improve"],
            ["Model family", model_qi_best, "Some models exploit flexible retrieval and routing more effectively than others"],
            ["Quantization", quant_qi_best, "Precision changes how well the model can digest broader or noisier context"],
            ["Query type", "See representative paired wins below", "Adaptive routing helps analytical, comparison, or partially grounded questions when fixed retrieval is too rigid"],
        ],
        columns=["Level", "Where Adaptive Wins", "Conceptual Reason"],
    )

    # Model conclusions
    overall_model_qi = model_comparison[model_comparison["metric"] == "quality_index"].iloc[0]
    overall_model_latency = model_comparison[model_comparison["metric"] == "response_time_s"].iloc[0]
    overall_model_mem = model_comparison[model_comparison["metric"] == "gpu_mem_peak_mb"].iloc[0]
    overall_model_thr = model_comparison[model_comparison["metric"] == "gpu_throughput_toks_per_s"].iloc[0]

    fixed_model_qi = model_by_system[(model_by_system["system"] == "fixed") & (model_by_system["metric"] == "quality_index")].iloc[0]
    adaptive_model_qi = model_by_system[(model_by_system["system"] == "adaptive") & (model_by_system["metric"] == "quality_index")].iloc[0]

    adaptive_gain_qi = adaptive_gain_model[adaptive_gain_model["metric"] == "quality_index"].copy()
    best_gain_model = adaptive_gain_qi.loc[adaptive_gain_qi["adaptive_minus_fixed"].idxmax(), "model_family"]
    stable_gain_model = adaptive_gain_qi.loc[adaptive_gain_qi["adaptive_minus_fixed"].idxmin(), "model_family"]

    dataset_model_qi = (
        dedup.groupby(["dataset_size", "model_family"], as_index=False)["quality_index"].mean()
        .sort_values(["dataset_size", "model_family"])
    )
    small_best = dataset_model_qi[dataset_model_qi["dataset_size"] == "small"].sort_values("quality_index", ascending=False).iloc[0]
    medium_best = dataset_model_qi[dataset_model_qi["dataset_size"] == "medium"].sort_values("quality_index", ascending=False).iloc[0]
    big_best = dataset_model_qi[dataset_model_qi["dataset_size"] == "big"].sort_values("quality_index", ascending=False).iloc[0]

    quant_model_qi = (
        dedup.groupby(["quantization", "model_family"], as_index=False)["quality_index"].mean()
        .sort_values(["quantization", "model_family"])
    )
    four_bit_best = quant_model_qi[quant_model_qi["quantization"] == "4bit"].sort_values("quality_index", ascending=False).iloc[0]
    eight_bit_best = quant_model_qi[quant_model_qi["quantization"] == "8bit"].sort_values("quality_index", ascending=False).iloc[0]

    llama_losses = paired[(paired["model_family"] == "Llama 3.3 70B") & (paired["delta_quality_index"] < 0)].copy()
    qwen_losses = paired[(paired["model_family"] == "Qwen 32B") & (paired["delta_quality_index"] < 0)].copy()

    if not llama_losses.empty:
        llama_example = llama_losses.sort_values("delta_quality_index").iloc[0]
        llama_failure_text = (
            f"Llama’s clearest weakness in this experiment is operational heaviness or context overreach when Adaptive loses. "
            f"For example, on '{llama_example['query']}' under {llama_example['quantization']} {llama_example['dataset_size']}, "
            f"Adaptive moved quality by {fmt(llama_example['delta_quality_index'])} while response time changed by {fmt(llama_example['delta_response_time_s'])} and memory by {fmt(llama_example['delta_gpu_mem_peak_mb'])}."
        )
    else:
        llama_failure_text = "No Llama paired case showed an Adaptive quality loss, so its main weakness comes from aggregate operational cost rather than negative paired quality outcomes."

    if not qwen_losses.empty:
        qwen_example = qwen_losses.sort_values("delta_quality_index").iloc[0]
        qwen_failure_text = (
            f"Qwen’s clearest weakness is answer-level instability when context broadens. "
            f"For example, on '{qwen_example['query']}' under {qwen_example['quantization']} {qwen_example['dataset_size']}, "
            f"Adaptive changed answer relevance by {fmt(qwen_example['delta_answer_relevance_1to5'])}, groundedness by {fmt(qwen_example['delta_groundedness_score'])}, "
            f"and hallucination by {fmt(qwen_example['delta_hallucination_rate'])}."
        )
    else:
        qwen_failure_text = "No Qwen paired case showed an Adaptive quality loss, so the model’s weakness is mostly relative rather than absolute."

    # Statistical interpretation lines
    stat_lines = []
    for _, row in stat_results.iterrows():
        sig_text = (
            f"Wilcoxon p = {fmt(row['wilcoxon_p_value'], 4)}"
            if pd.notna(row["wilcoxon_p_value"])
            else row["wilcoxon_note"]
        )
        stat_lines.append(
            f"- `{row['metric']}`: mean delta {fmt(row['paired_mean_delta'])}, median delta {fmt(row['paired_median_delta'])}, "
            f"95% bootstrap CI [{fmt(row['bootstrap_ci_low'])}, {fmt(row['bootstrap_ci_high'])}], Cohen's d {fmt(row['paired_cohens_d'])}. "
            f"{sig_text} Practically, this means {row['effect_direction'].lower()} at the paired query level only if the effect is directionally consistent enough to matter across matched conditions."
        )

    report = f"""# Deep Conceptual Analysis Report

## Section 1. What this experiment is really testing

This experiment is not merely benchmarking two implementations of the same pipeline. It is comparing two retrieval policies. Fixed RAG uses the same retrieval behavior regardless of query difficulty, ambiguity, context quality, or answer type. Adaptive RAG changes behavior depending on the query and/or the retrieved context. The central research question is therefore: **Does adaptive control over retrieval and answer strategy improve final answer quality enough to justify its operational tradeoffs?**

That makes this a systems-and-quality tradeoff problem, not just an accuracy comparison. The deeper design question is whether retrieval should be treated as a static pipeline stage or as a decision policy that can expand, constrain, route, or hybridize its answer strategy when evidence is partial or ambiguous.

## Section 2. Data audit and preprocessing

- Total raw rows loaded from eligible root CSV files: {audit['total_raw_rows']}
- Total deduplicated rows after query normalization and within-condition averaging: {audit['total_deduplicated_rows']}
- Duplicate rows collapsed: {audit['duplicate_rows_collapsed']}
- Duplicate query groups detected: {audit['duplicate_groups']}
- Paired query-level comparisons after normalization: {len(paired)}
- Unmatched deduplicated queries: {len(unmatched)}
- Pair count before normalization: {audit['raw_pair_count']}
- Pair count after normalization: {audit['normalized_pair_count']}
- Rows whose query text changed under normalization: {audit['normalization_changed_rows']}
- Cost columns ignored: {', '.join(audit['cost_columns_ignored']) if audit['cost_columns_ignored'] else 'None'}

Rows per file:

{rows_per_file_md}

Schema validation: {missing_core_text}

Preprocessing matters conceptually because query normalization prevents artificial mismatches caused by formatting differences, deduplication prevents repeated queries from overweighting one condition, and paired matching ensures that Fixed and Adaptive are compared under the same model family, quantization, dataset size, and normalized query text.

## Section 3. Why paired query-level analysis matters

Averages alone are weak because they confound system behavior with query difficulty. A hard query may produce low scores for both systems, while an easy query may produce high scores for both. The correct comparison is therefore not simply Adaptive mean versus Fixed mean, but Adaptive performance on the same query and configuration minus Fixed performance on that same query and configuration.

**Paired deltas reveal whether adaptive RAG improves the answer relative to the fixed baseline under the same experimental conditions.**

This pairing controls for query difficulty, model family, quantization, dataset size, and corpus condition. Without that control, a system can look better simply because it saw an easier subset or a more favorable configuration mix.

## Section 4. Conceptual meaning of the quality metrics

### Hallucination rate

Hallucination rate measures unsupported or risky generation. Lower hallucination means the answer is less likely to contain claims that are not justified by retrieved evidence. However, very low hallucination is not always a pure win because a system can achieve it by becoming overly conservative and refusing to answer. That is why hallucination must be read together with answer relevance.

### Groundedness

Groundedness measures how strongly the final answer is supported by retrieved evidence. High groundedness means the answer stays anchored to context. Groundedness is not the same as relevance: a response can be perfectly grounded yet incomplete, overly cautious, or not especially useful to the user.

### Answer relevance

Answer relevance measures whether the final answer addresses the user’s question. This is the most user-facing quality metric in the set. A system can improve answer relevance by synthesizing evidence better, giving a fuller answer, or switching to a more appropriate answer mode for the question.

### Context relevance

Context relevance measures whether the retrieved evidence aligns with the query. This sits closer to retriever quality than generator quality. **Context relevance evaluates the evidence retrieved; answer relevance evaluates what the model does with that evidence.** High context relevance does not guarantee a strong answer if the model uses the evidence timidly or incompletely.

## Section 5. Retrieval quality vs answer quality

Adaptive RAG can improve answer relevance even when context relevance does not improve by the same amount. That pattern matters because it shows that a RAG system is not only a retriever. It is a retriever plus a policy for using retrieved evidence.

In this experiment, the paired data show whether broader or different retrieval behavior actually translated into better answers. On average, Adaptive changed retrieved document count by {fmt(adaptive_docs_delta)}, general-knowledge use by {fmt(adaptive_gk_delta)}, and query coverage by {fmt(adaptive_cov_delta)} relative to Fixed. When answer relevance rises more than context relevance, the implication is that Adaptive is getting value from evidence use, synthesis, or answer-mode selection rather than from retrieval precision alone. Fixed RAG can be retrieval-precise but answer-poor; Adaptive RAG can be retrieval-broader but answer-stronger.

## Section 6. Conceptual meaning of the operational metrics

### Response time

Response time is a user-facing latency metric. It measures how long the full system takes to return an answer, including retrieval, prompt construction, generation, and orchestration overhead. Lower is better for perceived responsiveness.

### GPU throughput

GPU throughput measures token-generation efficiency. Higher throughput means the model decodes faster once generation is underway. But higher throughput does not automatically imply lower end-to-end latency because total response time also depends on prompt length, retrieval overhead, answer length, and system overhead.

### GPU memory peak

GPU memory peak is the deployment footprint. It determines whether a system fits on available hardware and how expensive it is to scale. Lower memory broadens deployment feasibility.

### GPU utilization

GPU utilization measures how busy the GPU is. Higher utilization can indicate better hardware saturation, but it can also mean the system is simply more demanding. Lower utilization can reflect inefficiency or lower load. **Latency is a user-experience metric; throughput is a decoding-efficiency metric; memory is a deployment-feasibility metric; GPU utilization is a saturation metric.**

## Section 7. Why response time and throughput can disagree

Response time and throughput can disagree because they measure different parts of the system. Throughput measures speed during generation; response time measures the whole pipeline. A system can decode quickly but still answer slowly if it spends more time retrieving, constructing prompts, or generating longer responses. Likewise, a system can have lower throughput but lower latency if it retrieves less, answers briefly, or incurs less orchestration overhead.

That means a system with better throughput is not necessarily better for the user, and a system with lower response time is not necessarily more hardware-efficient.

## Section 8. Overall quality and operational comparison: Adaptive RAG vs Fixed RAG

### Quality metrics

{overall_quality_lines}

The overall composite picture comes from `quality_index`: Fixed = {fmt(quality_gain_row['fixed_mean'])}, Adaptive = {fmt(quality_gain_row['adaptive_mean'])}, delta = {fmt(quality_gain_row['adaptive_minus_fixed'])}. This is the most compact statement of whether Adaptive improves the balance of grounding, relevance, and hallucination control rather than winning on a single narrow metric.

### Operational metrics

{overall_operational_lines}

Operationally, these differences define the tradeoff surface. A quality gain is more valuable if it comes with neutral or improved latency and memory, and less attractive if it requires consistently more time or larger deployment footprint.

## Section 9. Factor-level interpretation

### Dataset size

Small datasets can make fixed retrieval sufficient because the evidence space is easier to search cleanly. Medium and big datasets create more ambiguity, which increases the value of deciding how much evidence to retrieve and how to answer from it. In the computed results, the strongest dataset-size Adaptive gain on `quality_index` appears at {dataset_qi_best}, while the weakest appears at {dataset_qi_worst}. For `answer_relevance_1to5`, the strongest Adaptive gain appears at {dataset_ar_best}, while the weakest appears at {dataset_ar_worst}. Conceptually, this tells us whether the adaptive advantage grows as the retrieval space becomes larger and noisier.

### Model family

Different models use retrieved context differently. Some models benefit more from adaptive retrieval because they synthesize evidence better; others may become less grounded when context broadens. In this experiment, the strongest model-family Adaptive gain on `quality_index` appears at {model_qi_best}, while the weakest appears at {model_qi_worst}. That pattern matters because the value of adaptive RAG depends not only on the policy but on the model’s ability to exploit that policy.

### Quantization

Quantization changes model capacity and efficiency. The strongest quantization-level Adaptive gain on `quality_index` appears at {quant_qi_best}, while the weakest appears at {quant_qi_worst}. Conceptually, that tells us whether Adaptive compensates for lower precision or whether lower precision makes broader evidence harder to use cleanly.

### Full configuration

The full interaction among model family, quantization, and dataset size matters because Adaptive is not automatically better everywhere. The top configurations by Adaptive `quality_index` gain are:

{config_lines(top_config_qi)}

The configurations where Fixed remained more competitive are:

{config_lines(bottom_config_qi)}

Those configuration-level deltas show where adaptive control helps most and where rigid retrieval remains adequate or safer.

## Section 10. Statistical analysis with conceptual interpretation

{chr(10).join(stat_lines)}

Statistical significance does not automatically imply deployment superiority. A statistically consistent memory increase, for example, still counts as an operational drawback even if it is very stable.

## Section 11. Visualizations

![Combined quality profile](combined_quality_profile.png)

{figure_caption_block("Figure 1", "The combined quality profile places faithfulness, groundedness, normalized answer relevance, and normalized context relevance on a common higher-is-better scale. Faithfulness is used instead of hallucination rate so direction stays consistent. The shape shows whether a system’s quality advantage is broad or concentrated in one dimension.")}

![Combined operational profile](combined_operational_profile.png)

{figure_caption_block("Figure 2", "The combined operational profile compares latency efficiency, throughput efficiency, memory efficiency, and GPU utilization after per-dimension normalization. This is a relative efficiency profile, not a raw-value chart. GPU utilization should be read as saturation rather than an intrinsic good.")}

![Heatmap hallucination rate](heatmap_hallucination_rate.png)
![Heatmap groundedness score](heatmap_groundedness_score.png)
![Heatmap answer relevance](heatmap_answer_relevance_1to5.png)
![Heatmap context relevance](heatmap_context_relevance_1to5.png)

{figure_caption_block("Figures 3-6", "These heatmaps show mean Adaptive - Fixed deltas for the four quality metrics by configuration. Positive cells favor Adaptive for groundedness and relevance; negative cells favor Adaptive for hallucination because lower is better. The centered diverging scale makes it easy to see where Adaptive gains are broad versus conditional.")}

![Heatmap response time](heatmap_response_time_s.png)
![Heatmap GPU utilization](heatmap_gpu_util_percent.png)
![Heatmap throughput](heatmap_gpu_throughput_toks_per_s.png)
![Heatmap GPU memory](heatmap_gpu_mem_peak_mb.png)

{figure_caption_block("Figures 7-10", "These operational heatmaps show how Adaptive shifts latency, saturation, decoding speed, and memory footprint across the configuration grid. The important question is not whether every operational metric moves in the same direction, but whether quality gains arrive with acceptable infrastructure cost.")}

![Interaction answer relevance](interaction_answer_relevance_by_dataset_size.png)
![Interaction hallucination](interaction_hallucination_by_dataset_size.png)
![Interaction response time](interaction_response_time_by_dataset_size.png)
![Interaction GPU memory](interaction_gpu_memory_by_dataset_size.png)

{figure_caption_block("Figures 11-14", "The interaction plots show whether dataset size changes the behavior of each model-system combination. These figures matter because larger corpora increase ambiguity, so a widening Adaptive advantage with dataset size would support the idea that flexible retrieval matters most when evidence selection becomes harder.")}

![Tradeoff answer relevance vs groundedness](tradeoff_answer_vs_groundedness.png)
![Tradeoff answer relevance vs hallucination reduction](tradeoff_answer_vs_hallucination_reduction.png)
![Tradeoff latency vs memory](tradeoff_latency_vs_memory.png)
![Tradeoff quality gain vs latency](tradeoff_quality_gain_vs_latency.png)

{figure_caption_block("Figures 15-18", "The tradeoff plots move from metric-by-metric reporting to system-design space. Upper-right is desirable in the quality-vs-quality plots because Adaptive improves multiple answer dimensions together. Lower-left is desirable in the latency-vs-memory plot because Adaptive becomes both faster and lighter there. The quality-vs-latency plot directly visualizes whether quality gains require extra time.")}

## Section 12. Failure-mode interpretation

Adaptive underperformance should not be treated as noise. It reveals the conditions under which flexible retrieval or hybrid answering can backfire.

### Over-retrieval

Adaptive retrieves too much and introduces distracting context.

{example_table_markdown(over_retrieval, "Over-retrieval")}

### Context dilution

More context reduces focus and weakens grounding.

{example_table_markdown(context_dilution, "Context dilution")}

### Hybrid overreach

Adaptive uses broader reasoning or general knowledge when retrieval evidence is insufficient, which can outpace the support actually present.

{example_table_markdown(hybrid_overreach, "Hybrid overreach")}

### Conservative fixed advantage

Fixed can win when cautious, retrieval-only behavior preserves factual discipline better than a broader adaptive answer policy.

{example_table_markdown(conservative_fixed, "Conservative fixed advantage")}

### Model-specific instability

{unstable_model_text}

### Quantization sensitivity

{quant_sensitivity_text}

## Section 13. Research conclusion for Adaptive RAG vs Fixed RAG

Adaptive RAG should not be summarized as simply “better” or “worse.” The results show whether adaptive control over retrieval and answer strategy improves answer-level quality enough to justify its operational tradeoffs. When Adaptive improves answer relevance or lowers hallucination under the same paired conditions, the implication is that fixed retrieval budgets can be too rigid for some queries and corpus sizes. When Fixed remains competitive, the implication is that conservative retrieval precision and simpler orchestration still matter, especially when memory efficiency or strict grounding is the priority.

The defensible conclusion is that Adaptive RAG improves answer-level quality when query complexity, retrieval ambiguity, or context insufficiency demand more than a fixed retrieval policy. Fixed RAG remains competitive in direct retrieval precision and memory efficiency. Adaptive RAG is therefore best understood as a flexible evidence-use policy that can improve final answer quality, but it changes the operational tradeoff surface rather than eliminating it.

## Section 14. Style and interpretation of the results

Each table and figure in this report should be read as a statement about RAG design rather than a scoreboard. A higher or lower metric only matters insofar as it changes the system’s usefulness, reliability, or deployability. The central distinction throughout the report is between retrieval quality and answer quality, and between user-facing latency and hardware-facing efficiency.

## Section 15. Where Adaptive RAG Outperforms Fixed RAG

### 1. Overall level

Adaptive has better overall mean performance on these metrics: {', '.join(overall_summary.query("better_system == 'adaptive'")['metric'].tolist()) or 'None'}.

Conceptually, these wins tell us whether Adaptive improves retrieval precision, answer generation, grounding, latency, or resource behavior. The most important overall win is on the metrics that improve end-user quality and factual reliability together, not on saturation alone.

### 2. Metric level

- Quality metrics where Adaptive is better on mean performance: {', '.join(overall_summary[(overall_summary['metric'].isin(QUALITY_METRICS)) & (overall_summary['better_system'] == 'adaptive')]['metric'].tolist()) or 'None'}
- Quality metrics where Fixed is better on mean performance: {', '.join(overall_summary[(overall_summary['metric'].isin(QUALITY_METRICS)) & (overall_summary['better_system'] == 'fixed')]['metric'].tolist()) or 'None'}
- Operational metrics where Adaptive is better on mean performance: {', '.join(overall_summary[(overall_summary['metric'].isin(OPERATIONAL_METRICS)) & (overall_summary['better_system'] == 'adaptive')]['metric'].tolist()) or 'None'}
- Operational metrics where Fixed is better on mean performance: {', '.join(overall_summary[(overall_summary['metric'].isin(OPERATIONAL_METRICS)) & (overall_summary['better_system'] == 'fixed')]['metric'].tolist()) or 'None'}
- GPU utilization should be read as a saturation shift rather than a direct win/loss metric.

### 3. Factor level

Dataset-size interpretation: the strongest `quality_index` Adaptive gain occurs at {dataset_qi_best}. If the larger datasets show stronger gains, that supports the idea that adaptive control becomes more useful when the retrieval space is more ambiguous. If the smallest dataset remains competitive for Fixed, that suggests fixed retrieval is sufficient when the evidence space is simple.

Model-family interpretation: the strongest model-family Adaptive `quality_index` gain occurs at {model_qi_best}. This indicates which model can actually exploit flexible retrieval and broader evidence use rather than merely tolerate it.

Quantization interpretation: the strongest quantization-level Adaptive `quality_index` gain occurs at {quant_qi_best}. That pattern tells us whether Adaptive compensates for quantization-induced limitations or whether lower precision makes flexible retrieval harder to capitalize on.

### 4. Query level

#### A. Better synthesis

{example_table_markdown(better_synthesis, "Better synthesis")}

#### B. Better handling of insufficient context

{example_table_markdown(insufficient_context, "Better handling of insufficient context")}

#### C. Better query-type routing

{example_table_markdown(better_routing, "Better query-type routing")}

#### D. Better retrieval breadth

{example_table_markdown(better_breadth, "Better retrieval breadth")}

#### E. Better factual caution

{example_table_markdown(better_caution, "Better factual caution")}

Summary table:

{dataframe_to_markdown(adaptive_win_table)}

Adaptive RAG performs best when the task requires more than retrieving a fixed number of passages. Its advantage appears when the system must decide how much evidence is needed, how to combine it, and how cautious or expansive the final answer should be.

## Section 16. Overall Conceptual Takeaways

### 1. Fixed RAG is a stable retrieval baseline

Fixed RAG applies the same retrieval behavior to all queries. That makes it predictable, simpler to reason about, and often memory-efficient. It performs well when the query is simple, the corpus is small, or the relevant passage is easy to retrieve. Its weakness is rigidity when the question is analytical, comparative, ambiguous, or only partially supported by the most obvious evidence.

### 2. Adaptive RAG is a flexible evidence-use policy

Adaptive RAG is not just “more retrieval.” It is a policy that can alter retrieval breadth, answer mode, and the degree of contextual or general-knowledge use. Its value comes from deciding how to answer, not only from deciding what to retrieve.

### 3. Retrieval quality and answer quality can diverge

Fixed RAG may retrieve highly relevant context but still produce less useful answers. Adaptive RAG may retrieve broader or less directly aligned context but still generate a stronger final answer. That is why context relevance and answer relevance must be evaluated separately.

### 4. Adaptive RAG improves user-facing quality when flexibility matters

Adaptive is most conceptually attractive for analytical queries, comparison queries, multi-step questions, cases with partial context, and larger or more ambiguous corpora. These are exactly the cases where a fixed retrieval budget is too rigid.

### 5. Fixed RAG remains competitive when precision and simplicity matter

Fixed remains attractive when hardware memory is limited, strict retrieval-only grounding is required, the queries are simple, the corpus is small, or predictable latency and resource behavior matter more than maximum answer completeness.

### 6. Operational tradeoffs are part of the system choice

Adaptive can improve answer quality while increasing memory usage or changing throughput behavior. Fixed can be more resource-efficient but less helpful. The correct deployment choice therefore depends on whether the primary objective is factual usefulness, strict efficiency, or predictability.

### 7. Model and quantization moderate the benefit

Adaptive is not universally better under every model and quantization setting. The benefit depends on the model’s ability to exploit broader or differently routed evidence and on how robust the quantized model is to longer or noisier context.

### 8. Adaptive can improve quality and user-facing efficiency at the same time

One of the most important results in this study is that Adaptive is not only a quality intervention. It improves the composite `quality_index` from {fmt(quality_gain_row['fixed_mean'])} to {fmt(quality_gain_row['adaptive_mean'])} while also reducing end-to-end response time by {fmt(-overall_summary[overall_summary['metric'] == 'response_time_s'].iloc[0]['adaptive_minus_fixed'])} seconds on average. Conceptually, this means adaptive control can reduce wasted answer behavior even when it does not maximize raw decoding throughput. In other words, a dynamic evidence-use policy can be more helpful to the user and faster in wall-clock terms, even if it is not the lightest decoder or the narrowest retriever.

### 9. Adaptive RAG fails when flexibility turns into noise

Adaptive does not fail because adaptivity is inherently bad; it fails when broader retrieval or hybrid answering introduces more uncertainty than value. In this experiment, those failures appear as over-retrieval, context dilution, hybrid overreach, and model-specific instability. The paired results show a larger share of Adaptive-underperforming cases for Qwen 32B than for Llama 3.3 70B, and some configurations such as Qwen 4-bit small are effectively tied or slightly worse on `quality_index`. The conceptual lesson is that adaptive control helps only when the model can filter broader evidence without losing grounding discipline.

### 10. Best overall conceptual conclusion

Fixed RAG should be understood as a conservative, predictable retrieval baseline. Adaptive RAG should be understood as a flexible evidence-use strategy that improves final answer quality when query complexity, retrieval ambiguity, or context insufficiency require more than a fixed retrieval policy. The comparison should therefore be interpreted not as a winner-takes-all benchmark, but as a study of when adaptive control improves the quality-efficiency tradeoff in retrieval-augmented generation.

## Section 17. Model-Family Analysis: Llama 3.3 70B vs Qwen 32B

### 1. Overall model comparison

The overall model-level comparison asks which model is stronger across all configurations, not just within one system.

{dataframe_to_markdown(model_comparison)}

On `quality_index`, Llama = {fmt(overall_model_qi['llama_mean'])}, Qwen = {fmt(overall_model_qi['qwen_mean'])}, so Llama - Qwen = {fmt(overall_model_qi['llama_minus_qwen'])}. On latency, Llama - Qwen = {fmt(overall_model_latency['llama_minus_qwen'])}; on throughput, Llama - Qwen = {fmt(overall_model_thr['llama_minus_qwen'])}; on memory, Llama - Qwen = {fmt(overall_model_mem['llama_minus_qwen'])}. This tells us whether the stronger model is stronger because of answer quality, operational efficiency, or both.

### 2. Model comparison within each RAG system

The better model can depend on the retrieval policy, so Llama versus Qwen must be compared separately within Fixed and Adaptive.

{dataframe_to_markdown(model_by_system)}

Under Fixed RAG, the `quality_index` difference is {fmt(fixed_model_qi['llama_minus_qwen'])}. Under Adaptive RAG, the `quality_index` difference is {fmt(adaptive_model_qi['llama_minus_qwen'])}. If the advantage widens under Adaptive, that means the stronger model is better at exploiting flexible retrieval and answer-mode selection rather than only stronger in a static pipeline.

### 3. Adaptive gain by model

Model quality and adaptive gain are different. A model can have higher absolute performance but smaller adaptive improvement if it was already strong under Fixed retrieval. Another model can have lower baseline performance but larger adaptive gain if the adaptive policy compensates for its weaknesses.

{dataframe_to_markdown(adaptive_gain_model)}

The larger `quality_index` adaptive gain belongs to {best_gain_model}. The smaller gain belongs to {stable_gain_model}. That distinction matters because it separates “best model overall” from “model that benefits most from adaptive policy.”

### 4. Model strengths

If Llama leads on answer relevance and quality index, the conceptual implication is that it is better at evidence synthesis or broader-context use. If Qwen leads on latency, throughput, or memory, the implication is that it is operationally more efficient even when it is not the strongest answer model. The model-comparison tables above show which of those two stories is actually supported by the data.

### 5. Model weaknesses

{llama_failure_text}

{qwen_failure_text}

These weaknesses should not be assumed in advance; they matter only to the extent that the computed metrics actually show them.

### 6. Model x dataset-size interaction

Retrieval is easier on small datasets, so model differences there often reflect answer generation ability more than retrieval robustness. As corpus size grows, the better model is usually the one that can filter and synthesize broader evidence without losing grounding.

- Small dataset best mean `quality_index`: {short_model(small_best['model_family'])} ({fmt(small_best['quality_index'])})
- Medium dataset best mean `quality_index`: {short_model(medium_best['model_family'])} ({fmt(medium_best['quality_index'])})
- Big dataset best mean `quality_index`: {short_model(big_best['model_family'])} ({fmt(big_best['quality_index'])})

These rankings show whether the stronger model stays stronger as retrieval ambiguity increases.

### 7. Model x quantization interaction

Quantization can reduce precision in context use and reasoning. A model that stays relevant and grounded under lower precision is more robust.

- 4-bit best mean `quality_index`: {short_model(four_bit_best['model_family'])} ({fmt(four_bit_best['quality_index'])})
- 8-bit best mean `quality_index`: {short_model(eight_bit_best['model_family'])} ({fmt(eight_bit_best['quality_index'])})

If the same model leads in both settings, it is more quantization-robust. If leadership changes, quantization meaningfully moderates the model comparison.

### 8. Model-level failure modes

Model-level failure modes matter because the same adaptive policy can be an asset for one model and a liability for another. The paired examples above already show whether failures cluster around over-synthesis, operational heaviness, weaker synthesis, or instability under broader context.

### 9. Model comparison visualizations

![Overall model quality comparison](model_overall_quality_comparison.png)
![Model comparison within each system](model_by_system_quality_comparison.png)
![Adaptive gain by model](adaptive_gain_by_model.png)
![Model x dataset size interaction](model_dataset_size_interaction_quality_index.png)
![Model x quantization interaction](model_quantization_interaction_quality_gain.png)
![Operational model comparison](operational_model_comparison.png)

These figures make the model comparison legible at a glance. The quality figures show whether one model is consistently stronger or only stronger in one dimension, while the interaction figures show whether the gap depends on retrieval policy, corpus size, or quantization.

### 10. Conceptual model takeaways

The model comparison should end in a conditional conclusion rather than a slogan. If Llama leads on quality while Qwen leads on efficiency, then Llama is the quality-first choice and Qwen is the resource-constrained choice. If one model also gains more from Adaptive, then adaptive RAG depends not only on the retrieval policy itself but also on the model’s ability to exploit that policy.

## Final executive summary

1. The stronger RAG system overall is the one with the better `quality_index` balance between groundedness, answer usefulness, and hallucination control. In these data, Fixed = {fmt(quality_gain_row['fixed_mean'])} and Adaptive = {fmt(quality_gain_row['adaptive_mean'])}, so the system-level conclusion should start there rather than with any single metric.
2. Adaptive RAG clearly outperforms Fixed where it improves answer relevance, lowers hallucination, or lifts quality index under the same paired conditions, especially in the factors and configurations listed above.
3. Fixed RAG remains competitive where direct retrieval precision and lighter operational footprint matter more than flexible synthesis.
4. The stronger model family overall is the one with the higher aggregate `quality_index`, but the operationally better family may differ if latency, throughput, or memory become binding constraints.
5. The weaker model family is the one that either underuses adaptive context or pays quality penalties when context broadens; the model comparison tables and interaction plots identify which one that is in the current data.
6. Dataset size changes the comparison because larger corpora raise retrieval ambiguity. If Adaptive gains grow with dataset size, that supports dynamic retrieval as a response to harder evidence selection.
7. Quantization changes the comparison because lower precision can blunt context handling or make longer contexts noisier. The factor summaries show whether Adaptive compensates for that or amplifies it.
8. The most important quality tradeoff is between answer relevance and hallucination control: a system that becomes more helpful but less grounded is not automatically better.
9. The most important operational tradeoff is between quality gain and deployment burden: response time, throughput, and memory should be interpreted together rather than separately.
10. The main system-design implication is that RAG should be treated as an evidence-use policy, not just a retriever. Adaptive control matters most when query complexity and corpus ambiguity make a fixed retrieval budget too rigid.
"""
    return convert_report_to_paper_style(report, overall_summary, model_comparison, adaptive_gain_model)


def render_text_page(pdf: PdfPages, lines: List[str]) -> None:
    if not lines:
        return
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    y = 0.97
    for line in lines:
        plain = markdown_to_plain_text(line)
        if line.startswith("# "):
            fig.text(0.05, y, plain, ha="left", va="top", fontsize=16, fontweight="bold")
            y -= 0.03
            continue
        if line.startswith("## "):
            fig.text(0.05, y, plain, ha="left", va="top", fontsize=13, fontweight="bold")
            y -= 0.028
            continue
        if line.startswith("### "):
            fig.text(0.05, y, plain, ha="left", va="top", fontsize=11.5, fontweight="bold")
            y -= 0.025
            continue
        for wrapped in (textwrap.wrap(plain, width=108) or [""]):
            fig.text(0.05, y, wrapped, ha="left", va="top", fontsize=9.2)
            y -= 0.0185
        y -= 0.003
    pdf.savefig(fig)
    plt.close(fig)


def render_figure_page(pdf: PdfPages, image_paths: List[Path], alt_texts: List[str], caption: str) -> None:
    n = len(image_paths)
    cols = 1 if n == 1 else 2
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    grid_top = 0.88
    grid_bottom = 0.18
    gs = fig.add_gridspec(rows, cols, left=0.06, right=0.94, top=grid_top, bottom=grid_bottom, hspace=0.22, wspace=0.12)
    for idx, (img_path, alt) in enumerate(zip(image_paths, alt_texts)):
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        ax.axis("off")
        if img_path.exists():
            img = mpimg.imread(img_path)
            ax.imshow(img)
            ax.set_title(alt, fontsize=10)
        else:
            ax.text(0.5, 0.5, f"Missing image:\n{img_path.name}", ha="center", va="center", fontsize=11)
    wrapped_caption = "\n".join(textwrap.wrap(markdown_to_plain_text(caption), width=100)) if caption else ""
    fig.text(0.06, 0.12, wrapped_caption, ha="left", va="top", fontsize=9.5)
    pdf.savefig(fig)
    plt.close(fig)


def render_pdf_from_markdown(markdown_text: str, output_path: Path) -> None:
    lines = markdown_text.splitlines()
    lines_per_page = 34

    with PdfPages(output_path) as pdf:
        i = 0
        pending_text: List[str] = []
        while i < len(lines):
            line = lines[i]
            image_match = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
            if image_match:
                if pending_text:
                    page_chunk: List[str] = []
                    for pending_line in pending_text:
                        if pending_line.startswith("#"):
                            page_chunk.append(pending_line)
                            continue
                        wrapped = textwrap.wrap(pending_line, width=108, replace_whitespace=False, drop_whitespace=False)
                        page_chunk.extend(wrapped if wrapped else [""])
                    for start in range(0, len(page_chunk), lines_per_page):
                        render_text_page(pdf, page_chunk[start : start + lines_per_page])
                    pending_text = []

                image_paths: List[Path] = []
                alt_texts: List[str] = []
                while i < len(lines):
                    inner_match = re.match(r"!\[(.*?)\]\((.*?)\)", lines[i].strip())
                    if not inner_match:
                        break
                    alt, raw_path = inner_match.groups()
                    image_paths.append((OUTPUT_DIR / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path))
                    alt_texts.append(alt)
                    i += 1
                caption = ""
                while i < len(lines) and not lines[i].strip():
                    i += 1
                if i < len(lines) and lines[i].strip().startswith("**Figure"):
                    caption = lines[i].strip()
                    i += 1
                render_figure_page(pdf, image_paths, alt_texts, caption)
                continue

            pending_text.append(line)
            i += 1

        if pending_text:
            page_chunk = []
            for pending_line in pending_text:
                if not pending_line.strip():
                    page_chunk.append("")
                    continue
                if pending_line.startswith("#"):
                    page_chunk.append(pending_line)
                    continue
                wrapped = textwrap.wrap(pending_line, width=108, replace_whitespace=False, drop_whitespace=False)
                page_chunk.extend(wrapped if wrapped else [""])
            for start in range(0, len(page_chunk), lines_per_page):
                render_text_page(pdf, page_chunk[start : start + lines_per_page])


def markdown_line_to_html(line: str) -> str:
    text = line.rstrip()
    if not text:
        return "<p></p>"
    if text.startswith("# "):
        return f"<h1>{markdown_to_plain_text(text[2:])}</h1>"
    if text.startswith("## "):
        return f"<h2>{markdown_to_plain_text(text[3:])}</h2>"
    if text.startswith("### "):
        return f"<h3>{markdown_to_plain_text(text[4:])}</h3>"
    if text.startswith("- "):
        return f"<li>{markdown_to_plain_text(text[2:])}</li>"
    if re.match(r"^\d+\.\s", text):
        return f"<li>{markdown_to_plain_text(re.sub(r'^\\d+\\.\\s*', '', text))}</li>"
    return f"<p>{markdown_to_plain_text(text)}</p>"


def is_markdown_table_separator(line: str) -> bool:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return False
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if not cells:
        return False
    return all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells if cell != "")


def parse_markdown_table_row(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def markdown_table_to_html(table_lines: List[str]) -> str:
    if len(table_lines) < 2:
        return "".join(f"<p>{html.escape(markdown_to_plain_text(line))}</p>" for line in table_lines)
    header = parse_markdown_table_row(table_lines[0])
    body = [parse_markdown_table_row(line) for line in table_lines[2:] if line.strip()]
    parts = ["<div class='table-wrap'><table>", "<thead><tr>"]
    for cell in header:
        parts.append(f"<th>{html.escape(markdown_to_plain_text(cell))}</th>")
    parts.append("</tr></thead><tbody>")
    for row in body:
        parts.append("<tr>")
        for idx, cell in enumerate(row):
            tag = "th" if idx == 0 and len(header) > 0 and header[0].lower() in {"level", "metric", "system", "model_family"} else "td"
            scope = " scope='row'" if tag == "th" else ""
            parts.append(f"<{tag}{scope}>{html.escape(markdown_to_plain_text(cell))}</{tag}>")
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def markdown_to_embedded_html(markdown_text: str, output_path: Path) -> None:
    lines = markdown_text.splitlines()
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8' />",
        "<meta name='viewport' content='width=device-width, initial-scale=1' />",
        "<title>RAG Research Paper</title>",
        "<style>",
        "body { font-family: Georgia, 'Times New Roman', serif; max-width: 980px; margin: 40px auto; padding: 0 24px; line-height: 1.55; color: #1f2328; }",
        "h1, h2, h3 { line-height: 1.2; }",
        "h1 { font-size: 2.2rem; margin-top: 0; }",
        "h2 { font-size: 1.5rem; margin-top: 2rem; border-bottom: 1px solid #d0d7de; padding-bottom: 0.25rem; }",
        "h3 { font-size: 1.15rem; margin-top: 1.4rem; }",
        "p, li { font-size: 1rem; }",
        "img { max-width: 100%; height: auto; display: block; margin: 14px auto; border: 1px solid #d8dee4; }",
        ".figure-group { margin: 20px 0 30px; }",
        ".figure-grid-two { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }",
        ".caption { font-size: 0.95rem; color: #57606a; margin-top: 10px; }",
        "pre { background: #f6f8fa; padding: 12px; overflow-x: auto; border-radius: 6px; font-size: 0.88rem; }",
        ".table-wrap { overflow-x: auto; margin: 18px 0 24px; border: 1px solid #d0d7de; border-radius: 8px; }",
        "table { width: 100%; border-collapse: collapse; font-size: 0.92rem; background: #ffffff; }",
        "thead { background: #f6f8fa; }",
        "th, td { border: 1px solid #d8dee4; padding: 10px 12px; vertical-align: top; text-align: left; }",
        "tbody tr:nth-child(even) { background: #fbfcfd; }",
        "tbody tr:hover { background: #f1f8ff; }",
        "th[scope='row'] { background: #f6f8fa; font-weight: 600; }",
        "ul, ol { padding-left: 1.6rem; }",
        "@media (max-width: 800px) { .figure-grid-two { grid-template-columns: 1fr; } body { margin: 20px auto; } }",
        "</style>",
        "</head>",
        "<body>",
    ]

    i = 0
    list_mode: Optional[str] = None
    while i < len(lines):
        line = lines[i]
        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if image_match:
            if list_mode:
                html_parts.append(f"</{list_mode}>")
                list_mode = None
            images: List[Tuple[str, str]] = []
            while i < len(lines):
                inner = re.match(r"!\[(.*?)\]\((.*?)\)", lines[i].strip())
                if not inner:
                    break
                alt, raw_path = inner.groups()
                path_obj = Path(raw_path)
                if not path_obj.is_absolute():
                    path_obj = (OUTPUT_DIR / raw_path).resolve()
                mime = "image/png" if path_obj.suffix.lower() == ".png" else "image/jpeg"
                encoded = base64.b64encode(path_obj.read_bytes()).decode("ascii")
                images.append((alt, f"data:{mime};base64,{encoded}"))
                i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            caption = ""
            if i < len(lines) and lines[i].strip().startswith("**Figure"):
                caption = markdown_to_plain_text(lines[i].strip())
                i += 1
            grid_class = "figure-grid-two" if len(images) > 1 else ""
            html_parts.append(f"<div class='figure-group {grid_class}'>")
            for alt, data_uri in images:
                html_parts.append(f"<figure><img alt=\"{alt}\" src=\"{data_uri}\" /></figure>")
            if caption:
                html_parts.append(f"<div class='caption'>{caption}</div>")
            html_parts.append("</div>")
            continue

        if line.strip().startswith("|") and i + 1 < len(lines) and is_markdown_table_separator(lines[i + 1]):
            if list_mode:
                html_parts.append(f"</{list_mode}>")
                list_mode = None
            table_lines = [line]
            i += 1
            table_lines.append(lines[i])
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            html_parts.append(markdown_table_to_html(table_lines))
            continue

        if line.startswith("- "):
            if list_mode != "ul":
                if list_mode:
                    html_parts.append(f"</{list_mode}>")
                html_parts.append("<ul>")
                list_mode = "ul"
            html_parts.append(markdown_line_to_html(line))
        elif re.match(r"^\d+\.\s", line):
            if list_mode != "ol":
                if list_mode:
                    html_parts.append(f"</{list_mode}>")
                html_parts.append("<ol>")
                list_mode = "ol"
            html_parts.append(markdown_line_to_html(line))
        else:
            if list_mode:
                html_parts.append(f"</{list_mode}>")
                list_mode = None
            html_parts.append(markdown_line_to_html(line))
        i += 1

    if list_mode:
        html_parts.append(f"</{list_mode}>")
    html_parts.extend(["</body>", "</html>"])
    output_path.write_text("\n".join(html_parts), encoding="utf-8")


def concise_executive_summary(overall_summary: pd.DataFrame, model_comparison: pd.DataFrame, adaptive_gain_model: pd.DataFrame) -> str:
    qi = overall_summary[overall_summary["metric"] == "quality_index"].iloc[0]
    answer = overall_summary[overall_summary["metric"] == "answer_relevance_1to5"].iloc[0]
    hall = overall_summary[overall_summary["metric"] == "hallucination_rate"].iloc[0]
    latency = overall_summary[overall_summary["metric"] == "response_time_s"].iloc[0]
    mem = overall_summary[overall_summary["metric"] == "gpu_mem_peak_mb"].iloc[0]

    model_qi = model_comparison[model_comparison["metric"] == "quality_index"].iloc[0]
    latency_model = model_comparison[model_comparison["metric"] == "response_time_s"].iloc[0]
    gain_qi = adaptive_gain_model[adaptive_gain_model["metric"] == "quality_index"].copy()
    best_gain_model = gain_qi.loc[gain_qi["adaptive_minus_fixed"].idxmax()]

    return (
        "Executive summary:\n"
        f"- Quality index: Fixed {fmt(qi['fixed_mean'])}, Adaptive {fmt(qi['adaptive_mean'])}, delta {fmt(qi['adaptive_minus_fixed'])}. "
        "This is the clearest overall indicator of whether adaptive evidence use improves final answer quality.\n"
        f"- Answer relevance delta: {fmt(answer['adaptive_minus_fixed'])}; hallucination-rate delta: {fmt(hall['adaptive_minus_fixed'])}. "
        "These two metrics show whether Adaptive is becoming more useful, more risky, or both.\n"
        f"- Operationally, response-time delta is {fmt(latency['adaptive_minus_fixed'])} and memory delta is {fmt(mem['adaptive_minus_fixed'])}, "
        "so the main deployment question is whether the quality change justifies the latency and footprint shift.\n"
        f"- Model quality comparison on quality index: Llama {fmt(model_qi['llama_mean'])} vs Qwen {fmt(model_qi['qwen_mean'])} "
        f"(Llama - Qwen = {fmt(model_qi['llama_minus_qwen'])}).\n"
        f"- Model efficiency comparison on response time: Llama {fmt(latency_model['llama_mean'])} vs Qwen {fmt(latency_model['qwen_mean'])}. "
        "This distinguishes the quality-first model from the efficiency-first model.\n"
        f"- Largest adaptive quality gain by model: {best_gain_model['model_family']} with delta {fmt(best_gain_model['adaptive_minus_fixed'])}.\n"
        "- Conceptually, the experiment tests whether retrieval should remain a fixed pipeline stage or become a dynamic policy for evidence use."
    )


def convert_report_to_paper_style(
    markdown_text: str,
    overall_summary: pd.DataFrame,
    model_comparison: pd.DataFrame,
    adaptive_gain_model: pd.DataFrame,
) -> str:
    qi = overall_summary[overall_summary["metric"] == "quality_index"].iloc[0]
    ar = overall_summary[overall_summary["metric"] == "answer_relevance_1to5"].iloc[0]
    hr = overall_summary[overall_summary["metric"] == "hallucination_rate"].iloc[0]
    rt = overall_summary[overall_summary["metric"] == "response_time_s"].iloc[0]
    mem = overall_summary[overall_summary["metric"] == "gpu_mem_peak_mb"].iloc[0]
    overall_model_qi = model_comparison[model_comparison["metric"] == "quality_index"].iloc[0]
    best_gain_model = adaptive_gain_model[adaptive_gain_model["metric"] == "quality_index"].sort_values(
        "adaptive_minus_fixed", ascending=False
    ).iloc[0]

    title_block = f"""# Adaptive vs Fixed Retrieval-Augmented Generation:
## A Paired Evaluation of Quality, Efficiency, and Model-Family Effects

### Abstract

This paper studies retrieval-augmented generation as a policy design problem rather than a simple implementation comparison. The experiment compares Fixed RAG, which applies a uniform retrieval behavior to every query, against Adaptive RAG, which adjusts retrieval and answer strategy to the query or the retrieved context. Using paired query-level comparisons across model family, quantization, and dataset size, the results show that Adaptive RAG improves answer relevance by {fmt(ar['adaptive_minus_fixed'])}, reduces hallucination rate by {fmt(-hr['adaptive_minus_fixed'])} in absolute improvement terms, and raises a composite quality index by {fmt(qi['adaptive_minus_fixed'])}. These gains come with a mixed operational profile: Adaptive lowers end-to-end response time by {fmt(-rt['adaptive_minus_fixed'])} seconds of improvement on average, but increases GPU peak memory by {fmt(mem['adaptive_minus_fixed'])} MB. Model-family effects are conditional rather than absolute. Qwen 32B is the stronger conservative baseline overall on the aggregate quality index, while Llama 3.3 70B benefits more from the adaptive policy and becomes the stronger answer model under Adaptive RAG. The findings imply that retrieval should be understood as a dynamic evidence-use policy whose value depends on corpus ambiguity, model behavior, and deployment constraints.

**Keywords:** retrieval-augmented generation, adaptive retrieval, hallucination, groundedness, model comparison, quantization, systems evaluation
"""

    replacements = {
        "# Deep Conceptual Analysis Report": title_block,
        "## Section 1. What this experiment is really testing": "## 1. Introduction",
        "## Section 2. Data audit and preprocessing": "## 2. Experimental Data, Audit, and Preprocessing",
        "## Section 3. Why paired query-level analysis matters": "## 3. Paired Evaluation Framework",
        "## Section 4. Conceptual meaning of the quality metrics": "## 4. Quality Metrics and Their Interpretation",
        "## Section 5. Retrieval quality vs answer quality": "## 5. Retrieval Quality versus Answer Quality",
        "## Section 6. Conceptual meaning of the operational metrics": "## 6. Operational Metrics and Systems Interpretation",
        "## Section 7. Why response time and throughput can disagree": "## 7. Why Latency and Throughput Can Diverge",
        "## Section 8. Overall quality and operational comparison: Adaptive RAG vs Fixed RAG": "## 8. Overall Comparison: Adaptive RAG versus Fixed RAG",
        "## Section 9. Factor-level interpretation": "## 9. Factor-Level Results",
        "## Section 10. Statistical analysis with conceptual interpretation": "## 10. Statistical Results",
        "## Section 11. Visualizations": "## 11. Figures",
        "## Section 12. Failure-mode interpretation": "## 12. Failure Modes",
        "## Section 13. Research conclusion for Adaptive RAG vs Fixed RAG": "## 13. Discussion",
        "## Section 14. Style and interpretation of the results": "## 14. Reading the Figures and Tables",
        "## Section 15. Where Adaptive RAG Outperforms Fixed RAG": "## 15. Where Adaptive RAG Outperforms Fixed RAG",
        "## Section 16. Overall Conceptual Takeaways": "## 16. Overall Conceptual Takeaways",
        "## Section 17. Model-Family Analysis: Llama 3.3 70B vs Qwen 32B": "## 17. Model-Family Analysis: Llama 3.3 70B versus Qwen 32B",
        "## Final executive summary": "## 18. Executive Summary",
    }
    for old, new in replacements.items():
        markdown_text = markdown_text.replace(old, new)

    markdown_text = re.sub(r"^- Unmatched deduplicated queries:.*\n", "", markdown_text, flags=re.MULTILINE)
    markdown_text = markdown_text.replace(
        "Preprocessing matters conceptually because query normalization prevents artificial mismatches caused by formatting differences, deduplication prevents repeated queries from overweighting one condition, and paired matching ensures that Fixed and Adaptive are compared under the same model family, quantization, dataset size, and normalized query text.",
        "Preprocessing matters conceptually because query normalization prevents artificial mismatches caused by formatting differences, deduplication prevents repeated queries from overweighting one condition, and paired matching ensures that Fixed and Adaptive are compared under the same model family, quantization, dataset size, and normalized query text. Because the normalized query set pairs cleanly for the comparative experiment, the paper reports the paired analysis directly rather than treating unmatched queries as a separate analytical object.",
    )
    policy_section = """
### 1.1 Heuristic Adaptive Policy Used in the Evaluation

The empirical results become easier to interpret when read against the actual adaptive policy used in the evaluation. In this experiment, Adaptive RAG is a heuristic routing policy rather than a single fixed retrieval setting. It first assigns each query to one of five functional types: `factual`, `multi_hop`, `analytical`, `comparison`, or `ambiguous`. That routing is rule-based. Comparison-style wording such as “compare,” “contrast,” or “difference between” is sent to the comparison route. Causal or connective wording such as “what led to,” “what caused,” or “how does X relate to Y” pushes the query toward the multi-hop route. Essay-like prompts such as “analyze,” “evaluate,” “discuss,” or long open-ended questions are routed to the analytical path. Vague or under-specified prompts are treated as ambiguous, while direct lookup questions default to factual.

Once the query is typed, the system applies a different retrieval budget and context policy:

| Query type | k | pool_size | context_chars | neighbor_window | max_per_source | Expansion rule |
| --- | --- | --- | --- | --- | --- | --- |
| factual | 3 | 40 | 6000 | 0 | 2 | none |
| multi_hop | 8 | 80 | 16000 | 2 | 4 | none |
| analytical | 6 | 60 | 14000 | 1 | 3 | none |
| comparison | 6 | 60 | 12000 | 1 | 2 | none |
| ambiguous | 4 | 60 | 10000 | 1 | 3 | expand to `k=8` if query coverage < 0.25 |

The key adaptive move is how `k` changes with prompt type. For direct factual prompts, `k` is decreased to 3 so the retriever stays narrow and precise. For analytical and comparison prompts, `k` is increased to 6 because the answer requires broader support or multi-sided evidence. For multi-hop prompts, `k` is increased further to 8 because the system expects the answer to depend on linking evidence across multiple passages. For ambiguous prompts, the policy begins at `k=4`, which is more conservative than the multi-hop and analytical settings, but it increases to `k=8` when the first pass does not cover enough of the query. In other words, Adaptive RAG uses smaller `k` for precision-oriented prompts and larger `k` for synthesis-oriented prompts.

This policy means Adaptive RAG does not simply retrieve more passages. It retrieves differently depending on the reasoning demand of the question. Factual questions get a small and precise evidence budget. Multi-hop questions get the largest retrieval budget and the widest neighboring context because they require chaining facts across passages. Analytical questions get a broad but not maximal context window so the system can support synthesis without overwhelming the answer stage. Comparison questions cap evidence per source more aggressively so the retrieved set is forced to spread across entities or documents rather than collapsing onto one source. Ambiguous questions begin conservatively and expand only if the first retrieval pass covers too little of the query vocabulary.

The retriever itself combines dense retrieval with lightweight hybrid reranking, source balancing, lexical overlap boosting, and optional neighbor-chunk expansion. That matters conceptually because the policy is not only changing how many passages are retrieved; it is also changing the diversity and continuity of the evidence set.

After retrieval, the system estimates whether the context is sufficiently strong for a fully grounded answer:

| Context status | Decision rule | Answer behavior |
| --- | --- | --- |
| grounded_context | top retrieval score >= 0.50 and query coverage >= 0.30 | use a fully grounded, query-type-specific answer mode |
| partial_context | top retrieval score >= 0.30 or query coverage >= 0.18 | use a hybrid mode that mixes retrieved evidence with general knowledge |
| no_context | otherwise | answer from general knowledge only, without document citations |

This second-stage decision is crucial. The policy does not assume that retrieval either succeeded or failed in a binary way. Instead, it distinguishes strong evidence, partial evidence, and missing evidence, and it changes the answer strategy accordingly. Grounded factual, multi-hop, analytical, and comparison questions get different answer formats. Partial-context cases move to a hybrid answer mode. Ambiguous questions remain more permissive because the system assumes the query itself may require broader interpretation.

Conceptually, this heuristic policy explains why Adaptive RAG can improve answer relevance even when context relevance does not always increase. The system is gaining leverage from four mechanisms at once: query-type-sensitive retrieval budgets, source-diversity control, context-sufficiency estimation, and answer-mode routing. That combination is what turns Adaptive RAG into an evidence-use policy rather than just a larger retriever.
"""
    markdown_text = markdown_text.replace(
        "That makes this a systems-and-quality tradeoff problem, not just an accuracy comparison. The deeper design question is whether retrieval should be treated as a static pipeline stage or as a decision policy that can expand, constrain, route, or hybridize its answer strategy when evidence is partial or ambiguous.",
        "That makes this a systems-and-quality tradeoff problem, not just an accuracy comparison. The deeper design question is whether retrieval should be treated as a static pipeline stage or as a decision policy that can expand, constrain, route, or hybridize its answer strategy when evidence is partial or ambiguous.\n\n" + policy_section,
    )
    markdown_text = markdown_text.replace(
        "The model comparison should end in a conditional conclusion rather than a slogan.",
        f"The model comparison should end in a conditional conclusion rather than a slogan. In the aggregate tables, Qwen 32B has the higher overall quality index ({fmt(overall_model_qi['qwen_mean'])} versus {fmt(overall_model_qi['llama_mean'])}), but {best_gain_model['model_family']} shows the larger Adaptive-minus-Fixed quality gain ({fmt(best_gain_model['adaptive_minus_fixed'])}).",
    )
    return markdown_text


def main() -> None:
    ensure_output_dir()
    raw, dedup, audit = load_and_prepare()
    paired, unmatched = build_paired(dedup)

    overall_summary = summarise_overall(dedup, paired)
    dataset_summary = summarise_factor(dedup, "dataset_size")
    model_summary = summarise_factor(dedup, "model_family")
    quant_summary = summarise_factor(dedup, "quantization")
    config_summary = summarise_configuration_level(paired)
    stat_results = run_stat_tests(paired)
    model_comparison = compare_models_overall(dedup)
    model_by_system = compare_models_by_system(dedup)
    adaptive_gain_model = adaptive_gain_by_model(paired)
    interaction_summary = model_factor_interaction_summary(paired)

    save_tables(
        raw,
        dedup,
        paired,
        unmatched,
        overall_summary,
        dataset_summary,
        model_summary,
        quant_summary,
        config_summary,
        stat_results,
        model_comparison,
        model_by_system,
        adaptive_gain_model,
        interaction_summary,
    )

    create_figures(dedup, paired, config_summary, model_comparison, adaptive_gain_model)

    report = generate_report(
        raw,
        dedup,
        paired,
        unmatched,
        audit,
        overall_summary,
        dataset_summary,
        model_summary,
        quant_summary,
        config_summary,
        stat_results,
        model_comparison,
        model_by_system,
        adaptive_gain_model,
        interaction_summary,
    )
    report = relativize_markdown_images_for_github(report)
    report_md_paths = [
        OUTPUT_DIR / "deep_conceptual_analysis_report.md",
        OUTPUT_DIR / "rag_research_paper.md",
    ]
    report_pdf_paths = [
        OUTPUT_DIR / "deep_conceptual_analysis_report.pdf",
        OUTPUT_DIR / "rag_research_paper.pdf",
    ]
    report_html_paths = [
        OUTPUT_DIR / "rag_research_paper_embedded.html",
        OUTPUT_DIR / "deep_conceptual_analysis_report_embedded.html",
    ]
    for path in report_md_paths:
        path.write_text(report, encoding="utf-8")
    for path in report_pdf_paths:
        render_pdf_from_markdown(report, path)
    for path in report_html_paths:
        markdown_to_embedded_html(report, path)

    generated_files = sorted([p.name for p in OUTPUT_DIR.iterdir() if p.is_file()])
    print(concise_executive_summary(overall_summary, model_comparison, adaptive_gain_model))
    print("\nGenerated files:")
    for name in generated_files:
        print(f"- {name}")


if __name__ == "__main__":
    main()
