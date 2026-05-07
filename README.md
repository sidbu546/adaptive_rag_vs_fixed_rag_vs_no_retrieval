# NLP Project: Adaptive RAG vs Fixed RAG vs No-Retrieval

This repository evaluates three QA system styles under the same metric schema:

- `Adaptive RAG` (query-type-aware retrieval + answer routing)
- `Fixed RAG` (static retrieval policy)
- `No Retrieval` (generation-only baseline)

The project includes:

- model evaluation pipelines
- CSV result datasets across model/quantization/corpus-size settings
- statistical comparison scripts
- figure generation
- publication-ready report assets (`.md`, `.pdf`, `.html`, LaTeX bundle)

---

## 1) Project Goal

The core goal is to measure quality/grounding tradeoffs versus operational cost and latency across:

- Retrieval strategy: adaptive vs fixed vs none
- Model family: Llama vs Qwen
- Quantization: 4-bit vs 8-bit
- Dataset scale: small, medium, big

The key research question is whether retrieval should be treated as:

- a static pipeline step, or
- an adaptive decision policy that changes behavior by query type and context strength.

---

## 2) Repository Structure

Top-level directories and important files:

- `adaptive_rag/`
  - adaptive retrieval/generation pipeline code (`run_models.py`, `rag_core.py`, `rag_retriever_chroma.py`, `rag_eval.py`, `llm_manager.py`)
- `fixed_rag/`
  - fixed-policy experiment scripts (GPU-oriented, path-configured scripts)
- `results_datasets/`
  - evaluation CSVs for adaptive/fixed/no-retrieval runs
- `analysis_outputs/`
  - generated comparison tables, plots, and long-form reports
- `analysis/`
  - mirrored analysis scripts
- `no_retrieval.py`
  - no-retrieval baseline runner
- `analyze_rag_experiment.py`
  - comprehensive adaptive-vs-fixed analysis/report generator
- `generate_comprehensive_adapt_fixed_noretrieval.py`
  - full 3-way comparative report generator
- `generate_all_datasets_analysis.py`
  - all-datasets consolidated analysis
- `generate_operational_metrics_compact.py`
  - compact operational metrics figure
- `generate_tradeoff_quality_gain_figure.py`
  - quality-gain vs latency-delta tradeoff figure
- `final_report_acl_style.tex`
  - ACL-style paper source

---

## 3) Data and Output Conventions

### Input result CSV naming (critical for analysis scripts)

Analysis scripts parse metadata from filenames. Use these prefixes:

- Adaptive RAG: `rag_eval_*`
- Fixed RAG: `fixed_rag_results_*`
- No retrieval: `noretrieval_*`

Expected suffix patterns include model, quantization, and size tags such as:

- `llama` or `qwen`
- `4bit` or `8bit`
- `small`, `medium`, `big`

### Main results location

- Raw/curated experiment CSVs are in `results_datasets/`.
- Several analysis scripts expect matching CSVs in the same directory as the script (project root for top-level scripts), so copy/symlink as needed before running.

### Standard metric fields

Most generated CSVs include a common schema (or near-subset):

- quality: `hallucination_rate`, `groundedness_score`, `answer_relevance_1to5`, `context_relevance_1to5`, `query_coverage`, `confidence`
- performance: `response_time_s`, `llm_latency_s`, `gpu_throughput_toks_per_s`, `eff_gpu_throughput`, `gpu_util_percent`, `gpu_mem_percent`, `gpu_mem_peak_mb`
- retrieval/context: `retrieved_docs_count`, `top_retrieval_score`, `avg_retrieval_score`, `context_status`, `answer_mode_used`, `used_general_knowledge`
- bookkeeping: query text/type and deployment cost fields

---

## 4) Adaptive RAG System Design

`adaptive_rag/rag_core.py` is the main policy layer.

### Heuristic-first query typing (primary mode)

The adaptive pipeline classifies prompts into:

- `factual`
- `multi_hop`
- `analytical`
- `comparison`
- `ambiguous`

For this project, the important and intended classification mode is:

- `heuristic` (keyword/rule-based)

The heuristic classifier is not a toy fallback here. It is the policy backbone used to drive retrieval behavior deterministically across runs. In practical terms, it gives:

- stable experiment reproducibility (same query text -> same type decision)
- low overhead (no extra model pass for typing)
- interpretable routing logic that can be audited and tuned
- cleaner ablation against Fixed RAG and No Retrieval baselines

### Heuristic decision logic (what gets tagged as what)

The classifier in `adaptive_rag/rag_core.py` uses structured signal groups:

- **Comparison signals** first (highest-priority match), e.g. compare/contrast style prompts.
- **Multi-hop signals** next, focused on causal/relational wording with minimum query length checks.
- **Analytical signals** next, including evaluation/discussion language and long-form prompt cues.
- **Ambiguous signals** after that, including vague/opinion forms or very short under-specified prompts.
- If none of the above dominates, it defaults to `factual`.

This priority order is critical: a prompt with multiple cues is not randomly assigned. The routing is intentional so retrieval budgets map to reasoning complexity.

### Retrieval policy by query type

The classifier output is consumed by `RETRIEVAL_POLICIES`, which changes:

- top-k
- candidate pool size
- context window size
- neighbor chunk expansion
- per-source cap

Policy behavior is intentionally asymmetric:

- **factual**: small `k`, tighter context, precision-first retrieval
- **multi_hop**: larger `k`, wider candidate pool, extra neighbor windows for evidence chaining
- **analytical**: medium-high retrieval breadth with larger context budget for synthesis
- **comparison**: source diversity pressure (`max_per_source` constraints) to prevent one-source collapse
- **ambiguous**: starts conservative, then expands retrieval when early query coverage is weak

### Adaptive retrieval-control loop

For ambiguous or low-coverage cases, the policy can re-expand retrieval (`expansion_k`) instead of committing to early weak evidence. This is where Adaptive RAG stops being "just RAG with bigger k" and becomes a control loop:

1. classify query type
2. run type-specific retrieval budget
3. estimate context adequacy
4. expand/adjust retrieval if needed
5. then generate answer with the right prompt mode

### Context-strength routing and answer strategy

The system estimates context sufficiency and routes answer mode:

- `grounded_context`
- `partial_context`
- `no_context`

Prompt strategy changes with this status, and that decision directly controls citation behavior and grounding strictness:

- **grounded**: strict type-specific grounded prompts with evidence-centered answering
- **partial**: hybrid prompts that mix retrieved context and model priors
- **no context**: general-knowledge-only response mode without retrieval citations

This two-stage mechanism (heuristic type routing + context-strength routing) is the main differentiator from static retrieval. Fixed RAG keeps retrieval policy constant; Adaptive RAG changes both evidence acquisition and answer behavior based on query demand and context quality.

---

## 5) Fixed RAG and No-Retrieval Baselines

### Fixed RAG

`fixed_rag/` contains fixed-policy scripts with:

- static chunk/retrieval settings
- GPU-focused execution assumptions
- hardcoded path-style configs (originally cluster-oriented)

These scripts are useful as baseline generators but usually need path edits for a new machine.

### No Retrieval baseline

`no_retrieval.py` runs:

- PDF/prompts ingestion
- generation without retrieval grounding
- judge-based scoring and GPU telemetry logging
- CSV output compatible with downstream comparative analysis

CLI arguments include `--pdf_path`, `--prompts_path`, `--output_csv`, `--model_path`, quantization controls, cache settings, and generation params.

---

## 6) End-to-End Workflow (Start to Finish)

Use this flow for a full reproduction pipeline.

### Step A: Environment setup

Use Python 3.10+ (recommended 3.10/3.11) and install core packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy scipy matplotlib pillow torch transformers bitsandbytes sentence-transformers chromadb langchain-community langchain-text-splitters pypdf pymupdf pynvml openpyxl
```

Notes:

- `bitsandbytes` is needed for 4-bit/8-bit quantization modes.
- `pymupdf` is optional fallback-sensitive (`fitz` import path); `pypdf` is also used.
- On Apple Silicon/CPU-only systems, large-model GPU scripts may not be feasible without adaptation.

### Step B: Generate/collect experiment CSVs

1. Adaptive RAG runs from `adaptive_rag/run_models.py` (multiple configs).
2. Fixed RAG runs from scripts in `fixed_rag/` (after path updates).
3. No-retrieval runs from `no_retrieval.py`.
4. Place normalized result CSVs in `results_datasets/` and/or script-expected root locations.

### Step C: Run comparative analyses

Run one or more:

```bash
python analyze_rag_experiment.py
python generate_comprehensive_adapt_fixed_noretrieval.py
python generate_all_datasets_analysis.py
python generate_operational_metrics_compact.py
python generate_tradeoff_quality_gain_figure.py
```

### Step D: Inspect generated artifacts

Primary outputs appear in:

- `analysis_outputs/` (figures, CSV summaries, markdown/html/pdf reports)
- top-level `.md/.xlsx` outputs for comprehensive scripts

### Step E: Paper/report packaging

Use:

- `final_report_acl_style.tex`
- `final_report_refs.bib`
- assets from `analysis_outputs/`

to produce publication-style output (`final_report_acl_style.pdf` already present).

---

## 7) Command Examples

### Adaptive RAG run example

```bash
python adaptive_rag/run_models.py \
  --model_preset llama33_70b \
  --quantization_mode 4bit \
  --data_dir /path/to/data \
  --query_file /path/to/queries.txt \
  --out_csv rag_eval_llama_4bit_small_adaptive.csv \
  --classifier_mode heuristic
```

### No-retrieval run example

```bash
python no_retrieval.py \
  --pdf_path /path/to/document.pdf \
  --prompts_path /path/to/prompts.txt \
  --output_csv noretrieval_llama_4bit_small.csv \
  --model_path /path/to/local/model \
  --quant_mode 4bit
```

---

## 8) Analysis Outputs: What Each File Means

Typical `analysis_outputs/` artifacts:

- `cleaned_deduplicated_data.csv` - normalized pooled data
- `paired_query_level_deltas.csv` - adaptive vs fixed paired deltas
- `overall_metric_summary.csv` - aggregate metric means/comparisons
- `factor_summary_*.csv` - effects by dataset/model/quantization
- `configuration_level_deltas.csv` - config-level quality/latency differences
- `statistical_test_results.csv` - significance testing tables
- `model_by_system_comparison.csv` and `adaptive_gain_by_model.csv` - model-family deltas
- figure PNGs (heatmaps, tradeoff plots, interaction plots)
- report docs:
  - `deep_conceptual_analysis_report.md/.pdf/.html`
  - `rag_research_paper.md/.pdf/.html`

---

## 9) Reproducibility and Operational Notes

- GPU-heavy scripts are tuned for high-memory CUDA environments (often A100-class assumptions).
- Quantization and `CUDA_VISIBLE_DEVICES` materially change feasibility and speed.
- Some fixed scripts have hardcoded absolute paths and prompt lists; adjust before reruns.
- Analysis scripts rely on filename parsing; inconsistent names will be skipped or fail metadata parsing.
- Multiple-comparison corrections are generally not applied in provided statistical summaries (interpret p-values accordingly).

---

## 10) Troubleshooting

- **No CSVs found / parse errors**  
  Ensure filenames follow expected prefixes/tags (`rag_eval_`, `fixed_rag_results_`, `noretrieval_`, plus model/bit/size markers).

- **Out-of-memory during model load/inference**  
  Lower `max_new_tokens`, switch to `4bit`, reduce context/chunking settings, or use a smaller model.

- **bitsandbytes import/runtime errors**  
  Use `--quantization_mode none` temporarily or install a compatible CUDA+binaries stack.

- **Analysis outputs missing figures**  
  Confirm dependencies (`matplotlib`, `scipy`, `pillow`) and rerun analysis scripts from project root.

---


