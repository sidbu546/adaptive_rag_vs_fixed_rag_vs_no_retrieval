# Adaptive RAG vs Fixed RAG vs No-Retrieval LLM: Comprehensive Analysis Report

## Executive Summary

This report compares **Adaptive RAG**, **Fixed RAG**, and **No-Retrieval LLM** across all available experiment CSVs in this folder.
- Total observations: **416**
- Unique configurations: **20**
- Approaches included: Adaptive RAG, Fixed RAG, No Retrieval LLM

### Dataset Coverage

| Approach | Model | Quantization | Corpus | n |
|----------|-------|--------------|--------|---|
| Adaptive RAG | Llama-70B | 4-bit | Medium | 24 |
| Adaptive RAG | Llama-70B | 4-bit | Small | 20 |
| Adaptive RAG | Llama-70B | 8-bit | Medium | 24 |
| Adaptive RAG | Llama-70B | 8-bit | Small | 20 |
| Adaptive RAG | Qwen-32B | 4-bit | Medium | 24 |
| Adaptive RAG | Qwen-32B | 4-bit | Small | 20 |
| Adaptive RAG | Qwen-32B | 8-bit | Medium | 24 |
| Adaptive RAG | Qwen-32B | 8-bit | Small | 20 |
| Fixed RAG | Llama-70B | 4-bit | Medium | 20 |
| Fixed RAG | Llama-70B | 4-bit | Small | 20 |
| Fixed RAG | Qwen-32B | 4-bit | Medium | 20 |
| Fixed RAG | Qwen-32B | 4-bit | Small | 20 |
| No Retrieval LLM | Llama-70B | 4-bit | Medium | 20 |
| No Retrieval LLM | Llama-70B | 4-bit | Small | 20 |
| No Retrieval LLM | Llama-70B | 8-bit | Medium | 20 |
| No Retrieval LLM | Llama-70B | 8-bit | Small | 20 |
| No Retrieval LLM | Qwen-32B | 4-bit | Medium | 20 |
| No Retrieval LLM | Qwen-32B | 4-bit | Small | 20 |
| No Retrieval LLM | Qwen-32B | 8-bit | Medium | 20 |
| No Retrieval LLM | Qwen-32B | 8-bit | Small | 20 |

### Overall Means by Approach

| Approach | Hallucination | Groundedness | Answer Relevance | Context Relevance | Response Time (s) |
|----------|---------------|--------------|------------------|-------------------|-------------------|
| Adaptive RAG | 0.1074 | 0.7865 | 4.3182 | 3.7841 | 23.54 |
| Fixed RAG | 0.3809 | 0.6191 | 4.8750 | 4.0000 | 11.01 |
| No Retrieval LLM | 0.9309 | 0.0691 | 1.9625 | 4.9750 | 13.20 |

## Pairwise Statistical Comparisons (Welch's t-test)

### Adaptive RAG vs Fixed RAG

| Metric | Adaptive RAG (mean +/- std) | Fixed RAG (mean +/- std) | Diff | p-value | Cohen's d | Sig. |
|--------|---------------------|----------------------|------|---------|-----------|------|
| hallucination_rate | 0.1074 +/- 0.2247 | 0.3809 +/- 0.1750 | -0.2735 | 0.0000*** | -1.299 | Yes |
| groundedness_score | 0.7865 +/- 0.2572 | 0.6191 +/- 0.1750 | 0.1674 | 0.0000*** | 0.713 | Yes |
| answer_relevance_1to5 | 4.3182 +/- 0.8625 | 4.8750 +/- 0.4017 | -0.5568 | 0.0000*** | -0.742 | Yes |
| context_relevance_1to5 | 3.7841 +/- 1.2644 | 4.0000 +/- 0.8419 | -0.2159 | 0.1085 | -0.188 | No |
| confidence | 0.5221 +/- 0.1017 | 0.5628 +/- 0.1166 | -0.0406 | 0.0082** | -0.381 | Yes |
| response_time_s | 23.5364 +/- 23.9140 | 11.0091 +/- 5.9049 | 12.5273 | 0.0000*** | 0.623 | Yes |
| llm_latency_s | 23.5137 +/- 23.9110 | 10.6939 +/- 5.8962 | 12.8197 | 0.0000*** | 0.637 | Yes |
| gpu_throughput_toks_per_s | 9.5788 +/- 4.6616 | 7.9650 +/- 1.9089 | 1.6137 | 0.0001*** | 0.402 | Yes |
| eff_gpu_throughput | 7.1045 +/- 4.5537 | 7.9650 +/- 1.9089 | -0.8606 | 0.0342* | -0.219 | Yes |
| gpu_util_percent | 67.2609 +/- 24.3415 | 58.2090 +/- 16.7929 | 9.0519 | 0.0007*** | 0.406 | Yes |
| gpu_mem_percent | 75.2519 +/- 18.5629 | 51.1274 +/- 22.0879 | 24.1246 | 0.0000*** | 1.223 | Yes |
| gpu_mem_peak_mb | 47336.7812 +/- 16821.0333 | 23613.6391 +/- 10146.5125 | 23723.1422 | 0.0000*** | 1.575 | Yes |
| total_deployment_cost_usd | 2.0196 +/- 0.0199 | 0.0000 +/- 0.0000 | 2.0196 | 0.0000*** | 122.099 | Yes |
| retrieved_docs_count | 3.4773 +/- 2.0113 | 3.0000 +/- 0.0000 | 0.4773 | 0.0019** | 0.286 | Yes |
| top_retrieval_score | 0.5221 +/- 0.1017 | 0.5197 +/- 0.1117 | 0.0025 | 0.8672 | 0.023 | No |
| avg_retrieval_score | 0.5039 +/- 0.0979 | 0.4681 +/- 0.0935 | 0.0358 | 0.0057** | 0.371 | Yes |
| query_coverage | 0.5554 +/- 0.1926 | 0.8477 +/- 0.1194 | -0.2923 | 0.0000*** | -1.688 | Yes |

### Adaptive RAG vs No Retrieval LLM

| Metric | Adaptive RAG (mean +/- std) | No Retrieval LLM (mean +/- std) | Diff | p-value | Cohen's d | Sig. |
|--------|---------------------|----------------------|------|---------|-----------|------|
| hallucination_rate | 0.1074 +/- 0.2247 | 0.9309 +/- 0.1960 | -0.8235 | 0.0000*** | -3.893 | Yes |
| groundedness_score | 0.7865 +/- 0.2572 | 0.0691 +/- 0.1960 | 0.7174 | 0.0000*** | 3.118 | Yes |
| answer_relevance_1to5 | 4.3182 +/- 0.8625 | 1.9625 +/- 0.8964 | 2.3557 | 0.0000*** | 2.681 | Yes |
| context_relevance_1to5 | 3.7841 +/- 1.2644 | 4.9750 +/- 0.1566 | -1.1909 | 0.0000*** | -1.292 | Yes |
| confidence | 0.5221 +/- 0.1017 | 0.3882 +/- 0.1255 | 0.1339 | 0.0000*** | 1.178 | Yes |
| response_time_s | 23.5364 +/- 23.9140 | 13.2003 +/- 9.6963 | 10.3362 | 0.0000*** | 0.557 | Yes |
| llm_latency_s | 23.5137 +/- 23.9110 | 13.1603 +/- 9.6963 | 10.3534 | 0.0000*** | 0.558 | Yes |
| gpu_throughput_toks_per_s | 9.5788 +/- 4.6616 | 14.2484 +/- 9.9267 | -4.6697 | 0.0000*** | -0.612 | Yes |
| eff_gpu_throughput | 7.1045 +/- 4.5537 | 14.1185 +/- 9.7846 | -7.0140 | 0.0000*** | -0.934 | Yes |
| gpu_util_percent | 67.2609 +/- 24.3415 | 59.1534 +/- 14.6569 | 8.1075 | 0.0002*** | 0.399 | Yes |
| gpu_mem_percent | 75.2519 +/- 18.5629 | 47.9347 +/- 18.9042 | 27.3172 | 0.0000*** | 1.459 | Yes |
| gpu_mem_peak_mb | 47336.7812 +/- 16821.0333 | 45020.6500 +/- 18824.8037 | 2316.1312 | 0.2370 | 0.130 | No |
| total_deployment_cost_usd | 2.0196 +/- 0.0199 | 0.0092 +/- 0.0067 | 2.0104 | 0.0000*** | 132.666 | Yes |
| retrieved_docs_count | 3.4773 +/- 2.0113 | 2.0000 +/- 0.0000 | 1.4773 | 0.0000*** | 1.015 | Yes |
| top_retrieval_score | 0.5221 +/- 0.1017 | 0.3379 +/- 0.0660 | 0.1843 | 0.0000*** | 2.129 | Yes |
| avg_retrieval_score | 0.5039 +/- 0.0979 | 0.2102 +/- 0.0633 | 0.2937 | 0.0000*** | 3.528 | Yes |
| query_coverage | 0.5554 +/- 0.1926 | 1.0000 +/- 0.0000 | -0.4446 | 0.0000*** | -3.188 | Yes |

### Fixed RAG vs No Retrieval LLM

| Metric | Fixed RAG (mean +/- std) | No Retrieval LLM (mean +/- std) | Diff | p-value | Cohen's d | Sig. |
|--------|---------------------|----------------------|------|---------|-----------|------|
| hallucination_rate | 0.3809 +/- 0.1750 | 0.9309 +/- 0.1960 | -0.5500 | 0.0000*** | -2.906 | Yes |
| groundedness_score | 0.6191 +/- 0.1750 | 0.0691 +/- 0.1960 | 0.5500 | 0.0000*** | 2.906 | Yes |
| answer_relevance_1to5 | 4.8750 +/- 0.4017 | 1.9625 +/- 0.8964 | 2.9125 | 0.0000*** | 3.790 | Yes |
| context_relevance_1to5 | 4.0000 +/- 0.8419 | 4.9750 +/- 0.1566 | -0.9750 | 0.0000*** | -1.943 | Yes |
| confidence | 0.5628 +/- 0.1166 | 0.3882 +/- 0.1255 | 0.1745 | 0.0000*** | 1.423 | Yes |
| response_time_s | 11.0091 +/- 5.9049 | 13.2003 +/- 9.6963 | -2.1912 | 0.0314* | -0.254 | Yes |
| llm_latency_s | 10.6939 +/- 5.8962 | 13.1603 +/- 9.6963 | -2.4663 | 0.0155* | -0.286 | Yes |
| gpu_throughput_toks_per_s | 7.9650 +/- 1.9089 | 14.2484 +/- 9.9267 | -6.2834 | 0.0000*** | -0.767 | Yes |
| eff_gpu_throughput | 7.9650 +/- 1.9089 | 14.1185 +/- 9.7846 | -6.1534 | 0.0000*** | -0.762 | Yes |
| gpu_util_percent | 58.2090 +/- 16.7929 | 59.1534 +/- 14.6569 | -0.9444 | 0.6693 | -0.061 | No |
| gpu_mem_percent | 51.1274 +/- 22.0879 | 47.9347 +/- 18.9042 | 3.1926 | 0.2706 | 0.159 | No |
| gpu_mem_peak_mb | 23613.6391 +/- 10146.5125 | 45020.6500 +/- 18824.8037 | -21407.0109 | 0.0000*** | -1.301 | Yes |
| total_deployment_cost_usd | 0.0000 +/- 0.0000 | 0.0092 +/- 0.0067 | -0.0092 | 0.0000*** | -1.666 | Yes |
| retrieved_docs_count | 3.0000 +/- 0.0000 | 2.0000 +/- 0.0000 | 1.0000 | 0.0000*** | nan | Yes |
| top_retrieval_score | 0.5197 +/- 0.1117 | 0.3379 +/- 0.0660 | 0.1818 | 0.0000*** | 2.166 | Yes |
| avg_retrieval_score | 0.4681 +/- 0.0935 | 0.2102 +/- 0.0633 | 0.2579 | 0.0000*** | 3.453 | Yes |
| query_coverage | 0.8477 +/- 0.1194 | 1.0000 +/- 0.0000 | -0.1523 | 0.0000*** | -2.214 | Yes |

## Best Configurations

- Highest groundedness: `Adaptive RAG | Llama-70B | 4-bit | Small` (0.8492)
- Lowest hallucination: `Adaptive RAG | Qwen-32B | 8-bit | Small` (0.0644)
- Fastest response: `No Retrieval LLM | Qwen-32B | 4-bit | Small` (3.66s)

## Interpretation

- Hallucination (lower is better): **Adaptive RAG** best (0.1074), then Fixed RAG (0.3809), then No Retrieval LLM (0.9309).
- Groundedness (higher is better): **Adaptive RAG** best (0.7865), then Fixed RAG (0.6191), then No Retrieval LLM (0.0691).
- Latency (lower is better): **Fixed RAG** fastest (11.01s), then No Retrieval LLM (13.20s), then Adaptive RAG (23.54s).

## Validity Notes

- Fixed RAG currently appears only in 4-bit runs (no 8-bit fixed files found), so approach-level comparisons are partially confounded by quantization.
- Retrieval score metrics are architecture-dependent and should be interpreted primarily within approach rather than across approach.
- Results assume each row is an independent observation from the logging pipeline.