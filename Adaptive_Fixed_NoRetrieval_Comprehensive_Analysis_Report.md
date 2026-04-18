# Adaptive RAG vs Fixed RAG vs No Retrieval LLM: Comprehensive Analysis Report

This file is the **single consolidated analysis report** for this project (Adaptive RAG vs Fixed RAG vs No Retrieval LLM), including all datasets currently in the workspace.

## Executive Summary

This report provides a full comparative analysis of **Adaptive RAG**, **Fixed RAG**, and **No Retrieval LLM** across all available datasets in this project.
- **Total observations**: 496
- **Unique configurations**: 24
- **Approaches analyzed**: Adaptive RAG, Fixed RAG, No Retrieval LLM

### Key Findings at a Glance

| Finding | Detail |
|---------|--------|
| Lowest Hallucination | **Adaptive RAG** (0.1074) |
| Highest Groundedness | **Adaptive RAG** (0.7865) |
| Fastest Response | **Fixed RAG** (13.01s) |

---

## 1. Dataset Coverage and Balance

| Approach | Model | Quantization | Corpus Size | n |
|----------|-------|--------------|-------------|---|
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
| Fixed RAG | Llama-70B | 8-bit | Medium | 20 |
| Fixed RAG | Llama-70B | 8-bit | Small | 20 |
| Fixed RAG | Qwen-32B | 4-bit | Medium | 20 |
| Fixed RAG | Qwen-32B | 4-bit | Small | 20 |
| Fixed RAG | Qwen-32B | 8-bit | Medium | 20 |
| Fixed RAG | Qwen-32B | 8-bit | Small | 20 |
| No Retrieval LLM | Llama-70B | 4-bit | Medium | 20 |
| No Retrieval LLM | Llama-70B | 4-bit | Small | 20 |
| No Retrieval LLM | Llama-70B | 8-bit | Medium | 20 |
| No Retrieval LLM | Llama-70B | 8-bit | Small | 20 |
| No Retrieval LLM | Qwen-32B | 4-bit | Medium | 20 |
| No Retrieval LLM | Qwen-32B | 4-bit | Small | 20 |
| No Retrieval LLM | Qwen-32B | 8-bit | Medium | 20 |
| No Retrieval LLM | Qwen-32B | 8-bit | Small | 20 |

---

## 2. Overall Means by Approach

| Approach | Corpus scope | Hallucination | Groundedness | Answer Relevance | Context Relevance | Confidence | Response Time (s) | LLM Latency (s) | GPU Throughput (tok/s) | Effective GPU Throughput | GPU Util (%) | GPU Mem (%) | GPU Mem Peak (MB) |
|----------|--------------|---------------|--------------|------------------|-------------------|------------|-------------------|-----------------|------------------------|--------------------------|--------------|-------------|---------------------|
| Adaptive RAG | Small + Medium (combined) | 0.1074 | 0.7865 | 4.3182 | 3.7841 | 0.5221 | 23.54 | 23.51 | 9.5788 | 7.1045 | 67.26 | 75.25 | 47336.78 |
| Fixed RAG | Small + Medium (combined) | 0.3442 | 0.6558 | 4.8188 | 4.1312 | 0.5803 | 13.01 | 12.73 | 7.0776 | 7.0776 | 59.43 | 33.51 | 20396.26 |
| No Retrieval LLM | Small + Medium (combined) | 0.9309 | 0.0691 | 1.9625 | 4.9750 | 0.3882 | 13.20 | 13.16 | 14.2484 | 14.1185 | 59.15 | 47.93 | 45020.65 |

### 2.1 Overall Means by Approach and Corpus Size (Small vs Medium)

| Approach | Corpus | Hallucination | Groundedness | Answer Relevance | Context Relevance | Confidence | Response Time (s) | LLM Latency (s) | GPU Throughput (tok/s) | Effective GPU Throughput | GPU Util (%) | GPU Mem (%) | GPU Mem Peak (MB) |
|----------|--------|---------------|--------------|------------------|-------------------|------------|-------------------|-----------------|------------------------|--------------------------|--------------|-------------|---------------------|
| Adaptive RAG | Medium | 0.1336 | 0.7548 | 4.1354 | 3.4583 | 0.5010 | 21.61 | 21.59 | 9.9595 | 7.6268 | 69.47 | 76.15 | 47751.66 |
| Adaptive RAG | Small | 0.0760 | 0.8244 | 4.5375 | 4.1750 | 0.5476 | 25.85 | 25.82 | 9.1219 | 6.4777 | 64.61 | 74.18 | 46838.93 |
| Fixed RAG | Medium | 0.4222 | 0.5778 | 4.9625 | 3.9250 | 0.5320 | 12.21 | 11.94 | 7.1209 | 7.1209 | 55.57 | 40.59 | 23613.63 |
| Fixed RAG | Small | 0.2662 | 0.7338 | 4.6750 | 4.3375 | 0.6287 | 13.81 | 13.52 | 7.0343 | 7.0343 | 63.28 | 26.42 | 17178.90 |
| No Retrieval LLM | Medium | 0.9497 | 0.0503 | 1.9625 | 5.0000 | 0.3836 | 13.26 | 13.22 | 14.3490 | 14.2286 | 59.36 | 47.94 | 45025.25 |
| No Retrieval LLM | Small | 0.9122 | 0.0878 | 1.9625 | 4.9500 | 0.3928 | 13.14 | 13.10 | 14.1479 | 14.0083 | 58.94 | 47.93 | 45016.05 |

---

## 3. Pairwise Approach Comparisons (Welch's t-test)

**How to read p-values, Cohen's d, and Sig.:**
- `p-value`: Probability of observing a difference at least this large if the two groups were truly the same (two-tailed Welch's t-test).
- `*`, `**`, `***`: `*` means `p < 0.05`, `**` means `p < 0.01`, `***` means `p < 0.001`.
- `Cohen's d`: Standardized effect size; rough guide: `0.2` small, `0.5` medium, `0.8+` large.
- `Sig.`: `Yes` when `p < 0.05`, otherwise `No`.
- Sign of `Cohen's d`: positive means the first-named group has a higher mean; negative means lower.

### Adaptive RAG vs Fixed RAG

| Metric | Adaptive RAG (mean +/- std) | Fixed RAG (mean +/- std) | Diff | p-value | Cohen's d | Sig. |
|--------|-------------------|--------------------|------|---------|-----------|------|
| hallucination_rate | 0.1074 +/- 0.2247 | 0.3442 +/- 0.1675 | -0.2368 | 0.0000*** | -1.186 | Yes |
| groundedness_score | 0.7865 +/- 0.2572 | 0.6558 +/- 0.1675 | 0.1306 | 0.0000*** | 0.596 | Yes |
| answer_relevance_1to5 | 4.3182 +/- 0.8625 | 4.8187 +/- 0.5000 | -0.5006 | 0.0000*** | -0.702 | Yes |
| context_relevance_1to5 | 3.7841 +/- 1.2644 | 4.1312 +/- 0.7940 | -0.3472 | 0.0026** | -0.325 | Yes |
| confidence | 0.5221 +/- 0.1017 | 0.5803 +/- 0.1096 | -0.0582 | 0.0000*** | -0.551 | Yes |
| query_coverage | 0.5554 +/- 0.1926 | 0.8276 +/- 0.1333 | -0.2722 | 0.0000*** | -1.630 | Yes |
| response_time_s | 23.5364 +/- 23.9140 | 13.0096 +/- 7.6581 | 10.5269 | 0.0000*** | 0.582 | Yes |
| llm_latency_s | 23.5137 +/- 23.9110 | 12.7321 +/- 7.6625 | 10.7815 | 0.0000*** | 0.596 | Yes |
| gpu_throughput_toks_per_s | 9.5788 +/- 4.6616 | 7.0776 +/- 1.6621 | 2.5012 | 0.0000*** | 0.702 | Yes |
| eff_gpu_throughput | 7.1045 +/- 4.5537 | 7.0776 +/- 1.6621 | 0.0269 | 0.9417 | 0.008 | No |
| gpu_util_percent | 67.2609 +/- 24.3415 | 59.4270 +/- 14.1365 | 7.8339 | 0.0003*** | 0.389 | Yes |
| gpu_mem_percent | 75.2519 +/- 18.5629 | 33.5064 +/- 24.3799 | 41.7455 | 0.0000*** | 1.939 | Yes |
| gpu_mem_peak_mb | 47336.7812 +/- 16821.0333 | 20396.2633 +/- 9317.0677 | 26940.5180 | 0.0000*** | 1.957 | Yes |
| total_deployment_cost_usd | 2.0196 +/- 0.0199 | 0.0000 +/- 0.0000 | 2.0196 | 0.0000*** | 140.013 | Yes |
| retrieved_docs_count | 3.4773 +/- 2.0113 | 3.5000 +/- 0.5016 | -0.0227 | 0.8848 | -0.015 | No |
| top_retrieval_score | 0.5221 +/- 0.1017 | 0.5162 +/- 0.1083 | 0.0059 | 0.6072 | 0.056 | No |
| avg_retrieval_score | 0.5039 +/- 0.0979 | 0.4598 +/- 0.0908 | 0.0441 | 0.0000*** | 0.466 | Yes |

**Takeaway:** Adaptive RAG has significantly lower hallucination and higher groundedness than Fixed RAG (`p<0.001`, large effects). Fixed RAG is faster on wall-clock and LLM latency (`p<0.001`, medium effects) and scores higher on answer relevance (`p<0.001`); context relevance is also higher for Fixed RAG here (`p<0.01`, small-to-medium effect). Adaptive RAG uses more GPU memory and reports higher deployment cost in this log.

### Adaptive RAG vs No Retrieval LLM

| Metric | Adaptive RAG (mean +/- std) | No Retrieval LLM (mean +/- std) | Diff | p-value | Cohen's d | Sig. |
|--------|-------------------|--------------------|------|---------|-----------|------|
| hallucination_rate | 0.1074 +/- 0.2247 | 0.9309 +/- 0.1960 | -0.8235 | 0.0000*** | -3.893 | Yes |
| groundedness_score | 0.7865 +/- 0.2572 | 0.0691 +/- 0.1960 | 0.7174 | 0.0000*** | 3.118 | Yes |
| answer_relevance_1to5 | 4.3182 +/- 0.8625 | 1.9625 +/- 0.8964 | 2.3557 | 0.0000*** | 2.681 | Yes |
| context_relevance_1to5 | 3.7841 +/- 1.2644 | 4.9750 +/- 0.1566 | -1.1909 | 0.0000*** | -1.292 | Yes |
| confidence | 0.5221 +/- 0.1017 | 0.3882 +/- 0.1255 | 0.1339 | 0.0000*** | 1.178 | Yes |
| query_coverage | 0.5554 +/- 0.1926 | 1.0000 +/- 0.0000 | -0.4446 | 0.0000*** | -3.188 | Yes |
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

**Takeaway:** Adaptive RAG strongly outperforms No Retrieval on factual reliability (much lower hallucination, much higher groundedness; `p<0.001`, very large effects) and on answer relevance (`p<0.001`). No Retrieval shows higher context relevance and full query coverage in the logs, but paired with near-zero groundedness this reflects a brittle setup, not better answers. No Retrieval is faster and has higher throughput (`p<0.001`); peak GPU memory is not significantly different (`p=0.237`).

### Fixed RAG vs No Retrieval LLM

| Metric | Fixed RAG (mean +/- std) | No Retrieval LLM (mean +/- std) | Diff | p-value | Cohen's d | Sig. |
|--------|-------------------|--------------------|------|---------|-----------|------|
| hallucination_rate | 0.3442 +/- 0.1675 | 0.9309 +/- 0.1960 | -0.5867 | 0.0000*** | -3.219 | Yes |
| groundedness_score | 0.6558 +/- 0.1675 | 0.0691 +/- 0.1960 | 0.5867 | 0.0000*** | 3.219 | Yes |
| answer_relevance_1to5 | 4.8187 +/- 0.5000 | 1.9625 +/- 0.8964 | 2.8562 | 0.0000*** | 3.935 | Yes |
| context_relevance_1to5 | 4.1312 +/- 0.7940 | 4.9750 +/- 0.1566 | -0.8438 | 0.0000*** | -1.474 | Yes |
| confidence | 0.5803 +/- 0.1096 | 0.3882 +/- 0.1255 | 0.1921 | 0.0000*** | 1.630 | Yes |
| query_coverage | 0.8276 +/- 0.1333 | 1.0000 +/- 0.0000 | -0.1724 | 0.0000*** | -1.829 | Yes |
| response_time_s | 13.0096 +/- 7.6581 | 13.2003 +/- 9.6963 | -0.1907 | 0.8454 | -0.022 | No |
| llm_latency_s | 12.7321 +/- 7.6625 | 13.1603 +/- 9.6963 | -0.4281 | 0.6616 | -0.049 | No |
| gpu_throughput_toks_per_s | 7.0776 +/- 1.6621 | 14.2484 +/- 9.9267 | -7.1708 | 0.0000*** | -1.008 | Yes |
| eff_gpu_throughput | 7.0776 +/- 1.6621 | 14.1185 +/- 9.7846 | -7.0409 | 0.0000*** | -1.003 | Yes |
| gpu_util_percent | 59.4270 +/- 14.1365 | 59.1534 +/- 14.6569 | 0.2736 | 0.8651 | 0.019 | No |
| gpu_mem_percent | 33.5064 +/- 24.3799 | 47.9347 +/- 18.9042 | -14.4283 | 0.0000*** | -0.661 | Yes |
| gpu_mem_peak_mb | 20396.2633 +/- 9317.0677 | 45020.6500 +/- 18824.8037 | -24624.3867 | 0.0000*** | -1.658 | Yes |
| total_deployment_cost_usd | 0.0000 +/- 0.0000 | 0.0092 +/- 0.0067 | -0.0092 | 0.0000*** | -1.925 | Yes |
| retrieved_docs_count | 3.5000 +/- 0.5016 | 2.0000 +/- 0.0000 | 1.5000 | 0.0000*** | 4.229 | Yes |
| top_retrieval_score | 0.5162 +/- 0.1083 | 0.3379 +/- 0.0660 | 0.1784 | 0.0000*** | 1.989 | Yes |
| avg_retrieval_score | 0.4598 +/- 0.0908 | 0.2102 +/- 0.0633 | 0.2496 | 0.0000*** | 3.189 | Yes |

**Takeaway:** Fixed RAG is dramatically better than No Retrieval on hallucination and groundedness (`p<0.001`, very large effects) and on answer relevance (`p<0.001`). Mean response time and LLM latency do not differ meaningfully between these two (`p>0.6`). No Retrieval achieves higher token throughput but at the cost of quality; Fixed RAG uses less peak GPU memory on average (`p<0.001`).

---

## 4. Model Comparison: Qwen-32B vs Llama-70B

| Metric | Llama-70B (mean) | Qwen-32B (mean) | Diff | p-value | Cohen's d | Sig. |
|--------|------------|------------|------|---------|-----------|------|
| hallucination_rate | 0.4286 | 0.4702 | -0.0416 | 0.2462 | -0.104 | No |
| groundedness_score | 0.5400 | 0.4858 | 0.0542 | 0.1088 | 0.144 | No |
| answer_relevance_1to5 | 3.9798 | 3.4597 | 0.5202 | 0.0001*** | 0.363 | Yes |
| context_relevance_1to5 | 4.4355 | 4.1250 | 0.3105 | 0.0006*** | 0.310 | Yes |
| confidence | 0.5168 | 0.4787 | 0.0381 | 0.0020** | 0.280 | Yes |
| query_coverage | 0.7872 | 0.7861 | 0.0011 | 0.9558 | 0.005 | No |
| response_time_s | 18.1358 | 15.4771 | 2.6587 | 0.0748 | 0.160 | No |
| llm_latency_s | 18.0269 | 15.3650 | 2.6619 | 0.0748 | 0.160 | No |
| gpu_throughput_toks_per_s | 7.5357 | 13.0209 | -5.4852 | 0.0000*** | -0.853 | Yes |
| eff_gpu_throughput | 6.7981 | 11.9187 | -5.1206 | 0.0000*** | -0.778 | Yes |
| gpu_util_percent | 63.6731 | 60.5639 | 3.1092 | 0.0670 | 0.165 | No |
| gpu_mem_percent | 58.2524 | 47.6948 | 10.5576 | 0.0000*** | 0.397 | Yes |
| gpu_mem_peak_mb | 46792.9602 | 29005.3448 | 17787.6154 | 0.0000*** | 1.010 | Yes |
| total_deployment_cost_usd | 0.7203 | 0.7189 | 0.0013 | 0.9878 | 0.001 | No |
| retrieved_docs_count | 3.0081 | 3.0081 | 0.0000 | 1.0000 | 0.000 | No |
| top_retrieval_score | 0.4608 | 0.4608 | -0.0000 | 0.9975 | -0.000 | No |
| avg_retrieval_score | 0.3948 | 0.3951 | -0.0003 | 0.9852 | -0.002 | No |

---

## 5. Quantization Impact: 4-bit vs 8-bit

| Metric | 4-bit (mean) | 8-bit (mean) | Diff | p-value | Cohen's d | Sig. |
|--------|------------|------------|------|---------|-----------|------|
| hallucination_rate | 0.4750 | 0.4239 | 0.0511 | 0.1545 | 0.128 | No |
| groundedness_score | 0.4913 | 0.5345 | -0.0432 | 0.2016 | -0.115 | No |
| answer_relevance_1to5 | 3.7177 | 3.7218 | -0.0040 | 0.9754 | -0.003 | No |
| context_relevance_1to5 | 4.2218 | 4.3387 | -0.1169 | 0.1990 | -0.116 | No |
| confidence | 0.4867 | 0.5087 | -0.0220 | 0.0743 | -0.161 | No |
| query_coverage | 0.7931 | 0.7801 | 0.0130 | 0.5307 | 0.056 | No |
| response_time_s | 12.8916 | 20.7213 | -7.8297 | 0.0000*** | -0.484 | Yes |
| llm_latency_s | 12.7685 | 20.6234 | -7.8548 | 0.0000*** | -0.485 | Yes |
| gpu_throughput_toks_per_s | 13.0691 | 7.4874 | 5.5817 | 0.0000*** | 0.871 | Yes |
| eff_gpu_throughput | 12.3340 | 6.3828 | 5.9512 | 0.0000*** | 0.930 | Yes |
| gpu_util_percent | 70.1206 | 54.1164 | 16.0042 | 0.0000*** | 0.934 | Yes |
| gpu_mem_percent | 54.8449 | 51.1024 | 3.7425 | 0.1243 | 0.138 | No |
| gpu_mem_peak_mb | 29723.3715 | 46074.9335 | -16351.5620 | 0.0000*** | -0.911 | Yes |
| total_deployment_cost_usd | 0.7168 | 0.7223 | -0.0055 | 0.9496 | -0.006 | No |
| retrieved_docs_count | 2.8468 | 3.1694 | -0.3226 | 0.0109* | -0.230 | Yes |
| top_retrieval_score | 0.4619 | 0.4597 | 0.0022 | 0.8446 | 0.018 | No |
| avg_retrieval_score | 0.3976 | 0.3923 | 0.0053 | 0.7011 | 0.034 | No |

---

## 6. Corpus Size Impact: Small vs Medium

| Metric | Medium (mean) | Small (mean) | Diff | p-value | Cohen's d | Sig. |
|--------|------------|------------|------|---------|-----------|------|
| hallucination_rate | 0.4788 | 0.4181 | 0.0607 | 0.0911 | 0.152 | No |
| groundedness_score | 0.4794 | 0.5487 | -0.0693 | 0.0406* | -0.185 | Yes |
| answer_relevance_1to5 | 3.7148 | 3.7250 | -0.0102 | 0.9381 | -0.007 | No |
| context_relevance_1to5 | 4.0859 | 4.4875 | -0.4016 | 0.0000*** | -0.404 | Yes |
| confidence | 0.4740 | 0.5230 | -0.0490 | 0.0001*** | -0.362 | Yes |
| query_coverage | 0.7806 | 0.7930 | -0.0124 | 0.5483 | -0.054 | No |
| response_time_s | 16.0616 | 17.6009 | -1.5392 | 0.3032 | -0.093 | No |
| llm_latency_s | 15.9584 | 17.4827 | -1.5243 | 0.3085 | -0.092 | No |
| gpu_throughput_toks_per_s | 10.4441 | 10.1013 | 0.3428 | 0.5852 | 0.049 | No |
| eff_gpu_throughput | 9.5318 | 9.1734 | 0.3583 | 0.5722 | 0.051 | No |
| gpu_util_percent | 61.9695 | 62.2775 | -0.3081 | 0.8564 | -0.016 | No |
| gpu_mem_percent | 56.2216 | 49.5091 | 6.7125 | 0.0057** | 0.249 | Yes |
| gpu_mem_peak_mb | 39356.5215 | 36344.6255 | 3011.8960 | 0.0899 | 0.153 | No |
| total_deployment_cost_usd | 0.7596 | 0.6769 | 0.0827 | 0.3402 | 0.086 | No |
| retrieved_docs_count | 2.9688 | 3.0500 | -0.0812 | 0.5211 | -0.057 | No |
| top_retrieval_score | 0.4487 | 0.4736 | -0.0249 | 0.0290* | -0.197 | Yes |
| avg_retrieval_score | 0.3835 | 0.4072 | -0.0237 | 0.0882 | -0.154 | No |

---

## 7. Hallucination and Grounding Diagnostics

### 7.1 Hallucination Rate by Dimension

| Dimension | Value | Mean | Median | Std | % Zero | % Above 0.3 | n |
|-----------|-------|------|--------|-----|--------|-------------|---|
| approach | Adaptive RAG | 0.1074 | 0.0000 | 0.2247 | 75.0% | 15.3% | 176 |
| approach | Fixed RAG | 0.3442 | 0.3436 | 0.1675 | 2.5% | 57.5% | 160 |
| approach | No Retrieval LLM | 0.9309 | 1.0000 | 0.1960 | 3.8% | 96.2% | 160 |
| corpus_size | Medium | 0.4788 | 0.4391 | 0.3901 | 26.6% | 62.5% | 256 |
| corpus_size | Small | 0.4181 | 0.2703 | 0.4071 | 30.8% | 47.1% | 240 |
| model | Llama-70B | 0.4286 | 0.3155 | 0.4004 | 29.0% | 50.8% | 248 |
| model | Qwen-32B | 0.4702 | 0.4225 | 0.3976 | 28.2% | 59.3% | 248 |
| quantization | 4-bit | 0.4750 | 0.4241 | 0.4042 | 28.2% | 58.9% | 248 |
| quantization | 8-bit | 0.4239 | 0.3280 | 0.3931 | 29.0% | 51.2% | 248 |

### 7.2 Context Status Distribution by Approach

| Approach | grounded_context | partial_context | strong_context | weak_context |
|----------|--------|--------|--------|--------|
| Adaptive RAG | 61.4% | 38.6% | 0.0% | 0.0% |
| Fixed RAG | 0.0% | 69.4% | 25.6% | 5.0% |
| No Retrieval LLM | 3.1% | 0.6% | 0.0% | 96.2% |

### 7.3 General Knowledge Fallback Rate

| Approach | GK Usage % | n |
|----------|------------|---|
| Adaptive RAG | 38.64% | 176 |
| Fixed RAG | 0.00% | 160 |
| No Retrieval LLM | 100.00% | 160 |

---

## 8. Key Correlation Insights

| Metric A | Metric B | Pearson r |
|----------|----------|-----------|
| hallucination_rate | groundedness_score | -0.9743 |
| hallucination_rate | response_time_s | -0.1084 |
| groundedness_score | response_time_s | 0.0268 |
| answer_relevance_1to5 | context_relevance_1to5 | -0.2221 |
| hallucination_rate | retrieved_docs_count | -0.3869 |
| groundedness_score | retrieved_docs_count | 0.3360 |
| gpu_throughput_toks_per_s | response_time_s | -0.3361 |
| gpu_mem_peak_mb | gpu_throughput_toks_per_s | -0.2207 |
| confidence | hallucination_rate | -0.6786 |
| query_coverage | groundedness_score | -0.5690 |

---

## 9. Interaction Effects: Best and Worst Configurations

| Config | Hallucination | Groundedness | Ans. Relevance | Response Time (s) | GPU Mem Peak (MB) |
|--------|---------------|--------------|----------------|-------------------|-------------------|
| Adaptive RAG|Llama-70B|4-bit|Small | 0.0847 | 0.8492 | 4.8500 | 17.45 | 42138 |
| Adaptive RAG|Llama-70B|8-bit|Small | 0.0833 | 0.8322 | 4.9500 | 27.42 | 74563 |
| Adaptive RAG|Qwen-32B|8-bit|Small | 0.0644 | 0.8169 | 4.3000 | 44.69 | 43061 |
| Fixed RAG|Llama-70B|8-bit|Small | 0.1891 | 0.8109 | 4.5500 | 14.50 | 24469 |
| Adaptive RAG|Qwen-32B|4-bit|Small | 0.0714 | 0.7995 | 4.0500 | 13.86 | 27595 |
| Adaptive RAG|Llama-70B|4-bit|Medium | 0.1655 | 0.7730 | 4.7083 | 14.10 | 42130 |
| Adaptive RAG|Qwen-32B|8-bit|Medium | 0.1183 | 0.7583 | 3.4583 | 34.02 | 44004 |
| Adaptive RAG|Llama-70B|8-bit|Medium | 0.1174 | 0.7456 | 4.6667 | 27.36 | 74584 |
| Adaptive RAG|Qwen-32B|4-bit|Medium | 0.1333 | 0.7424 | 3.7083 | 10.94 | 30288 |
| Fixed RAG|Llama-70B|4-bit|Small | 0.2610 | 0.7390 | 4.7500 | 12.61 | 13587 |
| Fixed RAG|Qwen-32B|8-bit|Small | 0.2888 | 0.7112 | 4.6500 | 15.37 | 10106 |
| Fixed RAG|Llama-70B|8-bit|Medium | 0.3044 | 0.6956 | 4.9000 | 17.06 | 24035 |
| Fixed RAG|Qwen-32B|4-bit|Small | 0.3257 | 0.6743 | 4.7500 | 12.74 | 20554 |
| Fixed RAG|Llama-70B|4-bit|Medium | 0.4343 | 0.5657 | 5.0000 | 10.74 | 39965 |
| Fixed RAG|Qwen-32B|8-bit|Medium | 0.4474 | 0.5526 | 4.9500 | 13.10 | 10106 |
| Fixed RAG|Qwen-32B|4-bit|Medium | 0.5025 | 0.4975 | 5.0000 | 7.94 | 20348 |
| No Retrieval LLM|Llama-70B|8-bit|Small | 0.8446 | 0.1554 | 2.3500 | 14.60 | 71655 |
| No Retrieval LLM|Llama-70B|8-bit|Medium | 0.8671 | 0.1329 | 2.5000 | 14.03 | 71669 |
| No Retrieval LLM|Qwen-32B|8-bit|Small | 0.9119 | 0.0881 | 1.6000 | 10.27 | 49675 |
| No Retrieval LLM|Llama-70B|4-bit|Small | 0.9247 | 0.0753 | 2.1500 | 24.06 | 39041 |
| No Retrieval LLM|Qwen-32B|4-bit|Small | 0.9675 | 0.0325 | 1.7500 | 3.66 | 19693 |
| No Retrieval LLM|Qwen-32B|8-bit|Medium | 0.9725 | 0.0275 | 1.6500 | 12.24 | 49685 |
| No Retrieval LLM|Qwen-32B|4-bit|Medium | 0.9767 | 0.0233 | 1.6000 | 4.11 | 19693 |
| No Retrieval LLM|Llama-70B|4-bit|Medium | 0.9823 | 0.0177 | 2.1000 | 22.64 | 39054 |

- **Highest Groundedness**: `Adaptive RAG|Llama-70B|4-bit|Small` (0.8492)
- **Lowest Hallucination**: `Adaptive RAG|Qwen-32B|8-bit|Small` (0.0644)
- **Fastest Response**: `No Retrieval LLM|Qwen-32B|4-bit|Small` (3.66s)

---

## 10. Discussion and Recommendations

1. **Quality-first deployment**: Adaptive RAG is strongest on groundedness (0.7865) and Adaptive RAG is best on hallucination (0.1074).
2. **Latency-first deployment**: Fixed RAG is fastest on average (13.01s); this is the best baseline for strict response-time SLAs.
3. **Retrieval necessity**: Both Adaptive and Fixed retrieval approaches substantially outperform No Retrieval on groundedness/hallucination, supporting retrieval for factual QA workloads.
4. **Use-case-specific tuning**: If answer relevance is your primary metric and slight hallucination risk is acceptable, Fixed RAG may be attractive; otherwise Adaptive RAG is safer for factual reliability.

## 11. Threats to Validity and Notes

1. Dataset rows are treated as independent observations; repeated prompts across runs may introduce dependence.
2. Retrieval score scales can vary by architecture and pipeline settings; interpret retrieval-score comparisons with caution.
3. GPU/cost telemetry fields may include logging artifacts (including zeros) that should be validated against runtime logs.
4. Pairwise tests are uncorrected for multiple comparisons; use adjusted thresholds for publication-grade inferential claims.

## 12. Methodology Notes

- Total observations: 496
- Unique configurations: 24
- Statistical tests: Welch's t-test (two-tailed, unequal variance)
- Effect sizes: Cohen's d
- Significance markers: * p<0.05, ** p<0.01, *** p<0.001

*Report generated from 24 CSV files in this workspace using Pandas/SciPy.*