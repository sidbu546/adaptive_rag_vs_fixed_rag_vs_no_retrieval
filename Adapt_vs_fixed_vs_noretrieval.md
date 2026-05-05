# Adaptive RAG vs Fixed RAG vs No Retrieval LLM: Comprehensive Analysis Report

This report is generated from the **current full dataset collection** in this workspace and reflects all available adaptive, fixed, and no-retrieval runs.

## Executive Summary

- **Total observations**: 768
- **Unique configurations**: 36
- **Approaches analyzed**: Adaptive RAG, Fixed RAG, No Retrieval LLM

### Key Findings at a Glance

| Finding | Detail |
|---------|--------|
| Lowest Hallucination | **Fixed RAG|Qwen-32B|8-bit|Small** (0.0625) |
| Highest Groundedness | **Fixed RAG|Qwen-32B|8-bit|Small** (0.8979) |
| Fastest Response | **No Retrieval LLM|Qwen-32B|4-bit|Small** (8.24s) |

---

## 1. Dataset Coverage and Balance

| Approach | Model | Quantization | Corpus Size | source_file | n |
| --- | --- | --- | --- | --- | --- |
| Adaptive RAG | Llama-70B | 4-bit | Big | rag_eval_llama_4bit_big_adaptive.csv | 20 |
| Adaptive RAG | Llama-70B | 4-bit | Medium | rag_eval_llama_4bit_medium_adaptive.csv | 24 |
| Adaptive RAG | Llama-70B | 4-bit | Small | rag_eval_llama_4bit_small_adaptive.csv | 20 |
| Adaptive RAG | Llama-70B | 8-bit | Big | rag_eval_llama_8bit_big_adaptive.csv | 20 |
| Adaptive RAG | Llama-70B | 8-bit | Medium | rag_eval_llama_8bit_medium_adaptive.csv | 24 |
| Adaptive RAG | Llama-70B | 8-bit | Small | rag_eval_llama_8bit_small_adaptive.csv | 20 |
| Adaptive RAG | Qwen-32B | 4-bit | Big | rag_eval_qwen_4bit_big_adaptive.csv | 20 |
| Adaptive RAG | Qwen-32B | 4-bit | Medium | rag_eval_qwen_4bit_medium_adaptive.csv | 24 |
| Adaptive RAG | Qwen-32B | 4-bit | Small | rag_eval_qwen_4bit_small_adaptive.csv | 20 |
| Adaptive RAG | Qwen-32B | 8-bit | Big | rag_eval_qwen_8bit_big_adaptive.csv | 20 |
| Adaptive RAG | Qwen-32B | 8-bit | Medium | rag_eval_qwen_8bit_medium_adaptive.csv | 24 |
| Adaptive RAG | Qwen-32B | 8-bit | Small | rag_eval_qwen_8bit_small_adaptive.csv | 20 |
| Fixed RAG | Llama-70B | 4-bit | Big | fixed_rag_results_llama33_70b_4bit_big.csv | 20 |
| Fixed RAG | Llama-70B | 4-bit | Medium | fixed_rag_results_llama33_70b_4bit_medium.csv | 24 |
| Fixed RAG | Llama-70B | 4-bit | Small | fixed_rag_results_llama33_70b_4bit_small.csv | 20 |
| Fixed RAG | Llama-70B | 8-bit | Big | fixed_rag_results_llama33_70b_8bit_big.csv | 20 |
| Fixed RAG | Llama-70B | 8-bit | Medium | fixed_rag_results_llama33_70b_8bit_medium.csv | 24 |
| Fixed RAG | Llama-70B | 8-bit | Small | fixed_rag_results_llama33_70b_8bit_small.csv | 20 |
| Fixed RAG | Qwen-32B | 4-bit | Big | fixed_rag_results_qwen32b_4bit_big.csv | 20 |
| Fixed RAG | Qwen-32B | 4-bit | Medium | fixed_rag_results_qwen32b_4bit_medium.csv | 24 |
| Fixed RAG | Qwen-32B | 4-bit | Small | fixed_rag_results_qwen32b_4bit_small.csv | 20 |
| Fixed RAG | Qwen-32B | 8-bit | Big | fixed_rag_results_qwen32b_8bit_big.csv | 20 |
| Fixed RAG | Qwen-32B | 8-bit | Medium | fixed_rag_results_qwen32b_8bit_medium.csv | 24 |
| Fixed RAG | Qwen-32B | 8-bit | Small | fixed_rag_results_qwen32b_8bit_small.csv | 20 |
| No Retrieval LLM | Llama-70B | 4-bit | Big | noretrieval_llama_4bit_big.csv | 20 |
| No Retrieval LLM | Llama-70B | 4-bit | Medium | noretrieval_llama_4bit_medium.csv | 24 |
| No Retrieval LLM | Llama-70B | 4-bit | Small | noretrieval_llama_4bit_small.csv | 20 |
| No Retrieval LLM | Llama-70B | 8-bit | Big | noretrieval_llama_8bit_big.csv | 20 |
| No Retrieval LLM | Llama-70B | 8-bit | Medium | noretrieval_llama_8bit_medium.csv | 24 |
| No Retrieval LLM | Llama-70B | 8-bit | Small | noretrieval_llama_8bit_small.csv | 20 |
| No Retrieval LLM | Qwen-32B | 4-bit | Big | noretrieval_qwen_4bit_big.csv | 20 |
| No Retrieval LLM | Qwen-32B | 4-bit | Medium | noretrieval_qwen_4bit_medium.csv | 24 |
| No Retrieval LLM | Qwen-32B | 4-bit | Small | noretrieval_qwen_4bit_small.csv | 20 |
| No Retrieval LLM | Qwen-32B | 8-bit | Big | noretrieval_qwen_8bit_big.csv | 20 |
| No Retrieval LLM | Qwen-32B | 8-bit | Medium | noretrieval_qwen_8bit_medium.csv | 24 |
| No Retrieval LLM | Qwen-32B | 8-bit | Small | noretrieval_qwen_8bit_small.csv | 20 |

---

## 2. Overall Means by Approach

| Approach | hallucination_rate | groundedness_score | answer_relevance_1to5 | context_relevance_1to5 | confidence | query_coverage | response_time_s | llm_latency_s | gpu_throughput_toks_per_s | eff_gpu_throughput | gpu_util_percent | gpu_mem_percent | gpu_mem_peak_mb | total_deployment_cost_usd | retrieved_docs_count | top_retrieval_score | avg_retrieval_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adaptive RAG | 0.1958 | 0.6762 | 4.3281 | 3.707 | 0.5289 | 0.4897 | 27.6902 | 27.6597 | 9.8833 | 7.3017 | 67.7977 | 75.0938 | 46386.4277 | 2.0231 | 4.3438 | 0.5289 | 0.5103 |
| Fixed RAG | 0.306 | 0.6261 | 3.0938 | 3.8633 | 0.5808 | 0.8502 | 36.0784 | 10.9039 | 10.3417 | 10.3417 | 91.1469 | 62.1151 | 40530.5312 | 0.0 | 2.0 | 0.5385 | 0.5124 |
| No Retrieval LLM | 1.0 | 0.0 | 4.3359 | 1.0 | 0.0 | 0.7595 | 20.4888 | 16.5439 | 11.424 | 11.424 | 67.4054 | 69.0399 | 41232.8594 | 0.0 | 0.0 | 0.0 | 0.0 |

### 2.1 Overall Means by Approach and Corpus Size

| Approach | Corpus | hallucination_rate | groundedness_score | answer_relevance_1to5 | context_relevance_1to5 | confidence | query_coverage | response_time_s | llm_latency_s | gpu_throughput_toks_per_s | eff_gpu_throughput | gpu_util_percent | gpu_mem_percent | gpu_mem_peak_mb | total_deployment_cost_usd | retrieved_docs_count | top_retrieval_score | avg_retrieval_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Adaptive RAG | Big | 0.3903 | 0.4337 | 4.35 | 3.5375 | 0.5438 | 0.3451 | 36.8285 | 36.781 | 10.5532 | 7.7357 | 68.9785 | 74.7458 | 44295.65 | 2.0307 | 6.25 | 0.5438 | 0.5244 |
| Adaptive RAG | Medium | 0.1336 | 0.7548 | 4.1354 | 3.4583 | 0.501 | 0.5138 | 21.6071 | 21.5886 | 9.9595 | 7.6268 | 69.47 | 76.1489 | 47751.6562 | 2.018 | 3.3333 | 0.501 | 0.4829 |
| Adaptive RAG | Small | 0.076 | 0.8244 | 4.5375 | 4.175 | 0.5476 | 0.6054 | 25.8516 | 25.8237 | 9.1219 | 6.4777 | 64.61 | 74.1755 | 46838.9312 | 2.0215 | 3.65 | 0.5476 | 0.5292 |
| Fixed RAG | Big | 0.4542 | 0.45 | 2.55 | 3.7125 | 0.5068 | 0.8391 | 38.5323 | 13.028 | 10.5979 | 10.5979 | 91.3115 | 62.1227 | 40528.525 | 0.0 | 2.0 | 0.5836 | 0.5687 |
| Fixed RAG | Medium | 0.3208 | 0.6181 | 2.875 | 3.7188 | 0.5644 | 0.9011 | 34.6767 | 9.6498 | 10.1679 | 10.1679 | 90.7824 | 62.0952 | 40525.125 | 0.0 | 2.0 | 0.5136 | 0.4839 |
| Fixed RAG | Small | 0.14 | 0.8116 | 3.9 | 4.1875 | 0.6745 | 0.8 | 35.3065 | 10.2846 | 10.294 | 10.294 | 91.4197 | 62.1315 | 40539.025 | 0.0 | 2.0 | 0.5233 | 0.4904 |
| No Retrieval LLM | Big | 1.0 | 0.0 | 4.35 | 1.0 | 0.0 | 0.757 | 24.835 | 20.5271 | 11.2243 | 11.2243 | 66.5538 | 69.0902 | 41260.925 | 0.0 | 0.0 | 0.0 | 0.0 |
| No Retrieval LLM | Medium | 1.0 | 0.0 | 4.2396 | 1.0 | 0.0 | 0.7939 | 18.1811 | 14.3876 | 11.3734 | 11.3734 | 67.335 | 69.0142 | 41217.375 | 0.0 | 0.0 | 0.0 | 0.0 |
| No Retrieval LLM | Small | 1.0 | 0.0 | 4.4375 | 1.0 | 0.0 | 0.7207 | 18.912 | 15.1482 | 11.6844 | 11.6844 | 68.3416 | 69.0204 | 41223.375 | 0.0 | 0.0 | 0.0 | 0.0 |

---

## 3. Pairwise Approach Comparisons (Welch's t-test)

Welch's t-test is used for all pairwise comparisons to avoid equal-variance assumptions across systems. `Sig.` marks `p < 0.05`.

### Adaptive RAG vs Fixed RAG

| Metric | Adaptive RAG (mean +/- std) | Fixed RAG (mean +/- std) | Diff | p-value | Cohen's d | Sig. |
| --- | --- | --- | --- | --- | --- | --- |
| hallucination_rate | 0.1958 +/- 0.2809 | 0.3060 +/- 0.3583 | -0.1102 | 0.0001*** | -0.342 | Yes |
| groundedness_score | 0.6762 +/- 0.3006 | 0.6261 +/- 0.3598 | 0.0502 | 0.0875 | 0.151 | No |
| answer_relevance_1to5 | 4.3281 +/- 0.7832 | 3.0938 +/- 1.4332 | 1.2344 | 0.0000*** | 1.069 | Yes |
| context_relevance_1to5 | 3.7070 +/- 1.1932 | 3.8633 +/- 1.0705 | -0.1562 | 0.1195 | -0.138 | No |
| confidence | 0.5289 +/- 0.0944 | 0.5808 +/- 0.2093 | -0.0519 | 0.0003*** | -0.319 | Yes |
| query_coverage | 0.4897 +/- 0.1992 | 0.8502 +/- 0.1344 | -0.3605 | 0.0000*** | -2.122 | Yes |
| response_time_s | 27.6902 +/- 22.3331 | 36.0784 +/- 28.1549 | -8.3882 | 0.0002*** | -0.330 | Yes |
| llm_latency_s | 27.6597 +/- 22.3277 | 10.9039 +/- 8.9348 | 16.7558 | 0.0000*** | 0.985 | Yes |
| gpu_throughput_toks_per_s | 9.8833 +/- 4.5323 | 10.3417 +/- 5.4784 | -0.4584 | 0.3028 | -0.091 | No |
| eff_gpu_throughput | 7.3017 +/- 4.4644 | 10.3417 +/- 5.4784 | -3.0399 | 0.0000*** | -0.608 | Yes |
| gpu_util_percent | 67.7977 +/- 22.9357 | 91.1469 +/- 10.3573 | -23.3492 | 0.0000*** | -1.312 | Yes |
| gpu_mem_percent | 75.0938 +/- 17.5482 | 62.1151 +/- 28.0178 | 12.9786 | 0.0000*** | 0.555 | Yes |
| gpu_mem_peak_mb | 46386.4277 +/- 17515.5094 | 40530.5312 +/- 30310.2989 | 5855.8965 | 0.0077** | 0.237 | Yes |
| total_deployment_cost_usd | 2.0231 +/- 0.0186 | 0.0000 +/- 0.0000 | 2.0231 | 0.0000*** | 153.742 | Yes |
| retrieved_docs_count | 4.3438 +/- 2.2490 | 2.0000 +/- 0.0000 | 2.3438 | 0.0000*** | 1.474 | Yes |
| top_retrieval_score | 0.5289 +/- 0.0944 | 0.5385 +/- 0.1037 | -0.0096 | 0.2753 | -0.097 | No |
| avg_retrieval_score | 0.5103 +/- 0.0917 | 0.5124 +/- 0.0999 | -0.0021 | 0.8083 | -0.021 | No |

**Takeaway:** Adaptive is lower on hallucination (Diff=-0.1102, p=0.0001***) and higher on groundedness (Diff=0.0502, p=0.0875), with operational trade-offs visible in latency/memory columns.

### Adaptive RAG vs No Retrieval LLM

| Metric | Adaptive RAG (mean +/- std) | No Retrieval LLM (mean +/- std) | Diff | p-value | Cohen's d | Sig. |
| --- | --- | --- | --- | --- | --- | --- |
| hallucination_rate | 0.1958 +/- 0.2809 | 1.0000 +/- 0.0000 | -0.8042 | 0.0000*** | -4.048 | Yes |
| groundedness_score | 0.6762 +/- 0.3006 | 0.0000 +/- 0.0000 | 0.6762 | 0.0000*** | 3.181 | Yes |
| answer_relevance_1to5 | 4.3281 +/- 0.7832 | 4.3359 +/- 0.8192 | -0.0078 | 0.9122 | -0.010 | No |
| context_relevance_1to5 | 3.7070 +/- 1.1932 | 1.0000 +/- 0.0000 | 2.7070 | 0.0000*** | 3.209 | Yes |
| confidence | 0.5289 +/- 0.0944 | 0.0000 +/- 0.0000 | 0.5289 | 0.0000*** | 7.923 | Yes |
| query_coverage | 0.4897 +/- 0.1992 | 0.7595 +/- 0.1470 | -0.2698 | 0.0000*** | -1.541 | Yes |
| response_time_s | 27.6902 +/- 22.3331 | 20.4888 +/- 10.3504 | 7.2014 | 0.0000*** | 0.414 | Yes |
| llm_latency_s | 27.6597 +/- 22.3277 | 16.5439 +/- 9.1916 | 11.1158 | 0.0000*** | 0.651 | Yes |
| gpu_throughput_toks_per_s | 9.8833 +/- 4.5323 | 11.4240 +/- 4.6780 | -1.5408 | 0.0002*** | -0.335 | Yes |
| eff_gpu_throughput | 7.3017 +/- 4.4644 | 11.4240 +/- 4.6780 | -4.1223 | 0.0000*** | -0.902 | Yes |
| gpu_util_percent | 67.7977 +/- 22.9357 | 67.4054 +/- 18.5137 | 0.3922 | 0.8315 | 0.019 | No |
| gpu_mem_percent | 75.0938 +/- 17.5482 | 69.0399 +/- 15.7651 | 6.0539 | 0.0000*** | 0.363 | Yes |
| gpu_mem_peak_mb | 46386.4277 +/- 17515.5094 | 41232.8594 +/- 18675.4073 | 5153.5684 | 0.0014** | 0.285 | Yes |
| total_deployment_cost_usd | 2.0231 +/- 0.0186 | 0.0000 +/- 0.0000 | 2.0231 | 0.0000*** | 153.742 | Yes |
| retrieved_docs_count | 4.3438 +/- 2.2490 | 0.0000 +/- 0.0000 | 4.3438 | 0.0000*** | 2.731 | Yes |
| top_retrieval_score | 0.5289 +/- 0.0944 | 0.0000 +/- 0.0000 | 0.5289 | 0.0000*** | 7.923 | Yes |
| avg_retrieval_score | 0.5103 +/- 0.0917 | 0.0000 +/- 0.0000 | 0.5103 | 0.0000*** | 7.867 | Yes |

### Fixed RAG vs No Retrieval LLM

| Metric | Fixed RAG (mean +/- std) | No Retrieval LLM (mean +/- std) | Diff | p-value | Cohen's d | Sig. |
| --- | --- | --- | --- | --- | --- | --- |
| hallucination_rate | 0.3060 +/- 0.3583 | 1.0000 +/- 0.0000 | -0.6940 | 0.0000*** | -2.740 | Yes |
| groundedness_score | 0.6261 +/- 0.3598 | 0.0000 +/- 0.0000 | 0.6261 | 0.0000*** | 2.461 | Yes |
| answer_relevance_1to5 | 3.0938 +/- 1.4332 | 4.3359 +/- 0.8192 | -1.2422 | 0.0000*** | -1.064 | Yes |
| context_relevance_1to5 | 3.8633 +/- 1.0705 | 1.0000 +/- 0.0000 | 2.8633 | 0.0000*** | 3.783 | Yes |
| confidence | 0.5808 +/- 0.2093 | 0.0000 +/- 0.0000 | 0.5808 | 0.0000*** | 3.924 | Yes |
| query_coverage | 0.8502 +/- 0.1344 | 0.7595 +/- 0.1470 | 0.0907 | 0.0000*** | 0.644 | Yes |
| response_time_s | 36.0784 +/- 28.1549 | 20.4888 +/- 10.3504 | 15.5896 | 0.0000*** | 0.735 | Yes |
| llm_latency_s | 10.9039 +/- 8.9348 | 16.5439 +/- 9.1916 | -5.6400 | 0.0000*** | -0.622 | Yes |
| gpu_throughput_toks_per_s | 10.3417 +/- 5.4784 | 11.4240 +/- 4.6780 | -1.0823 | 0.0166* | -0.212 | Yes |
| eff_gpu_throughput | 10.3417 +/- 5.4784 | 11.4240 +/- 4.6780 | -1.0823 | 0.0166* | -0.212 | Yes |
| gpu_util_percent | 91.1469 +/- 10.3573 | 67.4054 +/- 18.5137 | 23.7415 | 0.0000*** | 1.583 | Yes |
| gpu_mem_percent | 62.1151 +/- 28.0178 | 69.0399 +/- 15.7651 | -6.9247 | 0.0006*** | -0.305 | Yes |
| gpu_mem_peak_mb | 40530.5312 +/- 30310.2989 | 41232.8594 +/- 18675.4073 | -702.3281 | 0.7524 | -0.028 | No |
| total_deployment_cost_usd | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 | NA | NA | No |
| retrieved_docs_count | 2.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 2.0000 | 0.0000*** | NA | Yes |
| top_retrieval_score | 0.5385 +/- 0.1037 | 0.0000 +/- 0.0000 | 0.5385 | 0.0000*** | 7.345 | Yes |
| avg_retrieval_score | 0.5124 +/- 0.0999 | 0.0000 +/- 0.0000 | 0.5124 | 0.0000*** | 7.250 | Yes |

---

## 4. Model Comparison: Llama-70B vs Qwen-32B

| Metric | Llama-70B (mean +/- std) | Qwen-32B (mean +/- std) | Diff | p-value | Cohen's d | Sig. |
| --- | --- | --- | --- | --- | --- | --- |
| hallucination_rate | 0.5418 +/- 0.4463 | 0.4594 +/- 0.4354 | 0.0824 | 0.0098** | 0.187 | Yes |
| groundedness_score | 0.3983 +/- 0.4141 | 0.4699 +/- 0.4026 | -0.0715 | 0.0154* | -0.175 | Yes |
| answer_relevance_1to5 | 4.2370 +/- 1.2490 | 3.6016 +/- 1.0694 | 0.6354 | 0.0000*** | 0.547 | Yes |
| context_relevance_1to5 | 3.1589 +/- 1.8046 | 2.5547 +/- 1.3175 | 0.6042 | 0.0000*** | 0.382 | Yes |
| confidence | 0.3488 +/- 0.2868 | 0.3910 +/- 0.3001 | -0.0422 | 0.0465* | -0.144 | Yes |
| query_coverage | 0.7134 +/- 0.2259 | 0.6862 +/- 0.2200 | 0.0272 | 0.0913 | 0.122 | No |
| response_time_s | 27.4474 +/- 18.6938 | 28.7242 +/- 25.7348 | -1.2768 | 0.4318 | -0.057 | No |
| llm_latency_s | 18.9706 +/- 15.7265 | 17.7678 +/- 17.0423 | 1.2028 | 0.3098 | 0.073 | No |
| gpu_throughput_toks_per_s | 9.2126 +/- 2.3534 | 11.8867 +/- 6.3200 | -2.6741 | 0.0000*** | -0.561 | Yes |
| eff_gpu_throughput | 8.5430 +/- 3.0305 | 10.8353 +/- 6.4893 | -2.2924 | 0.0000*** | -0.453 | Yes |
| gpu_util_percent | 80.1742 +/- 24.8027 | 70.7258 +/- 15.3957 | 9.4483 | 0.0000*** | 0.458 | Yes |
| gpu_mem_percent | 81.3071 +/- 12.5646 | 56.1921 +/- 21.8049 | 25.1150 | 0.0000*** | 1.411 | Yes |
| gpu_mem_peak_mb | 59532.0352 +/- 19848.9127 | 25901.1771 +/- 10042.0635 | 33630.8581 | 0.0000*** | 2.138 | Yes |
| total_deployment_cost_usd | 0.6740 +/- 0.9545 | 0.6747 +/- 0.9554 | -0.0006 | 0.9926 | -0.001 | No |
| retrieved_docs_count | 2.1146 +/- 2.2007 | 2.1146 +/- 2.2007 | 0.0000 | 1.0000 | 0.000 | No |
| top_retrieval_score | 0.3558 +/- 0.2646 | 0.3558 +/- 0.2646 | -0.0000 | 1.0000 | -0.000 | No |
| avg_retrieval_score | 0.3409 +/- 0.2538 | 0.3409 +/- 0.2538 | -0.0000 | 1.0000 | -0.000 | No |

---

## 5. Quantization Impact: 4-bit vs 8-bit

| Metric | 4-bit (mean +/- std) | 8-bit (mean +/- std) | Diff | p-value | Cohen's d | Sig. |
| --- | --- | --- | --- | --- | --- | --- |
| hallucination_rate | 0.5275 +/- 0.4359 | 0.4737 +/- 0.4479 | 0.0538 | 0.0922 | 0.122 | No |
| groundedness_score | 0.4173 +/- 0.4101 | 0.4509 +/- 0.4091 | -0.0337 | 0.2550 | -0.082 | No |
| answer_relevance_1to5 | 3.8646 +/- 1.2855 | 3.9740 +/- 1.1169 | -0.1094 | 0.2086 | -0.091 | No |
| context_relevance_1to5 | 2.8229 +/- 1.5699 | 2.8906 +/- 1.6458 | -0.0677 | 0.5598 | -0.042 | No |
| confidence | 0.3617 +/- 0.2902 | 0.3781 +/- 0.2981 | -0.0164 | 0.4408 | -0.056 | No |
| query_coverage | 0.7035 +/- 0.2261 | 0.6961 +/- 0.2206 | 0.0074 | 0.6450 | 0.033 | No |
| response_time_s | 16.3739 +/- 8.5509 | 39.7978 +/- 25.7757 | -23.4239 | 0.0000*** | -1.220 | Yes |
| llm_latency_s | 11.3963 +/- 8.5867 | 25.3421 +/- 19.1640 | -13.9458 | 0.0000*** | -0.939 | Yes |
| gpu_throughput_toks_per_s | 14.5199 +/- 3.8426 | 6.5794 +/- 1.6411 | 7.9404 | 0.0000*** | 2.688 | Yes |
| eff_gpu_throughput | 13.8549 +/- 3.8492 | 5.5234 +/- 2.0774 | 8.3314 | 0.0000*** | 2.694 | Yes |
| gpu_util_percent | 84.8003 +/- 12.1717 | 66.0997 +/- 23.9476 | 18.7007 | 0.0000*** | 0.984 | Yes |
| gpu_mem_percent | 68.4006 +/- 20.6277 | 69.0986 +/- 22.8863 | -0.6980 | 0.6572 | -0.032 | No |
| gpu_mem_peak_mb | 31538.9115 +/- 9504.9344 | 53894.3008 +/- 26850.2942 | -22355.3893 | 0.0000*** | -1.110 | Yes |
| total_deployment_cost_usd | 0.6715 +/- 0.9509 | 0.6772 +/- 0.9591 | -0.0057 | 0.9339 | -0.006 | No |
| retrieved_docs_count | 2.1146 +/- 2.2007 | 2.1146 +/- 2.2007 | 0.0000 | 1.0000 | 0.000 | No |
| top_retrieval_score | 0.3558 +/- 0.2646 | 0.3558 +/- 0.2646 | -0.0000 | 1.0000 | -0.000 | No |
| avg_retrieval_score | 0.3409 +/- 0.2538 | 0.3409 +/- 0.2538 | -0.0000 | 1.0000 | -0.000 | No |

---

## 6. Corpus Size Effects

### Small vs Medium
| Metric | Small (mean +/- std) | Medium (mean +/- std) | Diff | p-value | Cohen's d | Sig. |
| --- | --- | --- | --- | --- | --- | --- |
| hallucination_rate | 0.4053 +/- 0.4641 | 0.4848 +/- 0.4554 | -0.0795 | 0.0487* | -0.173 | Yes |
| groundedness_score | 0.5454 +/- 0.4457 | 0.4576 +/- 0.4244 | 0.0877 | 0.0218* | 0.202 | Yes |
| answer_relevance_1to5 | 4.2917 +/- 1.0422 | 3.7500 +/- 1.2797 | 0.5417 | 0.0000*** | 0.460 | Yes |
| context_relevance_1to5 | 3.1208 +/- 1.7254 | 2.7257 +/- 1.6050 | 0.3951 | 0.0071** | 0.238 | Yes |
| confidence | 0.4073 +/- 0.3153 | 0.3551 +/- 0.2917 | 0.0522 | 0.0505 | 0.173 | No |
| query_coverage | 0.7087 +/- 0.1907 | 0.7363 +/- 0.2154 | -0.0276 | 0.1196 | -0.135 | No |
| response_time_s | 26.6900 +/- 21.6326 | 24.8216 +/- 23.3227 | 1.8684 | 0.3407 | 0.083 | No |
| llm_latency_s | 17.0855 +/- 16.8003 | 15.2087 +/- 16.0134 | 1.8768 | 0.1923 | 0.115 | No |
| gpu_throughput_toks_per_s | 10.3668 +/- 5.0754 | 10.5003 +/- 5.0028 | -0.1335 | 0.7621 | -0.027 | No |
| eff_gpu_throughput | 9.4854 +/- 5.3804 | 9.7227 +/- 5.1830 | -0.2373 | 0.6081 | -0.045 | No |
| gpu_util_percent | 74.7904 +/- 21.9013 | 75.8625 +/- 21.2488 | -1.0720 | 0.5705 | -0.050 | No |
| gpu_mem_percent | 68.4425 +/- 22.0720 | 69.0861 +/- 22.0963 | -0.6436 | 0.7389 | -0.029 | No |
| gpu_mem_peak_mb | 42867.1104 +/- 23013.5834 | 43164.7188 +/- 22867.0077 | -297.6083 | 0.8821 | -0.013 | No |
| total_deployment_cost_usd | 0.6738 +/- 0.9550 | 0.6727 +/- 0.9530 | 0.0012 | 0.9887 | 0.001 | No |
| retrieved_docs_count | 1.8833 +/- 1.8304 | 1.7778 +/- 1.8458 | 0.1056 | 0.5113 | 0.057 | No |
| top_retrieval_score | 0.3569 +/- 0.2647 | 0.3382 +/- 0.2553 | 0.0188 | 0.4100 | 0.072 | No |
| avg_retrieval_score | 0.3399 +/- 0.2527 | 0.3222 +/- 0.2421 | 0.0176 | 0.4170 | 0.071 | No |

### Small vs Big
| Metric | Small (mean +/- std) | Big (mean +/- std) | Diff | p-value | Cohen's d | Sig. |
| --- | --- | --- | --- | --- | --- | --- |
| hallucination_rate | 0.4053 +/- 0.4641 | 0.6148 +/- 0.3760 | -0.2095 | 0.0000*** | -0.496 | Yes |
| groundedness_score | 0.5454 +/- 0.4457 | 0.2946 +/- 0.3019 | 0.2508 | 0.0000*** | 0.659 | Yes |
| answer_relevance_1to5 | 4.2917 +/- 1.0422 | 3.7500 +/- 1.1839 | 0.5417 | 0.0000*** | 0.486 | Yes |
| context_relevance_1to5 | 3.1208 +/- 1.7254 | 2.7500 +/- 1.4565 | 0.3708 | 0.0113* | 0.232 | Yes |
| confidence | 0.4073 +/- 0.3153 | 0.3502 +/- 0.2719 | 0.0572 | 0.0339* | 0.194 | Yes |
| query_coverage | 0.7087 +/- 0.1907 | 0.6471 +/- 0.2518 | 0.0616 | 0.0027** | 0.276 | Yes |
| response_time_s | 26.6900 +/- 21.6326 | 33.3986 +/- 21.4087 | -6.7086 | 0.0007*** | -0.312 | Yes |
| llm_latency_s | 17.0855 +/- 16.8003 | 23.4454 +/- 15.2704 | -6.3599 | 0.0000*** | -0.396 | Yes |
| gpu_throughput_toks_per_s | 10.3668 +/- 5.0754 | 10.7918 +/- 4.7665 | -0.4250 | 0.3448 | -0.086 | No |
| eff_gpu_throughput | 9.4854 +/- 5.3804 | 9.8526 +/- 5.0144 | -0.3673 | 0.4396 | -0.071 | No |
| gpu_util_percent | 74.7904 +/- 21.9013 | 75.6146 +/- 20.3680 | -0.8242 | 0.6696 | -0.039 | No |
| gpu_mem_percent | 68.4425 +/- 22.0720 | 68.6529 +/- 21.1639 | -0.2104 | 0.9151 | -0.010 | No |
| gpu_mem_peak_mb | 42867.1104 +/- 23013.5834 | 42028.3667 +/- 23308.5519 | 838.7438 | 0.6918 | 0.036 | No |
| total_deployment_cost_usd | 0.6738 +/- 0.9550 | 0.6769 +/- 0.9593 | -0.0030 | 0.9722 | -0.003 | No |
| retrieved_docs_count | 1.8833 +/- 1.8304 | 2.7500 +/- 2.7359 | -0.8667 | 0.0001*** | -0.372 | Yes |
| top_retrieval_score | 0.3569 +/- 0.2647 | 0.3758 +/- 0.2744 | -0.0189 | 0.4440 | -0.070 | No |
| avg_retrieval_score | 0.3399 +/- 0.2527 | 0.3644 +/- 0.2667 | -0.0245 | 0.3021 | -0.094 | No |

### Medium vs Big
| Metric | Medium (mean +/- std) | Big (mean +/- std) | Diff | p-value | Cohen's d | Sig. |
| --- | --- | --- | --- | --- | --- | --- |
| hallucination_rate | 0.4848 +/- 0.4554 | 0.6148 +/- 0.3760 | -0.1301 | 0.0004*** | -0.309 | Yes |
| groundedness_score | 0.4576 +/- 0.4244 | 0.2946 +/- 0.3019 | 0.1631 | 0.0000*** | 0.436 | Yes |
| answer_relevance_1to5 | 3.7500 +/- 1.2797 | 3.7500 +/- 1.1839 | 0.0000 | 1.0000 | 0.000 | No |
| context_relevance_1to5 | 2.7257 +/- 1.6050 | 2.7500 +/- 1.4565 | -0.0243 | 0.8554 | -0.016 | No |
| confidence | 0.3551 +/- 0.2917 | 0.3502 +/- 0.2719 | 0.0049 | 0.8410 | 0.017 | No |
| query_coverage | 0.7363 +/- 0.2154 | 0.6471 +/- 0.2518 | 0.0892 | 0.0000*** | 0.383 | Yes |
| response_time_s | 24.8216 +/- 23.3227 | 33.3986 +/- 21.4087 | -8.5770 | 0.0000*** | -0.382 | Yes |
| llm_latency_s | 15.2087 +/- 16.0134 | 23.4454 +/- 15.2704 | -8.2367 | 0.0000*** | -0.525 | Yes |
| gpu_throughput_toks_per_s | 10.5003 +/- 5.0028 | 10.7918 +/- 4.7665 | -0.2915 | 0.4942 | -0.060 | No |
| eff_gpu_throughput | 9.7227 +/- 5.1830 | 9.8526 +/- 5.0144 | -0.1299 | 0.7704 | -0.025 | No |
| gpu_util_percent | 75.8625 +/- 21.2488 | 75.6146 +/- 20.3680 | 0.2479 | 0.8915 | 0.012 | No |
| gpu_mem_percent | 69.0861 +/- 22.0963 | 68.6529 +/- 21.1639 | 0.4332 | 0.8185 | 0.020 | No |
| gpu_mem_peak_mb | 43164.7188 +/- 22867.0077 | 42028.3667 +/- 23308.5519 | 1136.3521 | 0.5739 | 0.049 | No |
| total_deployment_cost_usd | 0.6727 +/- 0.9530 | 0.6769 +/- 0.9593 | -0.0042 | 0.9597 | -0.004 | No |
| retrieved_docs_count | 1.7778 +/- 1.8458 | 2.7500 +/- 2.7359 | -0.9722 | 0.0000*** | -0.424 | Yes |
| top_retrieval_score | 0.3382 +/- 0.2553 | 0.3758 +/- 0.2744 | -0.0376 | 0.1061 | -0.142 | No |
| avg_retrieval_score | 0.3222 +/- 0.2421 | 0.3644 +/- 0.2667 | -0.0421 | 0.0603 | -0.166 | No |

---

## 7. Hallucination and Grounding Diagnostics

### 7.1 Hallucination Rate by Dimension
| Dimension | Value | Mean | Median | Std | % Zero | % Above 0.3 | n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| approach | Adaptive RAG | 0.1958 | 0.0000 | 0.2809 | 57.4% | 28.5% | 256 |
| approach | Fixed RAG | 0.3060 | 0.1667 | 0.3583 | 45.7% | 43.0% | 256 |
| approach | No Retrieval LLM | 1.0000 | 1.0000 | 0.0000 | 0.0% | 100.0% | 256 |
| model | Llama-70B | 0.5418 | 0.5000 | 0.4463 | 32.3% | 61.2% | 384 |
| model | Qwen-32B | 0.4594 | 0.3333 | 0.4354 | 36.5% | 53.1% | 384 |
| quantization | 4-bit | 0.5275 | 0.5000 | 0.4359 | 30.7% | 61.2% | 384 |
| quantization | 8-bit | 0.4737 | 0.4000 | 0.4479 | 38.0% | 53.1% | 384 |
| corpus size | Big | 0.6148 | 0.6000 | 0.3760 | 12.1% | 74.6% | 240 |
| corpus size | Medium | 0.4848 | 0.4000 | 0.4554 | 38.5% | 53.8% | 288 |
| corpus size | Small | 0.4053 | 0.0000 | 0.4641 | 51.7% | 43.8% | 240 |

### 7.2 Context Status Distribution by Approach
| Approach | grounded_context | no_context | partial_context | strong_context | weak_context |
| --- | --- | --- | --- | --- | --- |
| Adaptive RAG | 57.81 | 0.0 | 42.19 | 0.0 | 0.0 |
| Fixed RAG | 0.0 | 0.0 | 60.94 | 35.94 | 3.12 |
| No Retrieval LLM | 0.0 | 100.0 | 0.0 | 0.0 | 0.0 |

### 7.3 General Knowledge Fallback Rate
| Approach | GK Usage % | n |
| --- | --- | --- |
| Adaptive RAG | 42.19% | 256 |
| Fixed RAG | 0.00% | 256 |
| No Retrieval LLM | 100.00% | 256 |

---

## 8. Key Correlation Insights

| Metric A | Metric B | Pearson r |
| --- | --- | --- |
| hallucination_rate | groundedness_score | -0.9684 |
| hallucination_rate | response_time_s | -0.2304 |
| groundedness_score | response_time_s | 0.1727 |
| answer_relevance_1to5 | context_relevance_1to5 | 0.0048 |
| hallucination_rate | retrieved_docs_count | -0.5142 |
| groundedness_score | retrieved_docs_count | 0.4260 |
| gpu_throughput_toks_per_s | response_time_s | -0.5088 |
| gpu_mem_peak_mb | gpu_throughput_toks_per_s | -0.4432 |
| confidence | hallucination_rate | -0.8936 |
| query_coverage | groundedness_score | -0.1415 |

---

## 9. Interaction Effects: Best and Worst Configurations

| Config | hallucination_rate | groundedness_score | answer_relevance_1to5 | response_time_s | gpu_mem_peak_mb |
| --- | --- | --- | --- | --- | --- |
| Fixed RAG\|Qwen-32B\|8-bit\|Small | 0.0625 | 0.8979 | 3.70 | 60.87 | 11957 |
| Fixed RAG\|Qwen-32B\|4-bit\|Small | 0.0821 | 0.8705 | 3.70 | 19.54 | 20294 |
| Adaptive RAG\|Llama-70B\|4-bit\|Small | 0.0847 | 0.8492 | 4.85 | 17.45 | 42138 |
| Adaptive RAG\|Llama-70B\|8-bit\|Small | 0.0833 | 0.8322 | 4.95 | 27.42 | 74563 |
| Adaptive RAG\|Qwen-32B\|8-bit\|Small | 0.0644 | 0.8169 | 4.30 | 44.69 | 43061 |
| Adaptive RAG\|Qwen-32B\|4-bit\|Small | 0.0714 | 0.7995 | 4.05 | 13.86 | 27595 |
| Fixed RAG\|Llama-70B\|8-bit\|Small | 0.1405 | 0.7988 | 4.50 | 42.08 | 89859 |
| Fixed RAG\|Qwen-32B\|8-bit\|Medium | 0.1472 | 0.7890 | 2.46 | 60.81 | 11957 |
| Adaptive RAG\|Llama-70B\|4-bit\|Medium | 0.1655 | 0.7730 | 4.71 | 14.10 | 42130 |
| Fixed RAG\|Qwen-32B\|4-bit\|Medium | 0.1916 | 0.7631 | 2.58 | 16.33 | 20265 |
| Adaptive RAG\|Qwen-32B\|8-bit\|Medium | 0.1183 | 0.7583 | 3.46 | 34.02 | 44004 |
| Adaptive RAG\|Llama-70B\|8-bit\|Medium | 0.1174 | 0.7456 | 4.67 | 27.36 | 74584 |
| Adaptive RAG\|Qwen-32B\|4-bit\|Medium | 0.1333 | 0.7424 | 3.71 | 10.94 | 30288 |
| Fixed RAG\|Llama-70B\|4-bit\|Small | 0.2750 | 0.6792 | 3.70 | 18.72 | 40046 |
| Fixed RAG\|Llama-70B\|8-bit\|Medium | 0.3374 | 0.5864 | 4.00 | 45.52 | 89859 |
| Fixed RAG\|Qwen-32B\|8-bit\|Big | 0.3674 | 0.5626 | 2.45 | 68.86 | 11957 |
| Fixed RAG\|Qwen-32B\|4-bit\|Big | 0.3577 | 0.5506 | 2.35 | 16.46 | 20254 |
| Adaptive RAG\|Llama-70B\|8-bit\|Big | 0.3178 | 0.4763 | 4.35 | 45.14 | 74516 |
| Adaptive RAG\|Qwen-32B\|8-bit\|Big | 0.3326 | 0.4354 | 4.10 | 52.19 | 37357 |
| Adaptive RAG\|Qwen-32B\|4-bit\|Big | 0.3742 | 0.4248 | 3.95 | 19.92 | 23233 |
| Adaptive RAG\|Llama-70B\|4-bit\|Big | 0.5367 | 0.3985 | 5.00 | 30.06 | 42076 |
| Fixed RAG\|Llama-70B\|8-bit\|Big | 0.4626 | 0.3827 | 3.10 | 46.52 | 89859 |
| Fixed RAG\|Llama-70B\|4-bit\|Medium | 0.6069 | 0.3340 | 2.46 | 16.05 | 40019 |
| Fixed RAG\|Llama-70B\|4-bit\|Big | 0.6292 | 0.3042 | 2.30 | 22.29 | 40044 |
| No Retrieval LLM\|Llama-70B\|4-bit\|Big | 1.0000 | 0.0000 | 5.00 | 19.86 | 39624 |
| No Retrieval LLM\|Llama-70B\|4-bit\|Medium | 1.0000 | 0.0000 | 4.71 | 16.50 | 39601 |
| No Retrieval LLM\|Llama-70B\|4-bit\|Small | 1.0000 | 0.0000 | 4.75 | 16.95 | 39604 |
| No Retrieval LLM\|Llama-70B\|8-bit\|Big | 1.0000 | 0.0000 | 4.15 | 36.72 | 71030 |
| No Retrieval LLM\|Llama-70B\|8-bit\|Medium | 1.0000 | 0.0000 | 4.54 | 26.71 | 71008 |
| No Retrieval LLM\|Llama-70B\|8-bit\|Small | 1.0000 | 0.0000 | 4.60 | 28.28 | 71012 |
| No Retrieval LLM\|Qwen-32B\|4-bit\|Big | 1.0000 | 0.0000 | 4.05 | 11.67 | 19976 |
| No Retrieval LLM\|Qwen-32B\|4-bit\|Medium | 1.0000 | 0.0000 | 3.83 | 8.89 | 19954 |
| No Retrieval LLM\|Qwen-32B\|4-bit\|Small | 1.0000 | 0.0000 | 4.10 | 8.24 | 19953 |
| No Retrieval LLM\|Qwen-32B\|8-bit\|Big | 1.0000 | 0.0000 | 4.20 | 31.08 | 34414 |
| No Retrieval LLM\|Qwen-32B\|8-bit\|Medium | 1.0000 | 0.0000 | 3.88 | 20.63 | 34306 |
| No Retrieval LLM\|Qwen-32B\|8-bit\|Small | 1.0000 | 0.0000 | 4.30 | 22.18 | 34324 |

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
