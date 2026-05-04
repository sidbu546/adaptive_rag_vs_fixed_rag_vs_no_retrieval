# Adaptive vs Fixed Retrieval-Augmented Generation:
## A Paired Evaluation of Quality, Efficiency, and Model-Family Effects

### Abstract

This paper studies retrieval-augmented generation as a policy design problem rather than a simple implementation comparison. The experiment compares Fixed RAG, which applies a uniform retrieval behavior to every query, against Adaptive RAG, which adjusts retrieval and answer strategy to the query or the retrieved context. Using paired query-level comparisons across model family, quantization, and dataset size, the results show that Adaptive RAG improves answer relevance by 1.242, reduces hallucination rate by 0.107 in absolute improvement terms, and raises a composite quality index by 0.131. These gains come with a mixed operational profile: Adaptive lowers end-to-end response time by 7.113 seconds of improvement on average, but increases GPU peak memory by 5762.8 MB. Model-family effects are conditional rather than absolute. Qwen 32B is the stronger conservative baseline overall on the aggregate quality index, while Llama 3.3 70B benefits more from the adaptive policy and becomes the stronger answer model under Adaptive RAG. The findings imply that retrieval should be understood as a dynamic evidence-use policy whose value depends on corpus ambiguity, model behavior, and deployment constraints.

**Keywords:** retrieval-augmented generation, adaptive retrieval, hallucination, groundedness, model comparison, quantization, systems evaluation


## 1. Introduction

This experiment is not merely benchmarking two implementations of the same pipeline. It is comparing two retrieval policies. Fixed RAG uses the same retrieval behavior regardless of query difficulty, ambiguity, context quality, or answer type. Adaptive RAG changes behavior depending on the query and/or the retrieved context. The central research question is therefore: **Does adaptive control over retrieval and answer strategy improve final answer quality enough to justify its operational tradeoffs?**

That makes this a systems-and-quality tradeoff problem, not just an accuracy comparison. The deeper design question is whether retrieval should be treated as a static pipeline stage or as a decision policy that can expand, constrain, route, or hybridize its answer strategy when evidence is partial or ambiguous.

## 2. Experimental Data, Audit, and Preprocessing

- Total raw rows loaded from eligible root CSV files: 512
- Total deduplicated rows after query normalization and within-condition averaging: 480
- Duplicate rows collapsed: 32
- Duplicate query groups detected: 32
- Paired query-level comparisons after normalization: 240
- Pair count before normalization: 160
- Pair count after normalization: 240
- Rows whose query text changed under normalization: 80
- Cost columns ignored: total_deployment_cost_usd

Rows per file:

| source_file | rows |
| --- | --- |
| fixed_rag_results_llama33_70b_4bit_big.csv | 20 |
| fixed_rag_results_llama33_70b_4bit_medium.csv | 24 |
| fixed_rag_results_llama33_70b_4bit_small.csv | 20 |
| fixed_rag_results_llama33_70b_8bit_big.csv | 20 |
| fixed_rag_results_llama33_70b_8bit_medium.csv | 24 |
| fixed_rag_results_llama33_70b_8bit_small.csv | 20 |
| fixed_rag_results_qwen32b_4bit_big.csv | 20 |
| fixed_rag_results_qwen32b_4bit_medium.csv | 24 |
| fixed_rag_results_qwen32b_4bit_small.csv | 20 |
| fixed_rag_results_qwen32b_8bit_big.csv | 20 |
| fixed_rag_results_qwen32b_8bit_medium.csv | 24 |
| fixed_rag_results_qwen32b_8bit_small.csv | 20 |
| rag_eval_llama_4bit_big_adaptive.csv | 20 |
| rag_eval_llama_4bit_medium_adaptive.csv | 24 |
| rag_eval_llama_4bit_small_adaptive.csv | 20 |
| rag_eval_llama_8bit_big_adaptive.csv | 20 |
| rag_eval_llama_8bit_medium_adaptive.csv | 24 |
| rag_eval_llama_8bit_small_adaptive.csv | 20 |
| rag_eval_qwen_4bit_big_adaptive.csv | 20 |
| rag_eval_qwen_4bit_medium_adaptive.csv | 24 |
| rag_eval_qwen_4bit_small_adaptive.csv | 20 |
| rag_eval_qwen_8bit_big_adaptive.csv | 20 |
| rag_eval_qwen_8bit_medium_adaptive.csv | 24 |
| rag_eval_qwen_8bit_small_adaptive.csv | 20 |

Schema validation: No required metric columns were missing in the eligible root CSV files.

Preprocessing matters conceptually because query normalization prevents artificial mismatches caused by formatting differences, deduplication prevents repeated queries from overweighting one condition, and paired matching ensures that Fixed and Adaptive are compared under the same model family, quantization, dataset size, and normalized query text. Because the normalized query set pairs cleanly for the comparative experiment, the paper reports the paired analysis directly rather than treating unmatched queries as a separate analytical object.

## 3. Paired Evaluation Framework

Averages alone are weak because they confound system behavior with query difficulty. A hard query may produce low scores for both systems, while an easy query may produce high scores for both. The correct comparison is therefore not simply Adaptive mean versus Fixed mean, but Adaptive performance on the same query and configuration minus Fixed performance on that same query and configuration.

**Paired deltas reveal whether adaptive RAG improves the answer relative to the fixed baseline under the same experimental conditions.**

This pairing controls for query difficulty, model family, quantization, dataset size, and corpus condition. Without that control, a system can look better simply because it saw an easier subset or a more favorable configuration mix.

## 4. Quality Metrics and Their Interpretation

### Hallucination rate

Hallucination rate measures unsupported or risky generation. Lower hallucination means the answer is less likely to contain claims that are not justified by retrieved evidence. However, very low hallucination is not always a pure win because a system can achieve it by becoming overly conservative and refusing to answer. That is why hallucination must be read together with answer relevance.

### Groundedness

Groundedness measures how strongly the final answer is supported by retrieved evidence. High groundedness means the answer stays anchored to context. Groundedness is not the same as relevance: a response can be perfectly grounded yet incomplete, overly cautious, or not especially useful to the user.

### Answer relevance

Answer relevance measures whether the final answer addresses the user’s question. This is the most user-facing quality metric in the set. A system can improve answer relevance by synthesizing evidence better, giving a fuller answer, or switching to a more appropriate answer mode for the question.

### Context relevance

Context relevance measures whether the retrieved evidence aligns with the query. This sits closer to retriever quality than generator quality. **Context relevance evaluates the evidence retrieved; answer relevance evaluates what the model does with that evidence.** High context relevance does not guarantee a strong answer if the model uses the evidence timidly or incompletely.

## 5. Retrieval Quality versus Answer Quality

Adaptive RAG can improve answer relevance even when context relevance does not improve by the same amount. That pattern matters because it shows that a RAG system is not only a retriever. It is a retriever plus a policy for using retrieved evidence.

In this experiment, the paired data show whether broader or different retrieval behavior actually translated into better answers. On average, Adaptive changed retrieved document count by 2.500, general-knowledge use by 0.417, and query coverage by -0.356 relative to Fixed. When answer relevance rises more than context relevance, the implication is that Adaptive is getting value from evidence use, synthesis, or answer-mode selection rather than from retrieval precision alone. Fixed RAG can be retrieval-precise but answer-poor; Adaptive RAG can be retrieval-broader but answer-stronger.

## 6. Operational Metrics and Systems Interpretation

### Response time

Response time is a user-facing latency metric. It measures how long the full system takes to return an answer, including retrieval, prompt construction, generation, and orchestration overhead. Lower is better for perceived responsiveness.

### GPU throughput

GPU throughput measures token-generation efficiency. Higher throughput means the model decodes faster once generation is underway. But higher throughput does not automatically imply lower end-to-end latency because total response time also depends on prompt length, retrieval overhead, answer length, and system overhead.

### GPU memory peak

GPU memory peak is the deployment footprint. It determines whether a system fits on available hardware and how expensive it is to scale. Lower memory broadens deployment feasibility.

### GPU utilization

GPU utilization measures how busy the GPU is. Higher utilization can indicate better hardware saturation, but it can also mean the system is simply more demanding. Lower utilization can reflect inefficiency or lower load. **Latency is a user-experience metric; throughput is a decoding-efficiency metric; memory is a deployment-feasibility metric; GPU utilization is a saturation metric.**

## 7. Why Latency and Throughput Can Diverge

Response time and throughput can disagree because they measure different parts of the system. Throughput measures speed during generation; response time measures the whole pipeline. A system can decode quickly but still answer slowly if it spends more time retrieving, constructing prompts, or generating longer responses. Likewise, a system can have lower throughput but lower latency if it retrieves less, answers briefly, or incurs less orchestration overhead.

That means a system with better throughput is not necessarily better for the user, and a system with lower response time is not necessarily more hardware-efficient.

## 8. Overall Comparison: Adaptive RAG versus Fixed RAG

### Quality metrics

- `hallucination_rate`: Fixed mean = 0.316, Adaptive mean = 0.209, Adaptive - Fixed = -0.107. Adaptive is better. This metric matters because it reflects unsupported or risky generation.
- `groundedness_score`: Fixed mean = 0.617, Adaptive mean = 0.661, Adaptive - Fixed = 0.044. Adaptive is better. This metric matters because it reflects how strongly the answer stays anchored to retrieved evidence.
- `answer_relevance_1to5`: Fixed mean = 3.087, Adaptive mean = 4.329, Adaptive - Fixed = 1.242. Adaptive is better. This metric matters because it reflects how directly the final answer addresses the user question.
- `context_relevance_1to5`: Fixed mean = 3.829, Adaptive mean = 3.683, Adaptive - Fixed = -0.146. Fixed is better. This metric matters because it reflects how aligned the retrieved evidence is with the query.
- `quality_index`: Fixed mean = 0.634, Adaptive mean = 0.765, Adaptive - Fixed = 0.131. Adaptive is better. This metric matters because it reflects a composite of groundedness, answer usefulness, and hallucination control.

The overall composite picture comes from `quality_index`: Fixed = 0.634, Adaptive = 0.765, delta = 0.131. This is the most compact statement of whether Adaptive improves the balance of grounding, relevance, and hallucination control rather than winning on a single narrow metric.

### Operational metrics

- `response_time_s`: Fixed mean = 35.874, Adaptive mean = 28.761, Adaptive - Fixed = -7.113. Adaptive is better. This metric matters because it reflects end-user latency across the whole pipeline.
- `gpu_util_percent`: Fixed mean = 91.148, Adaptive mean = 67.540, Adaptive - Fixed = -23.609. Fixed drives higher GPU saturation. This metric matters because it reflects GPU saturation rather than an intrinsic quality win.
- `gpu_throughput_toks_per_s`: Fixed mean = 10.325, Adaptive mean = 9.835, Adaptive - Fixed = -0.490. Fixed is better. This metric matters because it reflects decoding efficiency during generation.
- `gpu_mem_peak_mb`: Fixed mean = 40530.5, Adaptive mean = 46293.3, Adaptive - Fixed = 5762.8. Fixed is better. This metric matters because it reflects deployment footprint and hardware feasibility.

Operationally, these differences define the tradeoff surface. A quality gain is more valuable if it comes with neutral or improved latency and memory, and less attractive if it requires consistently more time or larger deployment footprint.

## 9. Factor-Level Results

### Dataset size

Small datasets can make fixed retrieval sufficient because the evidence space is easier to search cleanly. Medium and big datasets create more ambiguity, which increases the value of deciding how much evidence to retrieve and how to answer from it. In the computed results, the strongest dataset-size Adaptive gain on `quality_index` appears at medium (0.193), while the weakest appears at small (0.066). For `answer_relevance_1to5`, the strongest Adaptive gain appears at big (1.800), while the weakest appears at small (0.637). Conceptually, this tells us whether the adaptive advantage grows as the retrieval space becomes larger and noisier.

### Model family

Different models use retrieved context differently. Some models benefit more from adaptive retrieval because they synthesize evidence better; others may become less grounded when context broadens. In this experiment, the strongest model-family Adaptive gain on `quality_index` appears at Llama 3.3 70B (0.216), while the weakest appears at Qwen 32B (0.047). That pattern matters because the value of adaptive RAG depends not only on the policy but on the model’s ability to exploit that policy.

### Quantization

Quantization changes model capacity and efficiency. The strongest quantization-level Adaptive gain on `quality_index` appears at 4bit (0.169), while the weakest appears at 8bit (0.094). Conceptually, that tells us whether Adaptive compensates for lower precision or whether lower precision makes broader evidence harder to use cleanly.

### Full configuration

The full interaction among model family, quantization, and dataset size matters because Adaptive is not automatically better everywhere. The top configurations by Adaptive `quality_index` gain are:

- Llama 4bit medium: quality gain 0.432, answer relevance delta 2.200, hallucination delta -0.430, latency delta -0.695, memory delta 2114.7
- Llama 4bit big: quality gain 0.250, answer relevance delta 2.700, hallucination delta -0.092, latency delta 7.762, memory delta 2032.0
- Llama 8bit medium: quality gain 0.197, answer relevance delta 0.800, hallucination delta -0.264, latency delta -13.766, memory delta -15278.0

The configurations where Fixed remained more competitive are:

- Qwen 4bit small: quality gain -0.001, answer relevance delta 0.350, hallucination delta -0.011, latency delta -5.689, memory delta 7300.6
- Qwen 8bit small: quality gain 0.009, answer relevance delta 0.600, hallucination delta 0.002, latency delta -16.188, memory delta 31104.1
- Qwen 4bit big: quality gain 0.058, answer relevance delta 1.600, hallucination delta 0.016, latency delta 3.465, memory delta 2979.0

Those configuration-level deltas show where adaptive control helps most and where rigid retrieval remains adequate or safer.

## 10. Statistical Results

- `hallucination_rate`: mean delta -0.107, median delta 0.000, 95% bootstrap CI [-0.151, -0.063], Cohen's d -0.301. Wilcoxon p = 0.0000 Practically, this means adaptive improves at the paired query level only if the effect is directionally consistent enough to matter across matched conditions.
- `groundedness_score`: mean delta 0.044, median delta 0.000, 95% bootstrap CI [0.001, 0.090], Cohen's d 0.126. Wilcoxon p = 0.3042 Practically, this means adaptive improves at the paired query level only if the effect is directionally consistent enough to matter across matched conditions.
- `answer_relevance_1to5`: mean delta 1.242, median delta 1.000, 95% bootstrap CI [1.062, 1.425], Cohen's d 0.860. Wilcoxon p = 0.0000 Practically, this means adaptive improves at the paired query level only if the effect is directionally consistent enough to matter across matched conditions.
- `context_relevance_1to5`: mean delta -0.146, median delta 0.000, 95% bootstrap CI [-0.263, -0.033], Cohen's d -0.157. Wilcoxon p = 0.0147 Practically, this means fixed improves at the paired query level only if the effect is directionally consistent enough to matter across matched conditions.
- `quality_index`: mean delta 0.131, median delta 0.064, 95% bootstrap CI [0.098, 0.165], Cohen's d 0.481. Wilcoxon p = 0.0000 Practically, this means adaptive improves at the paired query level only if the effect is directionally consistent enough to matter across matched conditions.
- `response_time_s`: mean delta -7.113, median delta -3.953, 95% bootstrap CI [-10.380, -3.908], Cohen's d -0.277. Wilcoxon p = 0.0003 Practically, this means adaptive improves at the paired query level only if the effect is directionally consistent enough to matter across matched conditions.
- `gpu_util_percent`: mean delta -23.609, median delta -20.447, 95% bootstrap CI [-27.273, -20.091], Cohen's d -0.822. Wilcoxon p = 0.0000 Practically, this means lower saturation for adaptive at the paired query level only if the effect is directionally consistent enough to matter across matched conditions.
- `gpu_throughput_toks_per_s`: mean delta -0.490, median delta -0.115, 95% bootstrap CI [-0.780, -0.202], Cohen's d -0.218. Wilcoxon p = 0.0025 Practically, this means fixed improves at the paired query level only if the effect is directionally consistent enough to matter across matched conditions.
- `gpu_mem_peak_mb`: mean delta 5762.8, median delta 2410.0, 95% bootstrap CI [3693.0, 7762.5], Cohen's d 0.356. Wilcoxon p = 0.0000 Practically, this means fixed improves at the paired query level only if the effect is directionally consistent enough to matter across matched conditions.

Statistical significance does not automatically imply deployment superiority. A statistically consistent memory increase, for example, still counts as an operational drawback even if it is very stable.

## 11. Figures

![Combined quality profile](combined_quality_profile.png)

**Figure 1.** The combined quality profile places faithfulness, groundedness, normalized answer relevance, and normalized context relevance on a common higher-is-better scale. Faithfulness is used instead of hallucination rate so direction stays consistent. The shape shows whether a system’s quality advantage is broad or concentrated in one dimension.


![Combined operational profile](combined_operational_profile.png)

**Figure 2.** The combined operational profile compares latency efficiency, throughput efficiency, memory efficiency, and GPU utilization after per-dimension normalization. This is a relative efficiency profile, not a raw-value chart. GPU utilization should be read as saturation rather than an intrinsic good.


![Heatmap hallucination rate](heatmap_hallucination_rate.png)
![Heatmap groundedness score](heatmap_groundedness_score.png)
![Heatmap answer relevance](heatmap_answer_relevance_1to5.png)
![Heatmap context relevance](heatmap_context_relevance_1to5.png)

**Figures 3-6.** These heatmaps show mean Adaptive - Fixed deltas for the four quality metrics by configuration. Positive cells favor Adaptive for groundedness and relevance; negative cells favor Adaptive for hallucination because lower is better. The centered diverging scale makes it easy to see where Adaptive gains are broad versus conditional.


![Heatmap response time](heatmap_response_time_s.png)
![Heatmap GPU utilization](heatmap_gpu_util_percent.png)
![Heatmap throughput](heatmap_gpu_throughput_toks_per_s.png)
![Heatmap GPU memory](heatmap_gpu_mem_peak_mb.png)

**Figures 7-10.** These operational heatmaps show how Adaptive shifts latency, saturation, decoding speed, and memory footprint across the configuration grid. The important question is not whether every operational metric moves in the same direction, but whether quality gains arrive with acceptable infrastructure cost.


![Interaction answer relevance](interaction_answer_relevance_by_dataset_size.png)
![Interaction hallucination](interaction_hallucination_by_dataset_size.png)
![Interaction response time](interaction_response_time_by_dataset_size.png)
![Interaction GPU memory](interaction_gpu_memory_by_dataset_size.png)

**Figures 11-14.** The interaction plots show whether dataset size changes the behavior of each model-system combination. These figures matter because larger corpora increase ambiguity, so a widening Adaptive advantage with dataset size would support the idea that flexible retrieval matters most when evidence selection becomes harder.


![Tradeoff answer relevance vs groundedness](tradeoff_answer_vs_groundedness.png)
![Tradeoff answer relevance vs hallucination reduction](tradeoff_answer_vs_hallucination_reduction.png)
![Tradeoff latency vs memory](tradeoff_latency_vs_memory.png)
![Tradeoff quality gain vs latency](tradeoff_quality_gain_vs_latency.png)

**Figures 15-18.** The tradeoff plots move from metric-by-metric reporting to system-design space. Upper-right is desirable in the quality-vs-quality plots because Adaptive improves multiple answer dimensions together. Lower-left is desirable in the latency-vs-memory plot because Adaptive becomes both faster and lighter there. The quality-vs-latency plot directly visualizes whether quality gains require extra time.


## 12. Failure Modes

Adaptive underperformance should not be treated as noise. It reveals the conditions under which flexible retrieval or hybrid answering can backfire.

### Over-retrieval

Adaptive retrieves too much and introduces distracting context.

- Query: Compare the base Transformer model and the big Transformer model.
  Model/quant/data: Llama, 4bit, small
  Fixed: answer relevance 5.000, hallucination 0.333, groundedness 0.667
  Adaptive: answer relevance 5.000, hallucination 1.000, groundedness 0.000
  Interpretation: Adaptive retrieved more material but answer relevance or grounding fell, which is consistent with broader retrieval introducing distracting evidence.

- Query: If a multilingual model performs well on English and Spanish but poorly on Thai and Tamil, determine whether tokenization, script frequency, morphology, or benchmark design is the most likely bottleneck.
  Model/quant/data: Qwen, 8bit, big
  Fixed: answer relevance 4.000, hallucination 0.000, groundedness 0.875
  Adaptive: answer relevance 4.000, hallucination 0.500, groundedness 0.300
  Interpretation: Adaptive retrieved more material but answer relevance or grounding fell, which is consistent with broader retrieval introducing distracting evidence.

- Query: Compare classification and regression tasks.
  Model/quant/data: Llama, 8bit, medium
  Fixed: answer relevance 5.000, hallucination 0.000, groundedness 0.900
  Adaptive: answer relevance 5.000, hallucination 0.500, groundedness 0.375
  Interpretation: Adaptive retrieved more material but answer relevance or grounding fell, which is consistent with broader retrieval introducing distracting evidence.


### Context dilution

More context reduces focus and weakens grounding.

- Query: Compare generative and discriminative modeling approaches.
  Model/quant/data: Qwen, 4bit, medium
  Fixed: answer relevance 1.000, hallucination 0.000, groundedness 1.000
  Adaptive: answer relevance 4.000, hallucination 0.700, groundedness 0.200
  Interpretation: Adaptive broadened the evidence space and the answer became less well anchored, suggesting that extra context diluted focus instead of clarifying it.

- Query: Compare generative and discriminative modeling approaches.
  Model/quant/data: Qwen, 8bit, medium
  Fixed: answer relevance 1.000, hallucination 0.000, groundedness 1.000
  Adaptive: answer relevance 4.000, hallucination 0.700, groundedness 0.250
  Interpretation: Adaptive broadened the evidence space and the answer became less well anchored, suggesting that extra context diluted focus instead of clarifying it.

- Query: Compare the base Transformer model and the big Transformer model.
  Model/quant/data: Llama, 4bit, small
  Fixed: answer relevance 5.000, hallucination 0.333, groundedness 0.667
  Adaptive: answer relevance 5.000, hallucination 1.000, groundedness 0.000
  Interpretation: Adaptive broadened the evidence space and the answer became less well anchored, suggesting that extra context diluted focus instead of clarifying it.


### Hybrid overreach

Adaptive uses broader reasoning or general knowledge when retrieval evidence is insufficient, which can outpace the support actually present.

- Query: If two chatbots have equal benchmark scores but users trust one far more, analyze the roles of tone, uncertainty expression, memory consistency, and conversational repair.
  Model/quant/data: Llama, 8bit, big
  Fixed: answer relevance 4.000, hallucination 0.125, groundedness 0.438
  Adaptive: answer relevance 5.000, hallucination 1.000, groundedness 0.000
  Interpretation: Adaptive relied more on hybrid or general-knowledge behavior and lost factual discipline, indicating that broader reasoning outpaced the available evidence.

- Query: Compare generative and discriminative modeling approaches.
  Model/quant/data: Qwen, 8bit, medium
  Fixed: answer relevance 1.000, hallucination 0.000, groundedness 1.000
  Adaptive: answer relevance 4.000, hallucination 0.700, groundedness 0.250
  Interpretation: Adaptive relied more on hybrid or general-knowledge behavior and lost factual discipline, indicating that broader reasoning outpaced the available evidence.

- Query: Compare generative and discriminative modeling approaches.
  Model/quant/data: Qwen, 4bit, medium
  Fixed: answer relevance 1.000, hallucination 0.000, groundedness 1.000
  Adaptive: answer relevance 4.000, hallucination 0.700, groundedness 0.200
  Interpretation: Adaptive relied more on hybrid or general-knowledge behavior and lost factual discipline, indicating that broader reasoning outpaced the available evidence.


### Conservative fixed advantage

Fixed can win when cautious, retrieval-only behavior preserves factual discipline better than a broader adaptive answer policy.

- Query: Compare the base Transformer model and the big Transformer model.
  Model/quant/data: Llama, 4bit, small
  Fixed: answer relevance 5.000, hallucination 0.333, groundedness 0.667
  Adaptive: answer relevance 5.000, hallucination 1.000, groundedness 0.000
  Interpretation: Fixed answered more cautiously and preserved grounding or hallucination control better, showing that conservative retrieval-only behavior can still win on factual discipline.

- Query: How do supervised learning tasks differ from density estimation tasks in objectives?
  Model/quant/data: Llama, 4bit, medium
  Fixed: answer relevance 2.000, hallucination 0.500, groundedness 0.250
  Adaptive: answer relevance 2.000, hallucination 1.000, groundedness 0.000
  Interpretation: Fixed answered more cautiously and preserved grounding or hallucination control better, showing that conservative retrieval-only behavior can still win on factual discipline.

- Query: Why was the Transformer considered a breakthrough?
  Model/quant/data: Llama, 8bit, small
  Fixed: answer relevance 5.000, hallucination 0.000, groundedness 1.000
  Adaptive: answer relevance 5.000, hallucination 0.500, groundedness 0.500
  Interpretation: Fixed answered more cautiously and preserved grounding or hallucination control better, showing that conservative retrieval-only behavior can still win on factual discipline.


### Model-specific instability

Qwen shows the larger share of Adaptive-underperforming pairs (0.383 of its pairs).

### Quantization sensitivity

8bit has the larger share of Adaptive-underperforming pairs (0.342).

## 13. Discussion

Adaptive RAG should not be summarized as simply “better” or “worse.” The results show whether adaptive control over retrieval and answer strategy improves answer-level quality enough to justify its operational tradeoffs. When Adaptive improves answer relevance or lowers hallucination under the same paired conditions, the implication is that fixed retrieval budgets can be too rigid for some queries and corpus sizes. When Fixed remains competitive, the implication is that conservative retrieval precision and simpler orchestration still matter, especially when memory efficiency or strict grounding is the priority.

The defensible conclusion is that Adaptive RAG improves answer-level quality when query complexity, retrieval ambiguity, or context insufficiency demand more than a fixed retrieval policy. Fixed RAG remains competitive in direct retrieval precision and memory efficiency. Adaptive RAG is therefore best understood as a flexible evidence-use policy that can improve final answer quality, but it changes the operational tradeoff surface rather than eliminating it.

## 14. Reading the Figures and Tables

Each table and figure in this report should be read as a statement about RAG design rather than a scoreboard. A higher or lower metric only matters insofar as it changes the system’s usefulness, reliability, or deployability. The central distinction throughout the report is between retrieval quality and answer quality, and between user-facing latency and hardware-facing efficiency.

## 15. Where Adaptive RAG Outperforms Fixed RAG

### 1. Overall level

Adaptive has better overall mean performance on these metrics: hallucination_rate, groundedness_score, answer_relevance_1to5, quality_index, response_time_s.

Conceptually, these wins tell us whether Adaptive improves retrieval precision, answer generation, grounding, latency, or resource behavior. The most important overall win is on the metrics that improve end-user quality and factual reliability together, not on saturation alone.

### 2. Metric level

- Quality metrics where Adaptive is better on mean performance: hallucination_rate, groundedness_score, answer_relevance_1to5, quality_index
- Quality metrics where Fixed is better on mean performance: context_relevance_1to5
- Operational metrics where Adaptive is better on mean performance: response_time_s
- Operational metrics where Fixed is better on mean performance: gpu_throughput_toks_per_s, gpu_mem_peak_mb
- GPU utilization should be read as a saturation shift rather than a direct win/loss metric.

### 3. Factor level

Dataset-size interpretation: the strongest `quality_index` Adaptive gain occurs at medium (0.193). If the larger datasets show stronger gains, that supports the idea that adaptive control becomes more useful when the retrieval space is more ambiguous. If the smallest dataset remains competitive for Fixed, that suggests fixed retrieval is sufficient when the evidence space is simple.

Model-family interpretation: the strongest model-family Adaptive `quality_index` gain occurs at Llama 3.3 70B (0.216). This indicates which model can actually exploit flexible retrieval and broader evidence use rather than merely tolerate it.

Quantization interpretation: the strongest quantization-level Adaptive `quality_index` gain occurs at 4bit (0.169). That pattern tells us whether Adaptive compensates for quantization-induced limitations or whether lower precision makes flexible retrieval harder to capitalize on.

### 4. Query level

#### A. Better synthesis

- Query: Analyze whether contextual embeddings truly solved polysemy better than static embeddings, or whether they mainly shifted ambiguity into downstream fine-tuning.
  Model/quant/data: Llama, 4bit, big
  Fixed: answer relevance 1.000, hallucination 1.000, groundedness 0.000
  Adaptive: answer relevance 5.000, hallucination 0.500, groundedness 0.500
  Interpretation: Adaptive improved answer relevance without needing an equally large context-relevance gain, which suggests better synthesis of the evidence rather than only better retrieval precision.

- Query: Compare generative and discriminative modeling approaches.
  Model/quant/data: Llama, 4bit, medium
  Fixed: answer relevance 1.000, hallucination 1.000, groundedness 0.000
  Adaptive: answer relevance 5.000, hallucination 0.800, groundedness 0.200
  Interpretation: Adaptive improved answer relevance without needing an equally large context-relevance gain, which suggests better synthesis of the evidence rather than only better retrieval precision.

- Query: Why can spam filtering and cancer diagnosis both be treated as classification problems?
  Model/quant/data: Llama, 4bit, medium
  Fixed: answer relevance 1.000, hallucination 0.500, groundedness 0.500
  Adaptive: answer relevance 5.000, hallucination 0.000, groundedness 0.750
  Interpretation: Adaptive improved answer relevance without needing an equally large context-relevance gain, which suggests better synthesis of the evidence rather than only better retrieval precision.


#### B. Better handling of insufficient context

- Query: Analyze why positional encoding is necessary in a non-recurrent architecture.
  Model/quant/data: Llama, 4bit, small
  Fixed: answer relevance 1.000, hallucination 0.500, groundedness 0.250
  Adaptive: answer relevance 5.000, hallucination 0.111, groundedness 0.889
  Interpretation: Adaptive was more useful when Fixed stayed conservative under partial context, indicating better handling of insufficient evidence or more flexible answer routing.

- Query: Explain the impact of parallelization on Transformer training efficiency.
  Model/quant/data: Llama, 4bit, small
  Fixed: answer relevance 1.000, hallucination 1.000, groundedness 0.000
  Adaptive: answer relevance 5.000, hallucination 0.000, groundedness 0.938
  Interpretation: Adaptive was more useful when Fixed stayed conservative under partial context, indicating better handling of insufficient evidence or more flexible answer routing.

- Query: Compare whether future NLP progress is more likely to come from bigger models, better retrieval systems, multimodal grounding, or improved symbolic-neural hybrids.
  Model/quant/data: Llama, 4bit, big
  Fixed: answer relevance 1.000, hallucination 1.000, groundedness 0.000
  Adaptive: answer relevance 5.000, hallucination 0.500, groundedness 0.250
  Interpretation: Adaptive was more useful when Fixed stayed conservative under partial context, indicating better handling of insufficient evidence or more flexible answer routing.


#### C. Better query-type routing

- Query: Analyze whether contextual embeddings truly solved polysemy better than static embeddings, or whether they mainly shifted ambiguity into downstream fine-tuning.
  Model/quant/data: Llama, 4bit, big
  Fixed: answer relevance 1.000, hallucination 1.000, groundedness 0.000
  Adaptive: answer relevance 5.000, hallucination 0.500, groundedness 0.500
  Interpretation: Adaptive switched to a different answer mode and that change coincided with a stronger answer, which is consistent with query-type routing helping beyond fixed retrieval.

- Query: Compare whether future NLP progress is more likely to come from bigger models, better retrieval systems, multimodal grounding, or improved symbolic-neural hybrids.
  Model/quant/data: Llama, 4bit, big
  Fixed: answer relevance 1.000, hallucination 1.000, groundedness 0.000
  Adaptive: answer relevance 5.000, hallucination 0.500, groundedness 0.250
  Interpretation: Adaptive switched to a different answer mode and that change coincided with a stronger answer, which is consistent with query-type routing helping beyond fixed retrieval.

- Query: Explain the impact of parallelization on Transformer training efficiency.
  Model/quant/data: Llama, 4bit, small
  Fixed: answer relevance 1.000, hallucination 1.000, groundedness 0.000
  Adaptive: answer relevance 5.000, hallucination 0.000, groundedness 0.938
  Interpretation: Adaptive switched to a different answer mode and that change coincided with a stronger answer, which is consistent with query-type routing helping beyond fixed retrieval.


#### D. Better retrieval breadth

- Query: Analyze whether contextual embeddings truly solved polysemy better than static embeddings, or whether they mainly shifted ambiguity into downstream fine-tuning.
  Model/quant/data: Llama, 4bit, big
  Fixed: answer relevance 1.000, hallucination 1.000, groundedness 0.000
  Adaptive: answer relevance 5.000, hallucination 0.500, groundedness 0.500
  Interpretation: Adaptive retrieved more or broader evidence and translated that extra breadth into a more complete answer.

- Query: Compare logistic regression, feedforward neural networks, and transformers for sentiment classification when training data is small, noisy, and domain-shifted.
  Model/quant/data: Llama, 4bit, big
  Fixed: answer relevance 1.000, hallucination 0.500, groundedness 0.500
  Adaptive: answer relevance 5.000, hallucination 0.667, groundedness 0.167
  Interpretation: Adaptive retrieved more or broader evidence and translated that extra breadth into a more complete answer.

- Query: Explain the impact of parallelization on Transformer training efficiency.
  Model/quant/data: Llama, 4bit, small
  Fixed: answer relevance 1.000, hallucination 1.000, groundedness 0.000
  Adaptive: answer relevance 5.000, hallucination 0.000, groundedness 0.938
  Interpretation: Adaptive retrieved more or broader evidence and translated that extra breadth into a more complete answer.


#### E. Better factual caution

- Query: What makes one algorithm better than another?
  Model/quant/data: Llama, 4bit, medium
  Fixed: answer relevance 2.000, hallucination 1.000, groundedness 0.000
  Adaptive: answer relevance 5.000, hallucination 0.000, groundedness 1.000
  Interpretation: Adaptive reduced hallucination without sacrificing usefulness, which is the most attractive quality gain because it improves reliability and user value together.

- Query: Explain the impact of parallelization on Transformer training efficiency.
  Model/quant/data: Llama, 8bit, small
  Fixed: answer relevance 3.000, hallucination 1.000, groundedness 0.000
  Adaptive: answer relevance 5.000, hallucination 0.000, groundedness 0.714
  Interpretation: Adaptive reduced hallucination without sacrificing usefulness, which is the most attractive quality gain because it improves reliability and user value together.

- Query: Why was the Transformer considered a breakthrough?
  Model/quant/data: Llama, 4bit, small
  Fixed: answer relevance 1.000, hallucination 1.000, groundedness 0.000
  Adaptive: answer relevance 5.000, hallucination 0.000, groundedness 0.833
  Interpretation: Adaptive reduced hallucination without sacrificing usefulness, which is the most attractive quality gain because it improves reliability and user value together.


Summary table:

| Level | Where Adaptive Wins | Conceptual Reason |
| --- | --- | --- |
| Overall | hallucination_rate, groundedness_score, answer_relevance_1to5, quality_index, response_time_s | Better evidence use and answer policy where Adaptive’s mean beats Fixed |
| Dataset size | big (1.800) | Larger retrieval spaces raise ambiguity, so adaptive control has more room to help when its deltas improve |
| Model family | Llama 3.3 70B (0.216) | Some models exploit flexible retrieval and routing more effectively than others |
| Quantization | 4bit (0.169) | Precision changes how well the model can digest broader or noisier context |
| Query type | See representative paired wins below | Adaptive routing helps analytical, comparison, or partially grounded questions when fixed retrieval is too rigid |

Adaptive RAG performs best when the task requires more than retrieving a fixed number of passages. Its advantage appears when the system must decide how much evidence is needed, how to combine it, and how cautious or expansive the final answer should be.

## 16. Overall Conceptual Takeaways

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

One of the most important results in this study is that Adaptive is not only a quality intervention. It improves the composite `quality_index` from 0.634 to 0.765 while also reducing end-to-end response time by 7.113 seconds on average. Conceptually, this means adaptive control can reduce wasted answer behavior even when it does not maximize raw decoding throughput. In other words, a dynamic evidence-use policy can be more helpful to the user and faster in wall-clock terms, even if it is not the lightest decoder or the narrowest retriever.

### 9. Adaptive RAG fails when flexibility turns into noise

Adaptive does not fail because adaptivity is inherently bad; it fails when broader retrieval or hybrid answering introduces more uncertainty than value. In this experiment, those failures appear as over-retrieval, context dilution, hybrid overreach, and model-specific instability. The paired results show a larger share of Adaptive-underperforming cases for Qwen 32B than for Llama 3.3 70B, and some configurations such as Qwen 4-bit small are effectively tied or slightly worse on `quality_index`. The conceptual lesson is that adaptive control helps only when the model can filter broader evidence without losing grounding discipline.

### 10. Best overall conceptual conclusion

Fixed RAG should be understood as a conservative, predictable retrieval baseline. Adaptive RAG should be understood as a flexible evidence-use strategy that improves final answer quality when query complexity, retrieval ambiguity, or context insufficiency require more than a fixed retrieval policy. The comparison should therefore be interpreted not as a winner-takes-all benchmark, but as a study of when adaptive control improves the quality-efficiency tradeoff in retrieval-augmented generation.

## 17. Model-Family Analysis: Llama 3.3 70B versus Qwen 32B

### 1. Overall model comparison

The overall model-level comparison asks which model is stronger across all configurations, not just within one system.

| metric | llama_mean | qwen_mean | llama_minus_qwen | better_model |
| --- | --- | --- | --- | --- |
| hallucination_rate | 0.32519789682539685 | 0.2000019708994709 | 0.12519592592592596 | Qwen |
| groundedness_score | 0.5848962433862434 | 0.6937684788359789 | -0.10887223544973557 | Qwen |
| answer_relevance_1to5 | 4.029166666666667 | 3.3875 | 0.6416666666666666 | Llama |
| context_relevance_1to5 | 4.2 | 3.3125 | 0.8875000000000002 | Llama |
| quality_index | 0.6847006898148149 | 0.7146318988095237 | -0.029931208994708802 | Qwen |
| response_time_s | 29.709334648195696 | 34.92546289484099 | -5.216128246645294 | Llama |
| gpu_util_percent | 83.44153366078132 | 75.24647018480306 | 8.195063475978259 | higher GPU saturation |
| gpu_throughput_toks_per_s | 8.85819785951246 | 11.30254921841481 | -2.4443513589023507 | Qwen |
| gpu_mem_peak_mb | 61640.26666666667 | 25183.5 | 36456.76666666667 | Qwen |

On `quality_index`, Llama = 0.685, Qwen = 0.715, so Llama - Qwen = -0.030. On latency, Llama - Qwen = -5.216; on throughput, Llama - Qwen = -2.444; on memory, Llama - Qwen = 36456.8. This tells us whether the stronger model is stronger because of answer quality, operational efficiency, or both.

### 2. Model comparison within each RAG system

The better model can depend on the retrieval policy, so Llama versus Qwen must be compared separately within Fixed and Adaptive.

| system | metric | llama_mean | qwen_mean | llama_minus_qwen | better_model |
| --- | --- | --- | --- | --- | --- |
| fixed | hallucination_rate | 0.42341166666666663 | 0.20924666666666666 | 0.21416499999999997 | Qwen |
| fixed | groundedness_score | 0.5011450000000001 | 0.7334066666666668 | -0.2322616666666667 | Qwen |
| fixed | answer_relevance_1to5 | 3.316666666666667 | 2.8583333333333334 | 0.4583333333333335 | Llama |
| fixed | context_relevance_1to5 | 4.258333333333334 | 3.4 | 0.8583333333333338 | Llama |
| fixed | quality_index | 0.5767717500000001 | 0.6911343333333332 | -0.11436258333333305 | Qwen |
| fixed | response_time_s | 31.711076666666667 | 40.036730000000006 | -8.325653333333339 | Llama |
| fixed | gpu_util_percent | 96.5830025 | 85.71367125 | 10.869331250000002 | higher GPU saturation |
| fixed | gpu_throughput_toks_per_s | 8.86754 | 11.783274166666668 | -2.9157341666666685 | Qwen |
| fixed | gpu_mem_peak_mb | 64946.875 | 16114.091666666667 | 48832.78333333333 | Qwen |
| adaptive | hallucination_rate | 0.226984126984127 | 0.19075727513227514 | 0.03622685185185187 | Qwen |
| adaptive | groundedness_score | 0.6686474867724868 | 0.6541302910052911 | 0.014517195767195679 | Llama |
| adaptive | answer_relevance_1to5 | 4.741666666666666 | 3.9166666666666665 | 0.8249999999999997 | Llama |
| adaptive | context_relevance_1to5 | 4.141666666666667 | 3.225 | 0.9166666666666665 | Llama |
| adaptive | quality_index | 0.7926296296296297 | 0.7381294642857142 | 0.05450016534391544 | Llama |
| adaptive | response_time_s | 27.707592629724726 | 29.814195789681982 | -2.106603159957256 | Llama |
| adaptive | gpu_util_percent | 70.30006482156263 | 64.77926911960613 | 5.5207957019565015 | higher GPU saturation |
| adaptive | gpu_throughput_toks_per_s | 8.848855719024918 | 10.821824270162955 | -1.9729685511380364 | Qwen |
| adaptive | gpu_mem_peak_mb | 58333.65833333333 | 34252.90833333333 | 24080.75 | Qwen |

Under Fixed RAG, the `quality_index` difference is -0.114. Under Adaptive RAG, the `quality_index` difference is 0.055. If the advantage widens under Adaptive, that means the stronger model is better at exploiting flexible retrieval and answer-mode selection rather than only stronger in a static pipeline.

### 3. Adaptive gain by model

Model quality and adaptive gain are different. A model can have higher absolute performance but smaller adaptive improvement if it was already strong under Fixed retrieval. Another model can have lower baseline performance but larger adaptive gain if the adaptive policy compensates for its weaknesses.

| model_family | metric | fixed_mean | adaptive_mean | adaptive_minus_fixed |
| --- | --- | --- | --- | --- |
| Llama 3.3 70B | hallucination_rate | 0.42341166666666663 | 0.226984126984127 | -0.1964275396825397 |
| Llama 3.3 70B | groundedness_score | 0.5011450000000001 | 0.6686474867724868 | 0.16750248677248675 |
| Llama 3.3 70B | answer_relevance_1to5 | 3.316666666666667 | 4.741666666666666 | 1.425 |
| Llama 3.3 70B | context_relevance_1to5 | 4.258333333333334 | 4.141666666666667 | -0.11666666666666667 |
| Llama 3.3 70B | quality_index | 0.5767717500000001 | 0.7926296296296297 | 0.21585787962962963 |
| Llama 3.3 70B | response_time_s | 31.711076666666667 | 27.707592629724726 | -4.003484036941941 |
| Llama 3.3 70B | gpu_util_percent | 96.5830025 | 70.30006482156263 | -26.282937678437367 |
| Llama 3.3 70B | gpu_throughput_toks_per_s | 8.86754 | 8.848855719024918 | -0.01868428097508271 |
| Llama 3.3 70B | gpu_mem_peak_mb | 64946.875 | 58333.65833333333 | -6613.216666666666 |
| Qwen 32B | hallucination_rate | 0.20924666666666666 | 0.19075727513227514 | -0.01848939153439154 |
| Qwen 32B | groundedness_score | 0.7334066666666668 | 0.6541302910052911 | -0.07927637566137566 |
| Qwen 32B | answer_relevance_1to5 | 2.8583333333333334 | 3.9166666666666665 | 1.0583333333333333 |
| Qwen 32B | context_relevance_1to5 | 3.4 | 3.225 | -0.175 |
| Qwen 32B | quality_index | 0.6911343333333332 | 0.7381294642857142 | 0.046995130952380955 |
| Qwen 32B | response_time_s | 40.036730000000006 | 29.814195789681982 | -10.22253421031802 |
| Qwen 32B | gpu_util_percent | 85.71367125 | 64.77926911960613 | -20.934402130393877 |
| Qwen 32B | gpu_throughput_toks_per_s | 11.783274166666668 | 10.821824270162955 | -0.9614498965037117 |
| Qwen 32B | gpu_mem_peak_mb | 16114.091666666667 | 34252.90833333333 | 18138.816666666666 |

The larger `quality_index` adaptive gain belongs to Llama 3.3 70B. The smaller gain belongs to Qwen 32B. That distinction matters because it separates “best model overall” from “model that benefits most from adaptive policy.”

### 4. Model strengths

If Llama leads on answer relevance and quality index, the conceptual implication is that it is better at evidence synthesis or broader-context use. If Qwen leads on latency, throughput, or memory, the implication is that it is operationally more efficient even when it is not the strongest answer model. The model-comparison tables above show which of those two stories is actually supported by the data.

### 5. Model weaknesses

Llama’s clearest weakness in this experiment is operational heaviness or context overreach when Adaptive loses. For example, on 'Compare the base Transformer model and the big Transformer model.' under 4bit small, Adaptive moved quality by -0.433 while response time changed by 1.205 and memory by 2068.0.

Qwen’s clearest weakness is answer-level instability when context broadens. For example, on 'If a multilingual model performs well on English and Spanish but poorly on Thai and Tamil, determine whether tokenization, script frequency, morphology, or benchmark design is the most likely bottleneck.' under 8bit big, Adaptive changed answer relevance by 0.000, groundedness by -0.575, and hallucination by 0.500.

These weaknesses should not be assumed in advance; they matter only to the extent that the computed metrics actually show them.

### 6. Model x dataset-size interaction

Retrieval is easier on small datasets, so model differences there often reflect answer generation ability more than retrieval robustness. As corpus size grows, the better model is usually the one that can filter and synthesize broader evidence without losing grounding.

- Small dataset best mean `quality_index`: Qwen (0.847)
- Medium dataset best mean `quality_index`: Qwen (0.715)
- Big dataset best mean `quality_index`: Qwen (0.583)

These rankings show whether the stronger model stays stronger as retrieval ambiguity increases.

### 7. Model x quantization interaction

Quantization can reduce precision in context use and reasoning. A model that stays relevant and grounded under lower precision is more robust.

- 4-bit best mean `quality_index`: Qwen (0.707)
- 8-bit best mean `quality_index`: Llama (0.731)

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

The model comparison should end in a conditional conclusion rather than a slogan. In the aggregate tables, Qwen 32B has the higher overall quality index (0.715 versus 0.685), but Llama 3.3 70B shows the larger Adaptive-minus-Fixed quality gain (0.216). If Llama leads on quality while Qwen leads on efficiency, then Llama is the quality-first choice and Qwen is the resource-constrained choice. If one model also gains more from Adaptive, then adaptive RAG depends not only on the retrieval policy itself but also on the model’s ability to exploit that policy.

## 18. Executive Summary

1. The stronger RAG system overall is the one with the better `quality_index` balance between groundedness, answer usefulness, and hallucination control. In these data, Fixed = 0.634 and Adaptive = 0.765, so the system-level conclusion should start there rather than with any single metric.
2. Adaptive RAG clearly outperforms Fixed where it improves answer relevance, lowers hallucination, or lifts quality index under the same paired conditions, especially in the factors and configurations listed above.
3. Fixed RAG remains competitive where direct retrieval precision and lighter operational footprint matter more than flexible synthesis.
4. The stronger model family overall is the one with the higher aggregate `quality_index`, but the operationally better family may differ if latency, throughput, or memory become binding constraints.
5. The weaker model family is the one that either underuses adaptive context or pays quality penalties when context broadens; the model comparison tables and interaction plots identify which one that is in the current data.
6. Dataset size changes the comparison because larger corpora raise retrieval ambiguity. If Adaptive gains grow with dataset size, that supports dynamic retrieval as a response to harder evidence selection.
7. Quantization changes the comparison because lower precision can blunt context handling or make longer contexts noisier. The factor summaries show whether Adaptive compensates for that or amplifies it.
8. The most important quality tradeoff is between answer relevance and hallucination control: a system that becomes more helpful but less grounded is not automatically better.
9. The most important operational tradeoff is between quality gain and deployment burden: response time, throughput, and memory should be interpreted together rather than separately.
10. The main system-design implication is that RAG should be treated as an evidence-use policy, not just a retriever. Adaptive control matters most when query complexity and corpus ambiguity make a fixed retrieval budget too rigid.
