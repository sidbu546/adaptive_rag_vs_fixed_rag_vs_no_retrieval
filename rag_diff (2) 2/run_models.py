import argparse
import json
import os
from dataclasses import asdict

import pandas as pd
import torch

from llm_manager import LLMManager
from rag_eval import evaluate_rag
from rag_retriever_chroma import build_chroma_rag


MODEL_PRESETS = {
    "llama33_70b": {
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "dtype": "bfloat16",
        "max_new_tokens": 384,
        "temperature": 0.2,
    },
    "qwen25_72b": {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "dtype": "bfloat16",
        "max_new_tokens": 384,
        "temperature": 0.2,
    },
    "qwen25_32b": {
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "dtype": "bfloat16",
        "max_new_tokens": 384,
        "temperature": 0.2,
    },
}


def _load_queries(query_file: str):
    path = query_file.lower()
    if path.endswith(".json"):
        with open(query_file, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return [{"query": x if isinstance(x, str) else x["query"]} for x in obj]
        raise ValueError("JSON query file must contain a list.")

    with open(query_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return [{"query": q} for q in lines]


def _apply_preset(args):
    if not args.model_preset:
        return args
    preset = MODEL_PRESETS[args.model_preset]
    args.model = preset["model"]
    if args.dtype == "auto":
        args.dtype = preset["dtype"]
    if args.max_new_tokens == parser_defaults["max_new_tokens"]:
        args.max_new_tokens = preset["max_new_tokens"]
    if args.temperature == parser_defaults["temperature"]:
        args.temperature = preset["temperature"]
    return args


def _print_gpu_info() -> None:
    """Print visible GPU info. No requirements — runs on any GPU."""
    if not torch.cuda.is_available():
        print("[run_models] No CUDA GPUs visible — running on CPU.")
        return
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gib = props.total_memory / (1024 ** 3)
        print(f"[run_models] cuda:{i}  {props.name}  {gib:.1f} GiB")


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct", help="HF model id")
    parser.add_argument("--model_preset", type=str, default="", choices=["", *MODEL_PRESETS.keys()])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    parser.set_defaults(trust_remote_code=True)

    parser.add_argument("--quantization_mode", type=str, default="4bit", choices=["none", "8bit", "4bit"])
    parser.add_argument("--load_in_4bit", action="store_true", help="Backward-compatible alias for --quantization_mode 4bit")
    parser.add_argument("--no_load_in_4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=None)
    parser.add_argument("--load_in_8bit", action="store_true", help="Backward-compatible alias for --quantization_mode 8bit")
    parser.add_argument("--no_load_in_8bit", dest="load_in_8bit", action="store_false")
    parser.set_defaults(load_in_8bit=None)

    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", choices=["nf4", "fp4"])
    parser.add_argument("--no_bnb_4bit_double_quant", action="store_true")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--attn_implementation", type=str, default="auto", choices=["auto", "flash_attention_2", "sdpa", "eager", "none"])
    parser.add_argument("--model_max_length", type=int, default=None)

    parser.add_argument("--allow_cpu_offload", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.92)

    parser.add_argument("--data_dir", type=str, default="/projectnb/cs585/students/siddhank/data_research/data")
    parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--chunk_size", type=int, default=700)
    parser.add_argument("--chunk_overlap", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--pool_size", type=int, default=60)
    parser.add_argument("--max_per_source", type=int, default=3)
    parser.add_argument("--preferred_sources", type=str, default="")
    parser.add_argument("--neighbor_window", type=int, default=1)
    parser.add_argument("--lexical_weight", type=float, default=0.20)

    parser.add_argument("--max_new_tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--no_do_sample", dest="do_sample", action="store_false")
    parser.set_defaults(do_sample=False)

    parser.add_argument("--max_claims", type=int, default=10)
    parser.add_argument("--gpu_index", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default="rag_eval.csv",
                        help="Path for the summary CSV (one row per query).")
    parser.add_argument("--out_json", type=str, default="",
                        help="Optional path for the full detailed results JSON.")
    parser.add_argument("--query_file", type=str, default="")
    parser.add_argument(
        "--classifier_mode", type=str, default="heuristic",
        choices=["heuristic", "llm", "probe"],
        help=(
            "heuristic: regex/keyword rules (fast, no extra inference); "
            "llm: zero-shot via the generator (recommended, no training data needed); "
            "probe: hidden-state linear probe (call ProbeClassifier.fit() with labeled data first)"
        ),
    )

    parser.add_argument("--gpu_cost_per_hour", type=float, default=3.0)
    parser.add_argument("--cpu_cost_per_hour", type=float, default=2.0)
    parser.add_argument("--fixed_cost_usd", type=float, default=2.0)
    return parser


def _resolve_quantization_aliases(args):
    if args.load_in_4bit is True and args.load_in_8bit is True:
        raise ValueError("Choose only one of --load_in_4bit or --load_in_8bit.")
    if args.load_in_4bit is True:
        args.quantization_mode = "4bit"
    elif args.load_in_8bit is True:
        args.quantization_mode = "8bit"
    return args


def main():
    parser = build_parser()
    args = parser.parse_args()
    args = _resolve_quantization_aliases(args)
    args = _apply_preset(args)

    print(f"\n[run_models] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
    _print_gpu_info()

    if args.model_preset:
        print(f"[run_models] Applied preset: {args.model_preset}")

    preferred_sources = [s.strip() for s in (args.preferred_sources or "").split(",") if s.strip()]

    rag_retriever = build_chroma_rag(
        data_dir=args.data_dir,
        exclude_paths=[args.query_file] if args.query_file else None,
        embed_model=args.embed_model,
        collection_name="txt_documents",
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        pool_size=args.pool_size,
        max_per_source=args.max_per_source,
        preferred_sources=preferred_sources,
        neighbor_window=args.neighbor_window,
        lexical_weight=args.lexical_weight,
    )

    llm_manager = LLMManager(
        model_name=args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        quantization_mode=args.quantization_mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        trust_remote_code=args.trust_remote_code,
        no_cpu_offload=not args.allow_cpu_offload,
        gpu_memory_utilization=args.gpu_memory_utilization,
        bnb_4bit_quant_type=args.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=not args.no_bnb_4bit_double_quant,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
        attn_implementation=args.attn_implementation,
        model_max_length=args.model_max_length,
    )

    if args.query_file:
        test_queries = _load_queries(args.query_file)
    else:
        # Default: 20 representative queries covering all 5 RARC query types.
        # Use --query_file queries_small.json / queries_medium.json / queries_large.json
        # (generated by eval_queries.py) for corpus-specific evaluation.
        test_queries = [
            # ── factual (4) ─────────────────────────────────────────────────
            {"query": "What year did the Frankfurt Parliament convene?"},
            {"query": "Who was the main architect of German unification in the 19th century?"},
            # ── multi-hop (4) ────────────────────────────────────────────────
            {"query": "How did the Congress of Vienna create conditions that fuelled the 1848 revolutions across Europe?"},
            {"query": "What role did industrialisation play in shaping the social base of 19th-century nationalist movements?"},
        ]

    rows, detailed = evaluate_rag(
        queries=test_queries,
        retriever=rag_retriever,
        rag_llm=llm_manager,
        judge_llm=llm_manager,
        top_k=args.top_k,
        min_score=args.min_score,
        max_claims=args.max_claims,
        gpu_index=args.gpu_index,
        gpu_cost_per_hour=args.gpu_cost_per_hour,
        cpu_cost_per_hour=args.cpu_cost_per_hour,
        fixed_cost_usd=args.fixed_cost_usd,
        classifier_mode=args.classifier_mode,
    )

    df = pd.DataFrame([asdict(r) for r in rows])

    cols = [
        "query",
        "query_type",
        "answer",
        "hallucination_rate",
        "groundedness_score",
        "answer_relevance_1to5",
        "context_relevance_1to5",
        "confidence",
        "response_time_s",
        "llm_latency_s",
        "gpu_throughput_toks_per_s",
        "eff_gpu_throughput",
        "gpu_util_percent",
        "gpu_mem_percent",
        "gpu_util_max_percent",
        "gpu_mem_avg_percent",
        "gpu_mem_peak_mb",
        "gpu_mem_torch_peak_mb",
        "gpu_monitor_samples",
        "total_deployment_cost_usd",
        "top_source",
        "context_status",
        "answer_mode_used",
        "used_general_knowledge",
        "retrieved_docs_count",
        "top_retrieval_score",
        "avg_retrieval_score",
        "query_coverage",
    ]

    df_view = df[cols]

    print("\n=== RESULTS ===")
    print(df_view)

    df_view.to_csv(args.out_csv, index=False)
    print(f"\nSaved CSV : {args.out_csv}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(detailed, f, indent=2, default=str, ensure_ascii=False)
        print(f"Saved JSON: {args.out_json}")


parser_defaults = {
    "max_new_tokens": 320,
    "temperature": 0.2,
}


if __name__ == "__main__":
    main()
