# ============================================================
# NO-RETRIEVAL LLM BASELINE — GPU-ONLY FIXED VERSION
# ============================================================

import os

# -------------------------
# ENV SETUP
# -------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import re
import json as _json_mod
import time
import argparse
import threading
import warnings
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# PDF reader
try:
    import fitz  # PyMuPDF
    USE_FITZ = True
except ImportError:
    USE_FITZ = False
    from pypdf import PdfReader

# GPU monitor
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


# ============================================================
# UTILITIES
# ============================================================

def print_header(msg: str):
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _resolve_nvml_index(torch_gpu_index: int) -> int:
    """
    Map torch-visible GPU index to physical NVML GPU index.

    Example:
    CUDA_VISIBLE_DEVICES=3 means torch sees GPU 3 as cuda:0,
    but NVML still sees it as physical GPU 3.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return int(torch_gpu_index)

    parts = [p.strip() for p in visible.split(",") if p.strip()]
    if not parts or torch_gpu_index >= len(parts):
        return int(torch_gpu_index)

    mapped = parts[torch_gpu_index]
    return int(mapped) if mapped.isdigit() else int(torch_gpu_index)


# ============================================================
# PROMPT LOADING
# ============================================================

def load_prompts_from_txt(txt_path: str) -> List[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


# ============================================================
# PDF LOADING
# ============================================================

def load_pdf_text(pdf_path: str) -> List[Dict[str, Any]]:
    pages = []

    if USE_FITZ:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            text = page.get_text("text")
            text = normalize_text(text)
            if text:
                pages.append({"page_num": i + 1, "text": text})
        doc.close()
    else:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = normalize_text(text)
            if text:
                pages.append({"page_num": i + 1, "text": text})

    return pages


# ============================================================
# GPU MONITOR
# ============================================================

class GPUMonitor:
    def __init__(self, interval_sec: float = 0.2, gpu_index: int = 0):
        self.interval_sec = interval_sec
        self.gpu_index = gpu_index
        self._thread = None
        self._stop_event = threading.Event()
        self.samples = []
        self.enabled = torch.cuda.is_available() and PYNVML_AVAILABLE

        if self.enabled:
            pynvml.nvmlInit()
            self._physical_index = _resolve_nvml_index(gpu_index)
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(self._physical_index)
            print(
                f"[GPUMonitor] torch index={gpu_index} -> "
                f"NVML physical index={self._physical_index} "
                f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')})"
            )

    def _sample_once(self):
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

            used_mb = mem.used / (1024 ** 2)
            total_mb = mem.total / (1024 ** 2)
            mem_percent = (mem.used / mem.total) * 100.0 if mem.total else 0.0

            self.samples.append({
                "gpu_util_percent": float(util.gpu),
                "gpu_mem_percent": float(mem_percent),
                "gpu_mem_used_mb": float(used_mb),
                "gpu_mem_total_mb": float(total_mb),
                "timestamp": time.time(),
            })
        except Exception:
            pass

    def _run(self):
        while not self._stop_event.is_set():
            self._sample_once()
            time.sleep(self.interval_sec)

    def start(self):
        self.samples = []
        if self.enabled:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        if self.enabled and self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=2)

            if not self.samples:
                self._sample_once()

    def summary(self):
        if not self.samples:
            return {
                "gpu_util_percent": 0.0,
                "gpu_mem_percent": 0.0,
                "gpu_util_max_percent": 0.0,
                "gpu_mem_avg_percent": 0.0,
                "gpu_mem_peak_mb": 0.0,
                "gpu_monitor_samples": 0,
            }

        gpu_utils = [s["gpu_util_percent"] for s in self.samples]
        gpu_mems_pct = [s["gpu_mem_percent"] for s in self.samples]
        gpu_mems_mb = [s["gpu_mem_used_mb"] for s in self.samples]

        return {
            "gpu_util_percent": float(np.mean(gpu_utils)),
            "gpu_mem_percent": float(np.mean(gpu_mems_pct)),
            "gpu_util_max_percent": float(np.max(gpu_utils)),
            "gpu_mem_avg_percent": float(np.mean(gpu_mems_pct)),
            "gpu_mem_peak_mb": float(np.max(gpu_mems_mb)),
            "gpu_monitor_samples": int(len(self.samples)),
        }


# ============================================================
# LLM LOADING — GPU ONLY
# ============================================================

def load_llm(
    model_path: str,
    gpu_max_memory: str,
    cpu_max_memory: str,
    quant_mode: str = "4bit",
):
    print_header("Checking CUDA")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This corrected version is GPU-only. "
            "Request a GPU node or check CUDA_VISIBLE_DEVICES."
        )

    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))

    print_header("Loading tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cleanup_memory()

    quant_config = None

    if quant_mode == "4bit":
        print_header("Preparing 4-bit config")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    elif quant_mode == "8bit":
        print_header("Preparing 8-bit config")
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    elif quant_mode == "none":
        print_header("Preparing FP16 config")
        quant_config = None

    else:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}")

    print_header("Loading LLM on GPU only")

    model_kwargs = {
        # IMPORTANT FIX:
        # This forces the entire model onto cuda:0.
        # Do NOT use device_map='auto' if you do not want CPU/disk dispatch.
        "device_map": {"": 0},

        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
        "local_files_only": True,
    }

    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )

    model.eval()
    cleanup_memory()

    print("Model loaded successfully on GPU.")

    if hasattr(model, "hf_device_map"):
        print("Device map:", model.hf_device_map)

    first_param_device = next(model.parameters()).device
    print("First parameter device:", first_param_device)

    if first_param_device.type != "cuda":
        raise RuntimeError(
            f"Model is not on GPU. First parameter is on {first_param_device}."
        )

    return tokenizer, model


# ============================================================
# PROMPT BUILDING
# ============================================================

def build_no_retrieval_messages(query: str) -> List[Dict[str, str]]:
    system_prompt = (
        "You are a knowledgeable assistant.\n"
        "Answer the user's question using only your own knowledge.\n"
        "Be concise but complete. If you are uncertain, say so rather than guessing.\n"
    )

    user_prompt = f"Question:\n{query}\n"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    model,
    tokenizer,
    query: str,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
) -> Dict[str, Any]:

    messages = build_no_retrieval_messages(query)

    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text_input,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )

    # IMPORTANT FIX:
    # Put inputs on the same actual device as the model.
    target_device = next(model.parameters()).device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    prompt_tokens = int(inputs["input_ids"].shape[1])

    cleanup_memory()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    gpu_monitor = GPUMonitor(interval_sec=0.2)
    gpu_monitor.start()

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    if do_sample:
        gen_kwargs["temperature"] = temperature

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    end_time = time.time()

    gpu_monitor.stop()

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    latency = end_time - start_time
    new_tokens = int(len(generated_ids))
    toks_per_sec = (new_tokens / latency) if latency > 0 else 0.0

    torch_peak_mb = 0.0
    if torch.cuda.is_available():
        torch_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    gpu_stats = gpu_monitor.summary()

    return {
        "answer": answer,
        "response_time_s": latency,
        "llm_latency_s": latency,
        "gpu_throughput_toks_per_s": float(toks_per_sec),
        "eff_gpu_throughput": float(toks_per_sec),
        "gpu_util_percent": gpu_stats["gpu_util_percent"],
        "gpu_mem_percent": gpu_stats["gpu_mem_percent"],
        "gpu_util_max_percent": gpu_stats["gpu_util_max_percent"],
        "gpu_mem_avg_percent": gpu_stats["gpu_mem_avg_percent"],
        "gpu_mem_peak_mb": gpu_stats["gpu_mem_peak_mb"],
        "gpu_mem_torch_peak_mb": torch_peak_mb,
        "gpu_monitor_samples": gpu_stats["gpu_monitor_samples"],
        "prompt_tokens": prompt_tokens,
        "generated_tokens": new_tokens,
    }


# ============================================================
# LLM-JUDGE EVALUATION
# ============================================================

MAX_CLAIMS_PER_QUERY = 12
JUDGE_MAX_NEW_TOKENS = 256
JUDGE_TEMPERATURE = 0.0
JUDGE_DO_SAMPLE = False

QA_RELEVANCE_PROMPT = """You are evaluating a RAG system.

Given:
- Question
- Answer

Score how well the Answer addresses the Question, regardless of grounding.
Return ONLY valid JSON:
{{
  "score": <integer 1-5>,
  "reason": "<short reason>"
}}

Question:
{question}

Answer:
{answer}
"""


def safe_json_loads(s: str) -> Dict[str, Any]:
    s = (s or "").strip()

    try:
        return _json_mod.loads(s)
    except Exception:
        pass

    m = re.search(r"\{.*\}", s, flags=re.DOTALL)

    if m:
        try:
            return _json_mod.loads(m.group(0))
        except Exception:
            return {}

    return {}


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()

    if not text:
        return []

    MIN_CLAIM_CHARS = 15

    line_parts = re.split(r"\n+", text)

    claims: List[str] = []

    for line in line_parts:
        line = line.strip()

        if not line:
            continue

        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line)
        sub_parts = re.split(r"(?<=[.!?])\s+", line)

        for p in sub_parts:
            p = p.strip()
            if len(p) >= MIN_CLAIM_CHARS:
                claims.append(p)

    if not claims and text:
        claims = [text]

    return claims


def judge_call_llm(model, tokenizer, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a careful evaluator. Return ONLY valid JSON, no preamble, no markdown.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text_input,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )

    # IMPORTANT FIX:
    # Put judge inputs on the same actual device as the model.
    target_device = next(model.parameters()).device
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": JUDGE_MAX_NEW_TOKENS,
        "do_sample": JUDGE_DO_SAMPLE,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }

    if JUDGE_DO_SAMPLE:
        gen_kwargs["temperature"] = JUDGE_TEMPERATURE

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def judge_answer_relevance(model, tokenizer, question: str, answer: str) -> Dict[str, Any]:
    raw = judge_call_llm(
        model,
        tokenizer,
        QA_RELEVANCE_PROMPT.format(question=question, answer=answer),
    )

    obj = safe_json_loads(raw)
    score = int(obj.get("score", 0) or 0)

    return {
        "answer_relevance_1to5": int(clamp(score, 1, 5)) if score else 0,
        "answer_relevance_reason": obj.get("reason", ""),
    }


def estimate_confidence(top_score: float, avg_score: float, groundedness: float) -> float:
    retrieval_signal = clamp((top_score + avg_score) / 2.0, 0.0, 1.0)
    conf = 0.55 * groundedness + 0.45 * retrieval_signal
    return round(clamp(conf, 0.0, 1.0), 4)


def estimate_query_coverage(query: str, answer: str) -> float:
    query_tokens = set(tokenize_words(query))
    answer_tokens = set(tokenize_words(answer))

    if not query_tokens:
        return 0.0

    covered = len(query_tokens.intersection(answer_tokens))
    return round(clamp(covered / len(query_tokens), 0.0, 1.0), 4)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdf_path", type=str, required=True)
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)

    parser.add_argument(
        "--model_path",
        type=str,
        default="/projectnb/cs585/students/siddhank/data_research/models/Llama-3.3-70B-Instruct",
        help=(
            "Path to a local model folder. Example: "
            "/projectnb/cs585/students/siddhank/data_research/models/Llama-3.3-70B-Instruct"
        ),
    )

    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true")

    parser.add_argument("--gpu_max_memory", type=str, default="36GiB")
    parser.add_argument("--cpu_max_memory", type=str, default="120GiB")

    parser.add_argument("--hf_cache_dir", type=str, default="./hf_cache")
    parser.add_argument("--quant_mode", type=str, default="4bit", choices=["4bit", "8bit", "none"])

    args = parser.parse_args()

    hf_cache_dir = Path(args.hf_cache_dir)
    output_csv_path = Path(args.output_csv)

    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir / "transformers")
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir / "datasets")

    print_header("Environment")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"))
    else:
        raise RuntimeError("No CUDA GPU detected. This script requires GPU.")

    pdf_path = Path(args.pdf_path)
    prompts_path = Path(args.prompts_path)
    model_path = Path(args.model_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    if model_path.exists() and model_path.is_dir():
        required = ["config.json"]

        tok_present = any(
            (model_path / f).exists()
            for f in ["tokenizer.json", "tokenizer.model", "tokenizer_config.json"]
        )

        weight_present = (
            any(model_path.glob("*.safetensors"))
            or any(model_path.glob("*.bin"))
            or any(model_path.glob("pytorch_model*.bin"))
        )

        for f in required:
            if not (model_path / f).exists():
                raise FileNotFoundError(
                    f"Missing {f} in {model_path}. "
                    "The folder does not look like a complete model snapshot."
                )

        if not tok_present:
            raise FileNotFoundError(
                f"No tokenizer file found in {model_path}. Expected one of: "
                "tokenizer.json, tokenizer.model, tokenizer_config.json"
            )

        if not weight_present:
            raise FileNotFoundError(
                f"No weight files found in {model_path}. Expected *.safetensors or *.bin"
            )

        print(f"Model folder OK: {model_path}")

    else:
        print(f"WARNING: --model_path is not a local folder: {model_path}")
        print(
            "Because local_files_only=True is used, Hugging Face IDs only work "
            "if already cached in HF_HOME."
        )

    prompts = load_prompts_from_txt(str(prompts_path))

    if not prompts:
        raise ValueError("Prompts file is empty.")

    print_header("Loading PDF")

    pages = load_pdf_text(str(pdf_path))

    if not pages:
        raise ValueError("No text could be extracted from the PDF.")

    print(f"Loaded {len(pages)} pages. PDF is only validated in no-retrieval baseline.")

    print_header("Loading LLM")

    tokenizer, model = load_llm(
        model_path=args.model_path,
        gpu_max_memory=args.gpu_max_memory,
        cpu_max_memory=args.cpu_max_memory,
        quant_mode=args.quant_mode,
    )

    print_header("Running No-Retrieval LLM baseline")

    rows = []

    for idx, query in enumerate(prompts, start=1):
        print(f"\n[{idx}/{len(prompts)}] Query: {query}")

        query_start = time.time()

        retrieved_docs_count = 0
        top_source = ""
        top_retrieval_score = 0.0
        avg_retrieval_score = 0.0

        gen = generate_answer(
            model=model,
            tokenizer=tokenizer,
            query=query,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
        )

        answer = gen["answer"]

        claims = split_sentences(answer)[:MAX_CLAIMS_PER_QUERY]
        claim_count = len(claims)
        unsupported_count = claim_count

        groundedness_score = 0.0
        hallucination_rate = 1.0 if claim_count > 0 else 0.0

        rel = judge_answer_relevance(
            model=model,
            tokenizer=tokenizer,
            question=query,
            answer=answer,
        )

        answer_relevance = rel["answer_relevance_1to5"]
        context_relevance = 1

        confidence = estimate_confidence(
            top_retrieval_score,
            avg_retrieval_score,
            groundedness_score,
        )

        query_coverage = estimate_query_coverage(query, answer)
        context_status = "no_context"

        row = {
            "query": query,
            "query_type": "no_retrieval",
            "answer": answer,

            "hallucination_rate": hallucination_rate,
            "groundedness_score": groundedness_score,
            "unsupported_count": unsupported_count,
            "claim_count": claim_count,

            "answer_relevance_1to5": answer_relevance,
            "answer_relevance_reason": rel["answer_relevance_reason"],
            "context_relevance_1to5": context_relevance,

            "confidence": confidence,

            "response_time_s": round(time.time() - query_start, 4),
            "llm_latency_s": round(gen["llm_latency_s"], 4),

            "gpu_throughput_toks_per_s": round(gen["gpu_throughput_toks_per_s"], 4),
            "eff_gpu_throughput": round(gen["eff_gpu_throughput"], 4),
            "gpu_util_percent": round(gen["gpu_util_percent"], 4),
            "gpu_mem_percent": round(gen["gpu_mem_percent"], 4),
            "gpu_util_max_percent": round(gen["gpu_util_max_percent"], 4),
            "gpu_mem_avg_percent": round(gen["gpu_mem_avg_percent"], 4),
            "gpu_mem_peak_mb": round(gen["gpu_mem_peak_mb"], 4),
            "gpu_mem_torch_peak_mb": round(gen["gpu_mem_torch_peak_mb"], 4),
            "gpu_monitor_samples": int(gen["gpu_monitor_samples"]),

            "prompt_tokens": int(gen["prompt_tokens"]),
            "generated_tokens": int(gen["generated_tokens"]),

            "total_deployment_cost_usd": 0.0,

            "top_source": top_source,
            "context_status": context_status,
            "answer_mode_used": "no_retrieval",
            "used_general_knowledge": True,

            "retrieved_docs_count": retrieved_docs_count,
            "top_retrieval_score": round(top_retrieval_score, 4),
            "avg_retrieval_score": round(avg_retrieval_score, 4),

            "query_coverage": query_coverage,
        }

        rows.append(row)

        pd.DataFrame(rows).to_csv(output_csv_path, index=False)

        print(f"Claims: {claim_count}")
        print(f"Answer relevance: {answer_relevance}/5")
        print("Answer preview:", answer[:180].replace("\n", " "), "...")
        print("Saved progress to:", output_csv_path)

        cleanup_memory()

    print_header("Done")

    final_df = pd.DataFrame(rows)
    final_df.to_csv(output_csv_path, index=False)

    print("Final CSV saved to:", output_csv_path)
    print("\nColumns:")
    print(list(final_df.columns))
    print("\nSample rows:")
    print(final_df.head(3))


if __name__ == "__main__":
    main()