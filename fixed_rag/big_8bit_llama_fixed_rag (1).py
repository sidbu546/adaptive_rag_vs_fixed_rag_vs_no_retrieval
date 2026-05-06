# ============================================================
# FIXED RAG FOR 1 PDF + FIXED PROMPTS + CSV OUTPUT
# GPU-ONLY Llama 3.3 70B Instruct 8-bit
# ============================================================
#
# IMPORTANT:
# 1) Restart kernel before running.
# 2) Set CUDA_VISIBLE_DEVICES in bash, not inside this script.
# 3) This script refuses CPU/offload for Llama.
# 4) If GPU cannot fit the model, it will fail instead of using CPU.
# 5) 8-bit Llama-70B needs ~70 GB just for weights. On an 80 GB A100,
#    KV cache headroom is tight. If you OOM, lower MAX_NEW_TOKENS,
#    truncate prompts more aggressively, or fall back to 4-bit.
# 6) 8-bit (bnb LLM.int8()) is significantly slower than 4-bit NF4.
#    Expect ~1.5-2.5x slower per token.
#
# Example bash:
#   export CUDA_VISIBLE_DEVICES=0
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   export TOKENIZERS_PARALLELISM=false
#   python Small_8bit_llama_gpu_only.py
# ============================================================

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
import gc
import re
import time
import threading
import warnings
from typing import List, Dict, Any
import json as _json_mod

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

try:
    import fitz  # PyMuPDF
    USE_FITZ = True
except ImportError:
    USE_FITZ = False
    from pypdf import PdfReader

from sentence_transformers import SentenceTransformer

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


# =========================
# 0) PATHS + CACHE
# =========================

PROJECT_ROOT = Path("/projectnb/cs585/students/siddhank/data_research")
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
ST_CACHE_DIR = PROJECT_ROOT / "sentence_transformers_cache"

for d in [
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    LOGS_DIR,
    HF_CACHE_DIR,
    ST_CACHE_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(ST_CACHE_DIR)


# =========================
# 1) USER CONFIG
# =========================

LOCAL_MODEL_PATH = str(MODELS_DIR / "Llama-3.3-70B-Instruct")
PDF_PATH = str(DATA_DIR / "ed3book_jan26.pdf")
OUTPUT_CSV_PATH = str(OUTPUTS_DIR / "fixed_rag_results_llama33_70b_8bit_big.csv")

# A100 80GB: leave headroom for KV cache. 8-bit weights ~70 GB.
# Setting too high gives no room for activations/KV cache and will OOM mid-generation.
GPU_MAX_MEMORY = "76GiB"

PROMPTS = [
    "Compare Byte-Pair Encoding, unigram tokenization, and character-level modeling for morphologically rich languages such as Turkish and Finnish; under what corpus and compute conditions would each approach outperform the others?",
    "If two language models have identical perplexity on a benchmark corpus but different downstream QA accuracy, analyze what this implies about the limits of perplexity as an evaluation metric.",
    "Compare n-gram smoothing methods, neural language models, and transformer pretraining for low-resource domains with rapidly shifting vocabulary such as medical emergencies.",
    "If a model uses retrieval-augmented generation and another uses only larger parametric memory, which is likely to age more gracefully over five years of changing knowledge, and why?",
    "Analyze whether contextual embeddings truly solved polysemy better than static embeddings, or whether they mainly shifted ambiguity into downstream fine-tuning.",
    "Compare logistic regression, feedforward neural networks, and transformers for sentiment classification when training data is small, noisy, and domain-shifted.",
    "If word error rate improves in an ASR system but user satisfaction falls, what hidden factors in dialogue systems or speech interfaces could explain the contradiction?",
    "Compare constituency parsing and dependency parsing for machine translation, information extraction, and question answering; where does each representation fail structurally?",
    "If a multilingual model performs well on English and Spanish but poorly on Thai and Tamil, determine whether tokenization, script frequency, morphology, or benchmark design is the most likely bottleneck.",
    "Analyze whether larger context windows reduce the need for external memory systems, or whether they mainly postpone retrieval problems at higher cost.",
    "Compare instruction tuning, preference optimization, and supervised fine-tuning as alignment strategies; which best improves usefulness without degrading factual robustness?",
    "If an LLM becomes stronger at chain-of-thought reasoning but weaker at calibration, is it actually more useful in high-stakes settings such as medicine or law?",
    "Compare dense retrieval and sparse retrieval for enterprise search when documents are technical, multilingual, and frequently updated.",
    "If a translation model achieves higher BLEU but lower human preference scores, what does this reveal about metric mismatch and semantic adequacy?",
    "Analyze whether scaling laws imply inevitable progress, or whether data quality, architecture changes, and energy constraints could break historical trends.",
    "Compare RNNs, LSTMs, and transformers for streaming speech recognition where latency matters more than peak benchmark accuracy.",
    "If a named entity recognizer performs poorly on emerging public figures and startups, determine whether the root cause is annotation lag, tokenization bias, knowledge staleness, or evaluation design.",
    "Compare FrameNet-style semantic role labeling and end-to-end transformer extraction for event understanding in legal documents with long cross-references.",
    "If two chatbots have equal benchmark scores but users trust one far more, analyze the roles of tone, uncertainty expression, memory consistency, and conversational repair.",
    "Compare whether future NLP progress is more likely to come from bigger models, better retrieval systems, multimodal grounding, or improved symbolic-neural hybrids."
]

CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
TOP_K_RETRIEVAL = 2

MAX_NEW_TOKENS = 220
TEMPERATURE = 0.0
DO_SAMPLE = False

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_QUERY_TYPE = "fixed_rag"
ANSWER_MODE_USED = "rag_only"
TOTAL_DEPLOYMENT_COST_USD = 0.0

MAX_CLAIMS_PER_QUERY = 12
JUDGE_MAX_NEW_TOKENS = 256
JUDGE_TEMPERATURE = 0.0
JUDGE_DO_SAMPLE = False


# =========================
# 2) UTILITIES
# =========================

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


def assert_local_llama_files_exist(local_model_path: str):
    p = Path(local_model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model path does not exist: {local_model_path}")

    files = list(p.glob("*"))
    if not files:
        raise FileNotFoundError(
            f"Llama folder is empty: {local_model_path}\n"
            f"Download the model files into this directory first."
        )

    present_names = {x.name for x in files}

    if "config.json" not in present_names:
        raise FileNotFoundError(f"Missing config.json in {local_model_path}")

    if not (("tokenizer.json" in present_names) or ("tokenizer.model" in present_names)):
        raise FileNotFoundError(
            f"Missing tokenizer.json or tokenizer.model in {local_model_path}"
        )

    has_weights = any(
        x.name.endswith(".safetensors") or x.name.endswith(".bin")
        for x in files
    )

    if not has_weights:
        raise FileNotFoundError(
            f"No model weights found in {local_model_path}"
        )


# =========================
# 3) PDF LOADING
# =========================

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


# =========================
# 4) CHUNKING
# =========================

def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size_words: int = 220,
    overlap_words: int = 40,
) -> List[Dict[str, Any]]:
    chunks = []

    for page in pages:
        words = page["text"].split()
        start = 0
        chunk_id = 0

        while start < len(words):
            end = min(start + chunk_size_words, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": f"page{page['page_num']}_chunk{chunk_id}",
                    "page_num": page["page_num"],
                    "text": chunk_text,
                    "source": f"{Path(PDF_PATH).name}:page_{page['page_num']}",
                })

            if end == len(words):
                break

            start = max(end - overlap_words, start + 1)
            chunk_id += 1

    return chunks


# =========================
# 5) EMBEDDINGS + RETRIEVAL
# =========================

def load_embedding_model(model_name: str):
    # Keep embedding model on CPU so GPU is reserved for Llama.
    return SentenceTransformer(
        model_name,
        cache_folder=str(ST_CACHE_DIR),
        device="cpu",
    )


def build_chunk_embeddings(embed_model, chunks: List[Dict[str, Any]]) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    embs = embed_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs


def retrieve_top_k(
    query: str,
    embed_model,
    chunk_embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    top_k: int = 3,
):
    q_emb = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    scores = np.dot(chunk_embeddings, q_emb)
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        item = dict(chunks[idx])
        item["retrieval_score"] = float(scores[idx])
        results.append(item)

    return results


# =========================
# 6) GPU MONITOR
# =========================

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
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def _run(self):
        while not self._stop_event.is_set():
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


# =========================
# 7) GPU-ONLY LLM LOADING
# =========================

def verify_model_fully_on_gpu(model):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if hasattr(model, "hf_device_map"):
        print("Device map:", model.hf_device_map)

        bad_locations = {
            name: dev
            for name, dev in model.hf_device_map.items()
            if str(dev) not in {"0", "cuda:0"}
        }

        if bad_locations:
            raise RuntimeError(
                "Model is not fully on GPU. Refusing to continue.\n"
                f"Non-GPU placements found: {bad_locations}"
            )

    for name, param in model.named_parameters():
        if param.device.type != "cuda":
            raise RuntimeError(
                f"Parameter is not on GPU: {name} is on {param.device}"
            )

    print("Verified: all model parameters are on GPU.")


def load_llama_model(local_model_path: str):
    assert_local_llama_files_exist(local_model_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Refusing to load Llama on CPU.")

    gpu_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    print_header("GPU check")
    print("Using GPU:", gpu_name)
    print(f"GPU memory: {total_gb:.2f} GiB")

    if total_gb < 78:
        print(
            "WARNING: This GPU appears to have less than 78 GiB memory. "
            "Llama 3.3 70B 8-bit GPU-only will almost certainly OOM. "
            "8-bit weights alone are ~70 GiB; you also need room for KV cache."
        )

    print_header("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cleanup_memory()

    print_header("Preparing GPU-only 8-bit config")
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    print_header("Loading Llama model GPU-only (8-bit)")

    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        quantization_config=bnb_config,

        # Force the whole quantized model onto visible GPU 0.
        # If you set CUDA_VISIBLE_DEVICES=1, that physical GPU becomes cuda:0 here.
        device_map={"": 0},

        # GPU-only memory limit. No CPU max-memory entry.
        max_memory={0: GPU_MAX_MEMORY},

        # No CPU/disk offload.
        offload_folder=None,
        offload_state_dict=False,

        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )

    model.eval()
    cleanup_memory()

    print("Model loaded successfully.")
    verify_model_fully_on_gpu(model)

    return tokenizer, model


# =========================
# 8) PROMPT BUILDING
# =========================

def build_rag_messages(query: str, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context_blocks = []

    for i, doc in enumerate(retrieved_docs, start=1):
        context_blocks.append(
            f"[Context {i} | source={doc['source']} | score={doc['retrieval_score']:.4f}]\n"
            f"{doc['text']}"
        )

    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a careful RAG assistant.\n"
        "Answer ONLY using the provided context.\n"
        "If the answer is not clearly present in the context, say that the context is insufficient.\n"
        "Be concise but complete.\n"
    )

    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Retrieved context:\n{context}\n\n"
        "Instructions:\n"
        "- Use only the retrieved context.\n"
        "- Do not rely on outside knowledge.\n"
        "- If context is insufficient, say so.\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# =========================
# 9) GENERATION
# =========================

def get_model_device(model):
    target_device = next(model.parameters()).device

    if target_device.type != "cuda":
        raise RuntimeError(f"Model is not on GPU. Found device: {target_device}")

    return target_device


def generate_answer(
    model,
    tokenizer,
    query: str,
    retrieved_docs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    messages = build_rag_messages(query, retrieved_docs)

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

    target_device = get_model_device(model)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    prompt_tokens = int(inputs["input_ids"].shape[1])

    cleanup_memory()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    gpu_monitor = GPUMonitor(interval_sec=0.2)
    gpu_monitor.start()

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE if DO_SAMPLE else None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

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


# =========================
# 10) LLM-JUDGE EVALUATION
# =========================

GROUNDING_PROMPT = """You are evaluating a RAG system.

Given:
- Context: passages retrieved from a knowledge base
- Claim: a single sentence from the model answer

Task:
Decide if the claim is FULLY supported by the Context.
- "supported": claim is directly stated or unambiguously implied by context
- "partially_supported": some support but missing key specifics / not fully justified
- "unsupported": not supported or contradicted by context

Return ONLY valid JSON with keys:
{{
  "label": "supported|partially_supported|unsupported",
  "evidence": "short quote or phrase from context or empty if unsupported"
}}

Context:
{context}

Claim:
{claim}
"""

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

CONTEXT_RELEVANCE_PROMPT = """You are evaluating a RAG system.

Given:
- Question
- Context

Score how relevant the Context is to answering the Question.
Return ONLY valid JSON:
{{
  "score": <integer 1-5>,
  "reason": "<short reason>"
}}

Question:
{question}

Context:
{context}
"""

LABEL_TO_SCORE = {
    "supported": 1.0,
    "partially_supported": 0.5,
    "unsupported": 0.0,
}


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

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", text)
    return [p.strip() for p in parts if p.strip()]


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

    target_device = get_model_device(model)
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=JUDGE_MAX_NEW_TOKENS,
            do_sample=JUDGE_DO_SAMPLE,
            temperature=JUDGE_TEMPERATURE if JUDGE_DO_SAMPLE else None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def judge_claims(
    model,
    tokenizer,
    context: str,
    answer: str,
    max_claims: int = MAX_CLAIMS_PER_QUERY,
) -> Dict[str, Any]:
    claims = split_sentences(answer)[:max_claims]
    judged = []

    for c in claims:
        p = GROUNDING_PROMPT.format(context=context, claim=c)
        raw = judge_call_llm(model, tokenizer, p)
        obj = safe_json_loads(raw)

        label = (obj.get("label") or "").strip().lower()

        if label not in LABEL_TO_SCORE:
            label = "unsupported"
            obj["evidence"] = ""

        judged.append({
            "claim": c,
            "label": label,
            "evidence": obj.get("evidence", ""),
        })

    if not judged:
        return {
            "claims": [],
            "hallucination_rate": 0.0,
            "groundedness_score": 0.0,
            "unsupported_count": 0,
            "claim_count": 0,
        }

    scores = [LABEL_TO_SCORE[j["label"]] for j in judged]
    unsupported = sum(1 for j in judged if j["label"] == "unsupported")
    claim_count = len(judged)

    return {
        "claims": judged,
        "hallucination_rate": float(unsupported / claim_count),
        "groundedness_score": float(np.mean(scores)),
        "unsupported_count": int(unsupported),
        "claim_count": int(claim_count),
    }


def judge_relevance(
    model,
    tokenizer,
    question: str,
    answer: str,
    context: str,
) -> Dict[str, Any]:
    qa_raw = judge_call_llm(
        model,
        tokenizer,
        QA_RELEVANCE_PROMPT.format(question=question, answer=answer),
    )

    qa_obj = safe_json_loads(qa_raw)
    qa_score = int(qa_obj.get("score", 0) or 0)

    ctx_raw = judge_call_llm(
        model,
        tokenizer,
        CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context),
    )

    ctx_obj = safe_json_loads(ctx_raw)
    ctx_score = int(ctx_obj.get("score", 0) or 0)

    return {
        "answer_relevance_1to5": int(clamp(qa_score, 1, 5)) if qa_score else 0,
        "answer_relevance_reason": qa_obj.get("reason", ""),
        "context_relevance_1to5": int(clamp(ctx_score, 1, 5)) if ctx_score else 0,
        "context_relevance_reason": ctx_obj.get("reason", ""),
    }


# =========================
# 11) METRICS
# =========================

def lexical_overlap_ratio(a: str, b: str) -> float:
    a_set = set(tokenize_words(a))
    b_set = set(tokenize_words(b))

    if not a_set or not b_set:
        return 0.0

    inter = len(a_set.intersection(b_set))
    return inter / max(1, len(a_set))


def estimate_confidence(
    top_score: float,
    avg_score: float,
    groundedness: float,
) -> float:
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


def infer_context_status(retrieved_docs_count: int, top_score: float) -> str:
    if retrieved_docs_count == 0:
        return "no_context"

    if top_score >= 0.60:
        return "strong_context"

    if top_score >= 0.35:
        return "partial_context"

    return "weak_context"


# =========================
# 12) MAIN PIPELINE
# =========================

def main():
    print_header("Environment")
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Refusing CPU execution.")

    print("GPU count:", torch.cuda.device_count())
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GiB")

    print_header("Loading PDF")

    if not Path(PDF_PATH).exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    pages = load_pdf_text(PDF_PATH)

    if not pages:
        raise ValueError("No text could be extracted from the PDF.")

    print(f"Loaded {len(pages)} pages.")

    print_header("Chunking PDF")

    chunks = chunk_pages(
        pages,
        chunk_size_words=CHUNK_SIZE_WORDS,
        overlap_words=CHUNK_OVERLAP_WORDS,
    )

    print(f"Created {len(chunks)} chunks.")

    print_header("Loading embedding model on CPU")
    embed_model = load_embedding_model(EMBED_MODEL_NAME)

    print_header("Building chunk embeddings")
    chunk_embeddings = build_chunk_embeddings(embed_model, chunks)
    print("Chunk embeddings shape:", chunk_embeddings.shape)

    print_header("Loading LLM GPU-only")
    tokenizer, model = load_llama_model(LOCAL_MODEL_PATH)

    print_header("Running fixed RAG")
    rows = []

    for idx, query in enumerate(PROMPTS, start=1):
        print(f"\n[{idx}/{len(PROMPTS)}] Query: {query}")

        query_start = time.time()

        retrieved_docs = retrieve_top_k(
            query=query,
            embed_model=embed_model,
            chunk_embeddings=chunk_embeddings,
            chunks=chunks,
            top_k=TOP_K_RETRIEVAL,
        )

        retrieved_docs_count = len(retrieved_docs)
        top_source = retrieved_docs[0]["source"] if retrieved_docs else ""
        top_retrieval_score = float(retrieved_docs[0]["retrieval_score"]) if retrieved_docs else 0.0
        avg_retrieval_score = float(np.mean([d["retrieval_score"] for d in retrieved_docs])) if retrieved_docs else 0.0

        gen = generate_answer(model, tokenizer, query, retrieved_docs)
        answer = gen["answer"]

        joined_context = "\n\n".join(
            f"[Context {i} | source={d['source']}]\n{d['text']}"
            for i, d in enumerate(retrieved_docs, start=1)
        )

        print("  judging claims...")

        grounding = judge_claims(
            model,
            tokenizer,
            context=joined_context,
            answer=answer,
            max_claims=MAX_CLAIMS_PER_QUERY,
        )

        groundedness_score = float(grounding["groundedness_score"])
        hallucination_rate = float(grounding["hallucination_rate"])
        claim_count = int(grounding["claim_count"])
        unsupported_count = int(grounding["unsupported_count"])

        print("  judging relevance...")

        rel = judge_relevance(
            model,
            tokenizer,
            question=query,
            answer=answer,
            context=joined_context,
        )

        answer_relevance = int(rel["answer_relevance_1to5"] or 0)
        context_relevance = int(rel["context_relevance_1to5"] or 0)

        confidence = estimate_confidence(
            top_retrieval_score,
            avg_retrieval_score,
            groundedness_score,
        )

        query_coverage = estimate_query_coverage(query, answer)
        context_status = infer_context_status(retrieved_docs_count, top_retrieval_score)
        used_general_knowledge = False
        total_response_time = time.time() - query_start

        row = {
            "query": query,
            "query_type": DEFAULT_QUERY_TYPE,
            "answer": answer,
            "hallucination_rate": round(hallucination_rate, 4),
            "groundedness_score": round(groundedness_score, 4),
            "claim_count": claim_count,
            "unsupported_count": unsupported_count,
            "answer_relevance_1to5": answer_relevance,
            "context_relevance_1to5": context_relevance,
            "confidence": confidence,
            "response_time_s": round(total_response_time, 4),
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
            "total_deployment_cost_usd": TOTAL_DEPLOYMENT_COST_USD,
            "top_source": top_source,
            "context_status": context_status,
            "answer_mode_used": ANSWER_MODE_USED,
            "used_general_knowledge": used_general_knowledge,
            "retrieved_docs_count": retrieved_docs_count,
            "top_retrieval_score": round(top_retrieval_score, 4),
            "avg_retrieval_score": round(avg_retrieval_score, 4),
            "query_coverage": query_coverage,
        }

        rows.append(row)
        pd.DataFrame(rows).to_csv(OUTPUT_CSV_PATH, index=False)

        print("Top source:", top_source)
        print("Top retrieval score:", round(top_retrieval_score, 4))
        print("Answer preview:", answer[:180].replace("\n", " "), "...")
        print("Saved progress to:", OUTPUT_CSV_PATH)

        cleanup_memory()

    print_header("Done")

    final_df = pd.DataFrame(rows)
    final_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Final CSV saved to:", OUTPUT_CSV_PATH)
    print("\nColumns:")
    print(list(final_df.columns))
    print("\nSample rows:")
    print(final_df.head(3))


if __name__ == "__main__":
    main()