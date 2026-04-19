# ============================================================
# FIXED RAG FOR 1 PDF + FIXED PROMPTS + CSV OUTPUT
# Safe for Qwen2.5-32B-Instruct 4-bit on RTX A6000
# ============================================================

# IMPORTANT:
# 1) Restart kernel before running.
# 2) Set environment variables BEFORE importing torch/transformers.
# 3) If you still hit OOM, reduce GPU_MAX_MEMORY below.
# ============================================================
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os
from pathlib import Path

# =========================
# 0) PATHS + CACHE
# =========================

PROJECT_ROOT = Path("/projectnb/cs505am/students/kaushik1/fixed_rag_project")
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = PROJECT_ROOT / "logs"
HF_CACHE_DIR = PROJECT_ROOT / "hf_cache"
ST_CACHE_DIR = PROJECT_ROOT / "sentence_transformers_cache"
OFFLOAD_DIR = PROJECT_ROOT / "offload"

for d in [PROJECT_ROOT, DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR, HF_CACHE_DIR, ST_CACHE_DIR, OFFLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE_DIR / "datasets")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(ST_CACHE_DIR)

# Helps reduce fragmentation-related CUDA OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Optional: avoid tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# 1) IMPORTS
# =========================

import gc
import re
import math
import time
import json
import queue
import threading
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

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

# Embeddings
from sentence_transformers import SentenceTransformer

# GPU monitor
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


# =========================
# 2) USER CONFIG
# =========================

LOCAL_MODEL_PATH = str(MODELS_DIR / "Qwen2.5-32B-Instruct")
PDF_PATH = str(DATA_DIR / "Small Corpus.pdf")
OUTPUT_CSV_PATH = str(OUTPUTS_DIR / "fixed_rag_results_qwen32b_4bit.csv")

# Fixed prompt list provided by you
PROMPTS = [
    "What architecture is proposed in Attention Is All You Need?",
    "How many encoder and decoder layers are used in the base Transformer model?",
    "What is the model dimensionality (d_model) used in the base model?",
    "What optimizer is used during training in the paper?",
    "Why does the Transformer train faster than recurrent sequence models?",
    "Why is positional encoding necessary in a model without recurrence or convolution?",
    "How does multi-head attention improve learning compared with a single attention head?",
    "Why is scaled dot-product attention divided by sqrt(dk) before softmax?",
    "Explain how the Transformer replaces recurrence while still modeling dependencies between distant tokens.",
    "How do residual connections and layer normalization contribute to stable training?",
    "Why does the combination of self-attention and feed-forward layers work effectively for sequence transduction?",
    "How does the decoder ensure autoregressive generation while using self-attention?",
    "Compare the Transformer with recurrent neural networks in terms of parallelization.",
    "Compare self-attention and encoder-decoder attention.",
    "Compare sinusoidal positional encodings and learned positional embeddings.",
    "Compare the base Transformer model and the big Transformer model.",
    "Why was the Transformer considered a breakthrough?",
    "What does attention actually mean in this paper?",
    "Why are multiple heads better than one?",
    "How does the model understand sequence order if all tokens are processed simultaneously?"
]

# Retrieval config
CHUNK_SIZE_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
TOP_K_RETRIEVAL = 3

# Generation config
MAX_NEW_TOKENS = 220
TEMPERATURE = 0.0
DO_SAMPLE = False

# Safer memory caps for RTX A6000 48GB class GPU.
# Start conservative.
GPU_MAX_MEMORY = "36GiB"   # try 34GiB if OOM persists
CPU_MAX_MEMORY = "120GiB"

# Embedding model
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Query typing
DEFAULT_QUERY_TYPE = "fixed_rag"

# Answer mode
ANSWER_MODE_USED = "rag_only"

# Deployment cost (local run = 0 unless you want to set estimate)
TOTAL_DEPLOYMENT_COST_USD = 0.0


# =========================
# 3) UTILITIES
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

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_words(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

def cosine_similarity_matrix(query_vec: np.ndarray, doc_mat: np.ndarray) -> np.ndarray:
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    d = doc_mat / (np.linalg.norm(doc_mat, axis=1, keepdims=True) + 1e-12)
    return np.dot(d, q)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# =========================
# 4) PDF LOADING
# =========================

def load_pdf_text(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Returns list of pages:
    [{"page_num": 1, "text": "..."}]
    """
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
# 5) CHUNKING
# =========================

def chunk_pages(pages: List[Dict[str, Any]], chunk_size_words=220, overlap_words=40) -> List[Dict[str, Any]]:
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
                    "source": f"{Path(PDF_PATH).name}:page_{page['page_num']}"
                })

            if end == len(words):
                break

            start = max(end - overlap_words, start + 1)
            chunk_id += 1

    return chunks


# =========================
# 6) EMBEDDINGS + RETRIEVAL
# =========================

def load_embedding_model(model_name: str):
    return SentenceTransformer(model_name, cache_folder=str(ST_CACHE_DIR))

def build_chunk_embeddings(embed_model, chunks: List[Dict[str, Any]]) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    embs = embed_model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embs

def retrieve_top_k(query: str, embed_model, chunk_embeddings: np.ndarray, chunks: List[Dict[str, Any]], top_k=3):
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = np.dot(chunk_embeddings, q_emb)
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        item = dict(chunks[idx])
        item["retrieval_score"] = float(scores[idx])
        results.append(item)

    return results


# =========================
# 7) GPU MONITOR
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
                    "timestamp": time.time()
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
                "gpu_monitor_samples": 0
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
            "gpu_monitor_samples": int(len(self.samples))
        }


# =========================
# 8) LLM LOADING
# =========================

def load_qwen_model(local_model_path: str):
    print_header("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        local_files_only=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cleanup_memory()

    print_header("Preparing 4-bit config")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    max_memory = {}
    if torch.cuda.is_available():
        max_memory[0] = GPU_MAX_MEMORY
    max_memory["cpu"] = CPU_MAX_MEMORY

    print_header("Loading Qwen model safely")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
        offload_folder=str(OFFLOAD_DIR),
        offload_state_dict=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    )

    model.eval()
    cleanup_memory()

    print("Model loaded successfully.")
    if hasattr(model, "hf_device_map"):
        print("Device map:", model.hf_device_map)

    return tokenizer, model


# =========================
# 9) PROMPT BUILDING
# =========================

def build_rag_messages(query: str, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context_blocks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        context_blocks.append(
            f"[Context {i} | source={doc['source']} | score={doc['retrieval_score']:.4f}]\n{doc['text']}"
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
# 10) GENERATION
# =========================

def generate_answer(model, tokenizer, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    messages = build_rag_messages(query, retrieved_docs)

    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text_input,
        return_tensors="pt",
        truncation=True,
        max_length=4096
    )

    # Move tensors to the same device as first input-receiving module
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    prompt_tokens = int(inputs["input_ids"].shape[1])

    cleanup_memory()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

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
            use_cache=True
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
# 11) HEURISTIC EVALUATION
# =========================
# NOTE:
# These are lightweight heuristics, not gold-standard evaluator metrics.
# They are useful for fixed benchmarking across your runs.

def lexical_overlap_ratio(a: str, b: str) -> float:
    a_set = set(tokenize_words(a))
    b_set = set(tokenize_words(b))
    if not a_set or not b_set:
        return 0.0
    inter = len(a_set.intersection(b_set))
    return inter / max(1, len(a_set))

def estimate_groundedness(answer: str, contexts: List[str]) -> float:
    if not answer.strip() or not contexts:
        return 0.0
    joined_context = " ".join(contexts)
    overlap = lexical_overlap_ratio(answer, joined_context)
    return round(clamp(overlap, 0.0, 1.0), 4)

def estimate_hallucination_rate(answer: str, contexts: List[str]) -> float:
    groundedness = estimate_groundedness(answer, contexts)
    return round(clamp(1.0 - groundedness, 0.0, 1.0), 4)

def estimate_answer_relevance_1to5(query: str, answer: str) -> int:
    overlap = lexical_overlap_ratio(query, answer)
    score = 1 + int(round(clamp(overlap * 5, 0, 4)))
    return int(clamp(score, 1, 5))

def estimate_context_relevance_1to5(query: str, contexts: List[str]) -> int:
    if not contexts:
        return 1
    joined_context = " ".join(contexts)
    overlap = lexical_overlap_ratio(query, joined_context)
    score = 1 + int(round(clamp(overlap * 5, 0, 4)))
    return int(clamp(score, 1, 5))

def estimate_confidence(top_score: float, avg_score: float, groundedness: float) -> float:
    # Combine retrieval strength + groundedness
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
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU only.")

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
        overlap_words=CHUNK_OVERLAP_WORDS
    )
    print(f"Created {len(chunks)} chunks.")

    print_header("Loading embedding model")
    embed_model = load_embedding_model(EMBED_MODEL_NAME)

    print_header("Building chunk embeddings")
    chunk_embeddings = build_chunk_embeddings(embed_model, chunks)
    print("Chunk embeddings shape:", chunk_embeddings.shape)

    print_header("Loading LLM")
    tokenizer, model = load_qwen_model(LOCAL_MODEL_PATH)

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
            top_k=TOP_K_RETRIEVAL
        )

        retrieved_docs_count = len(retrieved_docs)
        top_source = retrieved_docs[0]["source"] if retrieved_docs else ""
        top_retrieval_score = float(retrieved_docs[0]["retrieval_score"]) if retrieved_docs else 0.0
        avg_retrieval_score = float(np.mean([d["retrieval_score"] for d in retrieved_docs])) if retrieved_docs else 0.0

        gen = generate_answer(model, tokenizer, query, retrieved_docs)
        answer = gen["answer"]

        context_texts = [d["text"] for d in retrieved_docs]
        groundedness_score = estimate_groundedness(answer, context_texts)
        hallucination_rate = estimate_hallucination_rate(answer, context_texts)
        answer_relevance = estimate_answer_relevance_1to5(query, answer)
        context_relevance = estimate_context_relevance_1to5(query, context_texts)
        confidence = estimate_confidence(top_retrieval_score, avg_retrieval_score, groundedness_score)
        query_coverage = estimate_query_coverage(query, answer)

        context_status = infer_context_status(retrieved_docs_count, top_retrieval_score)

        used_general_knowledge = False  # forced by prompt design
        total_response_time = time.time() - query_start

        row = {
            "query": query,
            "query_type": DEFAULT_QUERY_TYPE,
            "answer": answer,
            "hallucination_rate": hallucination_rate,
            "groundedness_score": groundedness_score,
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

        # Save incrementally after each query
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