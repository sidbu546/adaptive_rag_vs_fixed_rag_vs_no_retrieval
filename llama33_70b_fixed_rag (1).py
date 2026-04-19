# ============================================================
# FIXED RAG FOR 1 PDF + PROMPT TXT + CSV OUTPUT
# Llama 3.3 70B Instruct - 8bit version
# ============================================================

import os

# -------------------------
# ENV SETUP (set before torch/transformers)
# -------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import re
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

# Embeddings
from sentence_transformers import SentenceTransformer

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
# CHUNKING
# ============================================================

def chunk_pages(
    pages: List[Dict[str, Any]],
    pdf_name: str,
    chunk_size_words: int = 260,
    overlap_words: int = 50,
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
                    "source": f"{pdf_name}:page_{page['page_num']}",
                })

            if end == len(words):
                break

            start = max(end - overlap_words, start + 1)
            chunk_id += 1

    return chunks


# ============================================================
# EMBEDDINGS + RETRIEVAL
# ============================================================

def load_embedding_model(model_name: str, cache_dir: str):
    return SentenceTransformer(model_name, cache_folder=cache_dir, device="cpu")


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
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    q_emb = embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = np.dot(chunk_embeddings, q_emb)
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        item = dict(chunks[idx])
        item["retrieval_score"] = float(scores[idx])
        results.append(item)

    return results


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


# ============================================================
# LLM LOADING
# ============================================================

def load_llama_model(
    model_path: str,
    gpu_max_memory: str,
    cpu_max_memory: str,
    quant_mode: str = "8bit",
):
    print_header("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=False,
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
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )
    elif quant_mode == "8bit":
        print_header("Preparing 8-bit config")
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant_mode != "none":
        raise ValueError(f"Unsupported quant_mode: {quant_mode}")

    max_memory = {}
    if torch.cuda.is_available():
        max_memory[0] = gpu_max_memory
    max_memory["cpu"] = cpu_max_memory

    print_header("Loading Llama model safely")
    model_kwargs = {
        "device_map": "auto",
        "max_memory": max_memory,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }

    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **model_kwargs,
    )

    model.eval()
    cleanup_memory()

    print("Model loaded successfully.")
    if hasattr(model, "hf_device_map"):
        print("Device map:", model.hf_device_map)

    return tokenizer, model


# ============================================================
# PROMPT BUILDING
# ============================================================

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


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    model,
    tokenizer,
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
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

    target_device = "cuda" if torch.cuda.is_available() else "cpu"
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
# HEURISTIC EVALUATION
# ============================================================

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


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, required=True)
    parser.add_argument("--prompts_path", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--quant_mode", type=str, default="8bit", choices=["4bit", "8bit", "none"])
    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--chunk_size_words", type=int, default=260)
    parser.add_argument("--chunk_overlap_words", type=int, default=50)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--gpu_max_memory", type=str, default="110GiB")
    parser.add_argument("--cpu_max_memory", type=str, default="180GiB")
    parser.add_argument("--hf_cache_dir", type=str, default="./hf_cache")
    parser.add_argument("--st_cache_dir", type=str, default="./sentence_transformers_cache")
    args = parser.parse_args()

    hf_cache_dir = Path(args.hf_cache_dir)
    st_cache_dir = Path(args.st_cache_dir)
    output_csv_path = Path(args.output_csv)

    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    st_cache_dir.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hf_cache_dir / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir / "transformers")
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir / "datasets")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(st_cache_dir)

    print_header("Environment")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU only.")

    pdf_path = Path(args.pdf_path)
    prompts_path = Path(args.prompts_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    prompts = load_prompts_from_txt(str(prompts_path))
    if not prompts:
        raise ValueError("Prompts file is empty.")

    print_header("Loading PDF")
    pages = load_pdf_text(str(pdf_path))
    if not pages:
        raise ValueError("No text could be extracted from the PDF.")
    print(f"Loaded {len(pages)} pages.")

    print_header("Chunking PDF")
    chunks = chunk_pages(
        pages=pages,
        pdf_name=pdf_path.name,
        chunk_size_words=args.chunk_size_words,
        overlap_words=args.chunk_overlap_words,
    )
    print(f"Created {len(chunks)} chunks.")

    print_header("Loading embedding model")
    embed_model = load_embedding_model(args.embed_model, str(st_cache_dir))

    print_header("Building chunk embeddings")
    chunk_embeddings = build_chunk_embeddings(embed_model, chunks)
    print("Chunk embeddings shape:", chunk_embeddings.shape)

    print_header("Loading LLM")
    tokenizer, model = load_llama_model(
        model_path=args.model_path,
        gpu_max_memory=args.gpu_max_memory,
        cpu_max_memory=args.cpu_max_memory,
        quant_mode=args.quant_mode,
    )

    print_header("Running fixed RAG")
    rows = []

    for idx, query in enumerate(prompts, start=1):
        print(f"\n[{idx}/{len(prompts)}] Query: {query}")

        query_start = time.time()

        retrieved_docs = retrieve_top_k(
            query=query,
            embed_model=embed_model,
            chunk_embeddings=chunk_embeddings,
            chunks=chunks,
            top_k=args.top_k,
        )

        retrieved_docs_count = len(retrieved_docs)
        top_source = retrieved_docs[0]["source"] if retrieved_docs else ""
        top_retrieval_score = float(retrieved_docs[0]["retrieval_score"]) if retrieved_docs else 0.0
        avg_retrieval_score = float(np.mean([d["retrieval_score"] for d in retrieved_docs])) if retrieved_docs else 0.0

        gen = generate_answer(
            model=model,
            tokenizer=tokenizer,
            query=query,
            retrieved_docs=retrieved_docs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
        )
        answer = gen["answer"]

        context_texts = [d["text"] for d in retrieved_docs]
        groundedness_score = estimate_groundedness(answer, context_texts)
        hallucination_rate = estimate_hallucination_rate(answer, context_texts)
        answer_relevance = estimate_answer_relevance_1to5(query, answer)
        context_relevance = estimate_context_relevance_1to5(query, context_texts)
        confidence = estimate_confidence(top_retrieval_score, avg_retrieval_score, groundedness_score)
        query_coverage = estimate_query_coverage(query, answer)

        context_status = infer_context_status(retrieved_docs_count, top_retrieval_score)

        row = {
            "query": query,
            "query_type": "fixed_rag",
            "answer": answer,
            "hallucination_rate": hallucination_rate,
            "groundedness_score": groundedness_score,
            "answer_relevance_1to5": answer_relevance,
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
            "total_deployment_cost_usd": 0.0,
            "top_source": top_source,
            "context_status": context_status,
            "answer_mode_used": "rag_only",
            "used_general_knowledge": False,
            "retrieved_docs_count": retrieved_docs_count,
            "top_retrieval_score": round(top_retrieval_score, 4),
            "avg_retrieval_score": round(avg_retrieval_score, 4),
            "query_coverage": query_coverage,
        }

        rows.append(row)
        pd.DataFrame(rows).to_csv(output_csv_path, index=False)

        print("Top source:", top_source)
        print("Top retrieval score:", round(top_retrieval_score, 4))
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