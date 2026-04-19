import re
import time
import enum
from copy import copy as _copy
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple


def now() -> float:
    return time.perf_counter()


def _to_text(x):
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        for k in ("answer", "output", "text", "content", "generated_text", "response"):
            if k in x and isinstance(x[k], str):
                return x[k]
        return str(x)
    if hasattr(x, "content") and isinstance(getattr(x, "content"), str):
        return x.content
    if isinstance(x, list) and x:
        return _to_text(x[0])
    return str(x)


def _call_llm(llm, prompt: str) -> str:
    if hasattr(llm, "invoke"):
        return _to_text(llm.invoke(prompt))
    if hasattr(llm, "generate"):
        return _to_text(llm.generate(prompt))
    if callable(llm):
        return _to_text(llm(prompt))
    raise TypeError("llm must provide .invoke(prompt), .generate(prompt), or be callable.")


def _call_llm_timed(llm, prompt: str) -> Dict[str, Any]:
    if hasattr(llm, "invoke_timed"):
        out = llm.invoke_timed(prompt)
        out.setdefault("llm_latency_s", None)
        out.setdefault("output_tokens", None)
        out.setdefault("tokens_per_sec", None)
        return out

    t0 = time.perf_counter()
    text = _call_llm(llm, prompt)
    t1 = time.perf_counter()
    return {"text": text, "llm_latency_s": float(t1 - t0), "output_tokens": None, "tokens_per_sec": None}


# ─── Query Types ──────────────────────────────────────────────────────────────

class QueryType(enum.Enum):
    FACTUAL    = "factual"     # direct lookup, single-hop, precise answer
    MULTI_HOP  = "multi_hop"   # connect evidence across multiple docs
    ANALYTICAL = "analytical"  # evaluate, analyze, essay-style reasoning
    COMPARISON = "comparison"  # compare / contrast entities or concepts
    AMBIGUOUS  = "ambiguous"   # vague, under-specified, or opinion-seeking


# ─── Retrieval Policies ───────────────────────────────────────────────────────

@dataclass
class RetrievalPolicy:
    """Per-query-type retrieval configuration for Adaptive RAG (RARC)."""
    k: int                            # final result count returned
    pool_size: int                    # initial dense candidate pool size
    context_chars: int                # max chars fed to the LLM prompt
    neighbor_window: int              # ±N neighbor chunks included per hit
    max_per_source: int               # diversity cap per source file
    expand_if_low_coverage: bool = False  # retry with larger k if coverage low
    expansion_k: int = 0              # k to use on the expansion pass
    coverage_threshold: float = 0.25  # coverage below this → trigger expansion


RETRIEVAL_POLICIES: Dict[QueryType, RetrievalPolicy] = {
    # Short, targeted: small k, no neighbor chunks, tight context window
    QueryType.FACTUAL: RetrievalPolicy(
        k=3, pool_size=40, context_chars=6_000,
        neighbor_window=0, max_per_source=2,
    ),
    # Multi-doc chaining: large k, wide pool, extra neighbor context
    QueryType.MULTI_HOP: RetrievalPolicy(
        k=8, pool_size=80, context_chars=16_000,
        neighbor_window=2, max_per_source=4,
    ),
    # Essay/evaluation: moderate k, generous context, 1 neighbor
    QueryType.ANALYTICAL: RetrievalPolicy(
        k=6, pool_size=60, context_chars=14_000,
        neighbor_window=1, max_per_source=3,
    ),
    # Comparison: low max_per_source forces source diversity across entities
    QueryType.COMPARISON: RetrievalPolicy(
        k=6, pool_size=60, context_chars=12_000,
        neighbor_window=1, max_per_source=2,
    ),
    # Ambiguous: start small, expand if coverage is insufficient
    QueryType.AMBIGUOUS: RetrievalPolicy(
        k=4, pool_size=60, context_chars=10_000,
        neighbor_window=1, max_per_source=3,
        expand_if_low_coverage=True, expansion_k=8, coverage_threshold=0.25,
    ),
}


# ─── Heuristic Classifier ─────────────────────────────────────────────────────

_MULTI_HOP_SIGNALS = [
    "what led to", "what caused", "chain of events", "sequence of",
    "in relation to", "following the", "as a result of",
    "after the", "what role did", "what happened after",
    "how does .* relate", "how did .* influence", "connection between",
]
_ANALYTICAL_SIGNALS = [
    "critically examine", "evaluate", "justify", "analyse", "analyze",
    "discuss", "assess", "to what extent", "with examples", "explain the",
    "implications of", "significance of", "impact of", "role of",
    "political, social", "structural reasons", "causes and effects",
    "in 200 words", "in 300 words", "in 350 words", "in 400 words",
    "how did .* fail", "why did .* fail",
]
_COMPARISON_SIGNALS = [
    "compare", "contrast", "difference between", "similarities between",
    "versus", " vs ", "better than", "worse than",
    "which is more", "how do .* differ", "were their goals",
    "both .* and",
]
_AMBIGUOUS_SIGNALS = [
    "what do you think", "your opinion", "is it possible", "could it be",
    "do you believe", "maybe", "perhaps", "any thoughts",
    "something about", "tell me about",
]


def _classify_heuristic(query: str) -> QueryType:
    """Structured heuristic fallback — no model required."""
    q = (query or "").lower()
    words = q.split()

    # Comparison signals are most specific — check first
    if any(s in q for s in _COMPARISON_SIGNALS):
        return QueryType.COMPARISON

    # Multi-hop: connective / causal reasoning with enough words to be substantive
    if any(re.search(s, q) for s in _MULTI_HOP_SIGNALS) and len(words) >= 8:
        return QueryType.MULTI_HOP

    # Analytical: explicit essay cues or long query without comparison
    if any(s in q for s in _ANALYTICAL_SIGNALS) or (
        len(words) >= 14 and query.count("?") <= 1
    ):
        return QueryType.ANALYTICAL

    # Ambiguous: opinion / vague signals, or very short query
    if any(s in q for s in _AMBIGUOUS_SIGNALS) or len(words) <= 4:
        return QueryType.AMBIGUOUS

    return QueryType.FACTUAL


# ─── Classifier Classes ───────────────────────────────────────────────────────

class QueryClassifier:
    """
    Heuristic baseline classifier — no model required.
    Used as the default and as the fallback for unfit probe classifiers.
    """
    def classify(self, query: str) -> QueryType:
        return _classify_heuristic(query)


class LLMZeroShotClassifier(QueryClassifier):
    """
    Zero-shot classification via the generator itself.

    The LLM's one-word response is a direct product of its internal activations,
    making this a lightweight, training-free approach to query typing that
    leverages the model's full language understanding.
    Falls back to heuristic on any generation error.
    """
    _LABEL_MAP = {
        "factual":    QueryType.FACTUAL,
        "multi_hop":  QueryType.MULTI_HOP,
        "multi-hop":  QueryType.MULTI_HOP,
        "analytical": QueryType.ANALYTICAL,
        "comparison": QueryType.COMPARISON,
        "ambiguous":  QueryType.AMBIGUOUS,
    }

    def __init__(self, llm_manager):
        self.llm = llm_manager

    def classify(self, query: str) -> QueryType:
        prompt = (
            "Classify the following search query into EXACTLY ONE category.\n"
            "Categories:\n"
            "  factual    – direct fact lookup, precise single answer\n"
            "  multi_hop  – requires chaining evidence across multiple documents\n"
            "  analytical – requires evaluation, essay-style analysis or synthesis\n"
            "  comparison – compares or contrasts two or more entities\n"
            "  ambiguous  – vague, under-specified, or opinion-seeking\n\n"
            f"Query: {query}\n\n"
            "Respond with ONE word only (no punctuation, no explanation):\n"
        )
        try:
            out = self.llm.invoke_timed(prompt)
            raw = (out.get("text") or "").strip().lower().split()[0].strip(".,;:")
            return self._LABEL_MAP.get(raw, _classify_heuristic(query))
        except Exception:
            return _classify_heuristic(query)


class ProbeClassifier(QueryClassifier):
    """
    Linear probe trained on the generator's last-layer hidden states.

    Implements the professor's suggestion: 'a classifier trained on the
    generator's activations might just work.'  Mean-pooling the final hidden
    layer gives a compact query representation; a logistic regression probe
    is then trained on labeled examples.

    Before .fit() or .load() is called the classifier degrades gracefully to
    the heuristic baseline — no hard dependency on sklearn at import time.

    Usage
    -----
    probe = ProbeClassifier(llm_manager)
    probe.fit(labeled_queries)       # list of (query_str, label_str)
    qtype = probe.classify("…")
    probe.save("probe.pkl")          # persist between runs
    probe.load("probe.pkl")          # restore
    """

    def __init__(self, llm_manager):
        self.llm = llm_manager
        self._probe = None   # sklearn LogisticRegression, set by .fit()

    # ── internal feature extractor ──────────────────────────────────────────

    def _get_hidden_state(self, query: str):
        """Mean-pool the last hidden layer → compact, stable 1-D embedding."""
        import torch
        import numpy as np
        model = self.llm.model
        tok   = self.llm.tokenizer
        inputs = tok(query, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        h = out.hidden_states[-1].mean(dim=1).squeeze(0).float().cpu().numpy()
        return h

    # ── public API ───────────────────────────────────────────────────────────

    def fit(self, labeled_pairs) -> None:
        """
        Train the probe.

        labeled_pairs : iterable of (query_str, label_str)
        label_str must be one of:
            'factual' | 'multi_hop' | 'analytical' | 'comparison' | 'ambiguous'
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        queries, labels = zip(*labeled_pairs)
        X = np.vstack([self._get_hidden_state(q) for q in queries])
        self._probe = LogisticRegression(
            C=1.0, max_iter=500, solver="lbfgs", multi_class="auto"
        )
        self._probe.fit(X, list(labels))
        print(
            f"[ProbeClassifier] Fitted on {len(queries)} examples  "
            f"classes={self._probe.classes_.tolist()}"
        )

    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self._probe, f)

    def load(self, path: str) -> None:
        import pickle
        with open(path, "rb") as f:
            self._probe = pickle.load(f)

    def classify(self, query: str) -> QueryType:
        if self._probe is None:
            return _classify_heuristic(query)
        h = self._get_hidden_state(query).reshape(1, -1)
        label = self._probe.predict(h)[0]
        return LLMZeroShotClassifier._LABEL_MAP.get(label, _classify_heuristic(query))


def make_classifier(mode: str = "heuristic", llm_manager=None) -> QueryClassifier:
    """
    Factory for RARC query classifiers.

    mode = 'heuristic'  – regex/keyword rules, no model needed (default)
           'llm'        – zero-shot via the generator (recommended, no training data)
           'probe'      – hidden-state linear probe (call .fit() before use)
    """
    if mode == "llm" and llm_manager is not None:
        return LLMZeroShotClassifier(llm_manager)
    if mode == "probe" and llm_manager is not None:
        return ProbeClassifier(llm_manager)
    return QueryClassifier()


# Backward-compat shim: existing callers that still import _is_complex_prompt
def _is_complex_prompt(query: str) -> bool:
    return _classify_heuristic(query) in (
        QueryType.MULTI_HOP, QueryType.ANALYTICAL, QueryType.COMPARISON
    )


# ─── Utilities ────────────────────────────────────────────────────────────────

def _extract_length_hint(query: str) -> str:
    m = re.search(r"\b(\d{2,4})\s+words?\b", query.lower())
    if not m:
        return ""
    try:
        n = int(m.group(1))
    except Exception:
        return ""
    return f"Target length: about {n} words."


def _prepare_context(results: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    sections: List[str] = []
    used = 0
    for idx, doc in enumerate(results, start=1):
        md = doc.get("metadata") or {}
        source   = md.get("source_file", md.get("source", "unknown"))
        page     = md.get("page", "unknown")
        chunk_id = md.get("chunk_id", "NA")
        score    = doc.get("similarity_score")
        label    = (
            f"[Doc {idx} | source={source} | page={page} "
            f"| chunk_id={chunk_id} | score={score}]"
        )
        content = (doc.get("content") or "").strip()
        block = f"{label}\n{content}"
        if used + len(block) > max_chars:
            remaining = max_chars - used
            if remaining > 200:
                sections.append(block[:remaining])
            break
        sections.append(block)
        used += len(block)
    return "\n\n".join(sections)


def _estimate_context_strength(results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    if not results:
        return {
            "status": "no_context", "top_score": 0.0,
            "avg_score": 0.0, "retrieved_docs": 0, "query_coverage": 0.0,
        }
    query_tokens = {
        t for t in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(t) >= 4
    }
    top_scores = [
        float(r.get("similarity_score") or 0.0)
        for r in results if r.get("similarity_score") is not None
    ]
    joined   = " ".join((r.get("content") or "") for r in results[:4]).lower()
    overlap  = sum(1 for tok in query_tokens if tok in joined)
    coverage = (overlap / max(1, len(query_tokens))) if query_tokens else 0.0
    top_score = max(top_scores) if top_scores else 0.0
    avg_score = (sum(top_scores) / len(top_scores)) if top_scores else 0.0

    if top_score >= 0.50 and coverage >= 0.30:
        status = "grounded_context"
    elif top_score >= 0.30 or coverage >= 0.18:
        status = "partial_context"
    else:
        status = "no_context"

    return {
        "status": status,
        "top_score": float(top_score),
        "avg_score": float(avg_score),
        "retrieved_docs": len(results),
        "query_coverage": float(coverage),
    }


# ─── Per-type Prompt Builders ─────────────────────────────────────────────────

def _build_factual_prompt(query: str, context: str) -> str:
    return f"""You are a precise retrieval-augmented assistant.

Answer the question directly and concisely using ONLY the retrieved context.
Cite the source document inline as [Doc X] for every factual claim.
Give a short, specific answer — avoid padding or vague generalisations.
{_extract_length_hint(query)}

Retrieved context:
{context}

Question: {query}

Answer:
"""


def _build_multihop_prompt(query: str, context: str) -> str:
    return f"""You are a multi-step reasoning assistant.

The question requires connecting evidence from multiple documents.
Step 1 — Identify and cite the key fact(s) from each relevant document as [Doc X].
Step 2 — Explicitly link those facts into a coherent reasoning chain.
Step 3 — State a clear conclusion supported by that chain.
Do not introduce claims absent from the retrieved context.
{_extract_length_hint(query)}

Retrieved context:
{context}

Question: {query}

Answer (show reasoning chain):
"""


def _build_analytical_prompt(query: str, context: str) -> str:
    return f"""You are a careful analytical reasoning assistant.

Your answer MUST visibly depend on the context; it should differ from a generic textbook answer.
Structure:
1. One-sentence thesis answering the question directly.
2. Two or more evidence-based paragraphs, each citing sources inline as [Doc X].
3. A brief synthesis or judgement.
Use general knowledge only to connect ideas; do not introduce facts that contradict the context.
{_extract_length_hint(query)}

Retrieved context:
{context}

Question: {query}

Answer:
"""


def _build_comparison_prompt(query: str, context: str) -> str:
    return f"""You are a structured comparison assistant.

Compare the entities or concepts in the question using evidence from the retrieved context.
Organise as:
- Similarities: cite [Doc X] for each point
- Differences:  cite [Doc X] for each point
- Overall judgement

Draw on all relevant documents; use a different document per entity where possible.
Do not invent attributes absent from the context.
{_extract_length_hint(query)}

Retrieved context:
{context}

Question: {query}

Answer:
"""


def _build_partial_prompt(query: str, context: str) -> str:
    return f"""You are a careful assistant answering with mixed evidence.

Use the retrieved context where helpful; use general knowledge to fill reasonable gaps.
Cite retrieved points as [Doc X]; do not cite general background knowledge.
Do not mention missing context, retrieval quality, or system limitations.
{_extract_length_hint(query)}

Retrieved context:
{context}

Question: {query}

Answer:
"""


def _build_no_context_prompt(query: str) -> str:
    return f"""You are a knowledgeable assistant.

Answer the question using your general knowledge only.
Do not mention documents, context, retrieval, or sources.
Do not output citations like [Doc X].
For analytical questions, provide a thesis, evidence-based explanation, and conclusion.
{_extract_length_hint(query)}

Question: {query}

Answer:
"""


def _select_prompt(
    query: str,
    context: str,
    qtype: QueryType,
    context_status: str,
) -> Tuple[str, str, bool]:
    """Returns (prompt_str, answer_mode_used, used_general_knowledge)."""
    if context_status == "no_context":
        return _build_no_context_prompt(query), "general_knowledge_only", True

    if context_status == "partial_context":
        return _build_partial_prompt(query, context), "hybrid_context_plus_knowledge", True

    # grounded_context — dispatch to type-specific prompt
    dispatch = {
        QueryType.FACTUAL:    (_build_factual_prompt,    "grounded_factual",    False),
        QueryType.MULTI_HOP:  (_build_multihop_prompt,   "grounded_multihop",   False),
        QueryType.ANALYTICAL: (_build_analytical_prompt, "grounded_analytical", False),
        QueryType.COMPARISON: (_build_comparison_prompt, "grounded_comparison", False),
        QueryType.AMBIGUOUS:  (_build_partial_prompt,    "hybrid_ambiguous",    True),
    }
    builder, mode, use_gen = dispatch[qtype]
    return builder(query, context), mode, use_gen


# ─── Main RAG Entry Point ─────────────────────────────────────────────────────

def rag_advanced(
    query: str,
    retriever,
    llm,
    top_k: int,
    min_score: float = 0.0,
    return_context: bool = False,
    debug: bool = True,
    classifier: Optional[QueryClassifier] = None,
) -> Dict[str, Any]:
    """
    Adaptive RAG (RARC) entry point.

    Pipeline
    --------
    1. Classify the query → QueryType (factual / multi_hop / analytical / comparison / ambiguous)
    2. Look up the matching RetrievalPolicy (k, pool_size, context_chars, neighbor_window, …)
    3. Retrieve adaptively — ChromaBalancedRAGRetriever.adaptive_retrieve() handles
       iterative expansion for AMBIGUOUS queries automatically
    4. Build a type-specific prompt (factual → concise; multi_hop → chain-of-thought;
       analytical → thesis+evidence; comparison → structured table-like; ambiguous → hybrid)
    5. Return answer + full metadata including query_type
    """
    timings: Dict[str, Any] = {}
    t_total0 = now()

    # ── 1. Classify ──────────────────────────────────────────────────────────
    clf   = classifier if classifier is not None else QueryClassifier()
    qtype = clf.classify(query)
    policy = RETRIEVAL_POLICIES[qtype]

    # Respect caller's explicit top_k as a lower bound so users can always
    # request more results than the policy default via --top_k.
    effective_k = max(policy.k, top_k) if top_k > policy.k else policy.k

    # ── 2. Retrieve ───────────────────────────────────────────────────────────
    t0 = now()
    if hasattr(retriever, "adaptive_retrieve"):
        effective_policy = _copy(policy)
        effective_policy.k = effective_k
        results = retriever.adaptive_retrieve(
            query, policy=effective_policy, score_threshold=min_score
        )
    else:
        results = retriever.retrieve(query, top_k=effective_k, score_threshold=min_score)
    timings["retrieve_s"] = now() - t0

    context_state = _estimate_context_strength(results, query)

    if debug:
        print(f"\n[DEBUG] Query      : {query}")
        print(f"[DEBUG] QueryType  : {qtype.value}  k={effective_k}  "
              f"pool={policy.pool_size}  ctx_chars={policy.context_chars}")
        print(f"[DEBUG] Retrieved  : {len(results)} docs  "
              f"context_status={context_state['status']}")
        for r in results[:min(5, len(results))]:
            md = r.get("metadata") or {}
            print(
                "  score=", r.get("similarity_score"),
                " source=", md.get("source_file", md.get("source", "unknown")),
                " page=", md.get("page", "NA"),
                " neighbor=", r.get("is_neighbor", False),
            )

    # ── 3. Build context string ───────────────────────────────────────────────
    context = _prepare_context(results, max_chars=policy.context_chars) if results else ""

    sources: List[Dict[str, Any]] = []
    for doc in results:
        md      = doc.get("metadata", {}) or {}
        content = doc.get("content", "") or ""
        sources.append({
            "source":      md.get("source_file", md.get("source", "unknown")),
            "page":        md.get("page", "unknown"),
            "chunk_id":    md.get("chunk_id", None),
            "score":       float(doc.get("similarity_score", 0.0) or 0.0),
            "preview":     (content[:300] + "...") if len(content) > 300 else content,
            "is_neighbor": bool(doc.get("is_neighbor", False)),
        })

    # ── 4. Select type-specific prompt ───────────────────────────────────────
    prompt, answer_mode_used, used_general_knowledge = _select_prompt(
        query, context, qtype, context_state["status"]
    )

    # ── 5. Generate answer ───────────────────────────────────────────────────
    llm_out = _call_llm_timed(llm, prompt)
    answer  = (llm_out.get("text") or "").strip()

    timings["llm_s"]          = float(llm_out.get("llm_latency_s") or 0.0)
    timings["output_tokens"]  = llm_out.get("output_tokens")
    timings["tokens_per_sec"] = float(llm_out.get("tokens_per_sec") or 0.0)
    timings["total_s"]        = now() - t_total0

    confidence = max(
        [float(doc.get("similarity_score", 0.0) or 0.0)
         for doc in results if doc.get("similarity_score") is not None] + [0.0]
    )

    out: Dict[str, Any] = {
        "answer":               answer,
        "sources":              sources,
        "confidence":           confidence,
        "timings":              timings,
        "context_status":       context_state["status"],
        "answer_mode_used":     answer_mode_used,
        "used_general_knowledge": used_general_knowledge,
        "retrieved_docs_count": context_state["retrieved_docs"],
        "top_retrieval_score":  context_state["top_score"],
        "avg_retrieval_score":  context_state["avg_score"],
        "query_coverage":       context_state["query_coverage"],
        "query_type":           qtype.value,
    }
    if return_context:
        out["context"] = context
    return out
