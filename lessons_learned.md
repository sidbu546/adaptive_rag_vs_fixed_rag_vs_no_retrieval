# Lessons Learned & Future Scope

This document records the reality of building **Star — the Massachusetts Housing Law Assistant**: the experiments that failed, the lessons we drew from them, and the directions we recommend for the next cohort.

The project as it stands is a Gradio application (`legal_app.py`) backed by two Chroma vector stores — Legal Tactics chapters scraped from masslegalhelp.org and the Massachusetts General Laws scraped from malegislature.gov — combined with OpenAI embeddings, a cross-encoder reranker, sentence-level passage compression, and role-aware prompting for tenants, landlords, and general users.

---

## Table of Contents

- [1. What Didn't Work](#1-what-didnt-work)
- [2. Lessons Learned](#2-lessons-learned)
- [3. Future Recommendations](#3-future-recommendations)

---

## 1. What Didn't Work

The following are the dead ends, failed experiments, and technical hurdles encountered during development. They are documented here so that subsequent cohorts do not repeat them.

### 1.1 PDF-based ingestion of the Legal Tactics handbook

The first ingestion pipeline parsed the Legal Tactics PDFs directly (`Legal-Tactics-Book.zip`). Vestiges of this approach remain in `config.py` (`DATA_DIR`, `TEMP_PDF_DIR`) and in the hard-coded `LEGAL_TACTICS_URLS` map in `scraping_load.py`.

The approach failed for three reasons:

- PDF text extraction produced inconsistent chunks; headers, footers, and page numbers leaked into chunk bodies, and section boundaries were lost.
- Citations could only point to an entire PDF, not to a specific section. A question about security deposits would return a sixty-page document as its "source."
- Re-embedding required re-downloading and re-parsing the entire archive.

We migrated to chapter-level scraping with anchor URLs (`masslegalhelp_legal_tactics_scraper.py` → `…cleaner.py` → `…_to_chroma.py`). Each section now stores a `section_url` ending in `#anchor-id`, allowing citations to deep-link to the exact heading on masslegalhelp.org. The PDF code paths and the URL mapping remain in the repository as a fallback but are not used by the live application.

### 1.2 Local language model via Ollama

We implemented an Ollama backend (`ollama_llm.py`) so the chatbot could operate without an OpenAI API key. It is wired through the `LLM` interface and selectable by setting `DEFAULT_LLM = "ollama"` in `config.py`.

The approach failed for the following reasons:

- `deepseek-r1:1.5b` was the only model small enough to run on a typical laptop, and it could not reliably follow the structured output template. Sections were skipped, merged, or hallucinated.
- The model frequently fabricated MGL chapter and section numbers that did not exist in the retrieved context.
- Larger Ollama models (7B and above) followed the format more reliably but were too slow for an interactive chat experience on the available hardware.

The Ollama path is preserved as scaffolding, but `DEFAULT_LLM = "openai"` (gpt-4o-mini) is the only configuration we trust for end users.

### 1.3 The staging / query-rewriting stage

`prompt_engineering_user.py` contains `STAGING_PROMPT_TEMPLATE`, originally intended to (a) reject off-topic questions and (b) rewrite vague user input into a single clean question prior to retrieval.

The stage failed in practice:

- The off-topic filter rejected too aggressively. Legitimate housing questions phrased casually were sometimes flagged as out of scope.
- The rewrite step occasionally dropped key facts — timeframes, dollar amounts, locations — which then degraded retrieval.
- It introduced an additional LLM call per query, roughly doubling latency, with no measurable improvement in answer quality.

We now bypass this stage explicitly in `vector_store.py` (`# Stage 1 — skip gating, use original query as-is`). The off-topic guardrail was relocated into the main system prompt. The staging template remains in the file but is dead code in the current pipeline.

### 1.4 Strict role-based filtering at retrieval

We initially implemented hard role filtering: when a user identified as a tenant, only chunks tagged `role: tenant` were eligible for retrieval.

The approach failed because:

- Many chunks are genuinely role-neutral (definitions, statutory text, court procedure) and were being filtered out.
- For some questions there were not enough role-tagged chunks to fill the context window. Recall collapsed and answer quality fell noticeably.

We replaced strict filtering with **role-balanced retrieval** (`combined_similarity_search_with_scores` in `scraping_load.py`): role-matching chunks receive priority slots, while role-neutral chunks fill the remainder. The `MIN_LT_RESULTS` and `MIN_LAW_RESULTS` parameters in `config.py` are set to `0` because every stricter value we tested degraded answer quality on at least one evaluation question.

### 1.5 Naive top-k similarity search

The first version of the retrieval layer used plain top-k cosine similarity from a single Chroma store. Two failure modes dominated:

- **Redundancy.** The top four results were frequently near-duplicate paragraphs from the same section, leaving the language model with one fact repeated four times.
- **Source imbalance.** The Legal Tactics text is more verbose and embeds "closer" to natural-language questions than the terse statutory text of the MGL. As a result, MGL passages were almost never retrieved, and answers lacked statutory citations.

We replaced this with: (a) **MMR** (`max_marginal_relevance_search`, `fetch_k = 24` → `k = 10`) per source for diversity; (b) **two parallel retrievals** (one per Chroma store) merged and deduplicated; and (c) a **cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over the union, with a final cut to `FINAL_CONTEXT_K = 4`.

### 1.6 Long, uncompressed chunks diluting the prompt

With `CHUNK_SIZE = 700`, retrieved chunks frequently contained only one or two genuinely relevant sentences. Passing the full chunks to the LLM was wasteful in tokens and diluted attention.

We added a lightweight per-chunk sentence compressor (`compress_doc_for_query` in `vector_store.py`): it scores each sentence by token overlap with the query, retains the top two, and includes their immediate neighbours for context. Chunks shorter than 500 characters are passed through unchanged. We deliberately chose a heuristic over an LLM-based compressor; the latter was both slower and not measurably better.

### 1.7 Markdown citations not rendering as clickable links

The language model produces Markdown citations of the form `[1] [Chapter 3: Security Deposits](https://…)`. By default Gradio's Markdown component sanitises HTML, which broke `target="_blank"` and in some cases the links themselves.

The resolution is in `legal_app.py`: `make_links_clickable` rewrites the Markdown link syntax into raw `<a target="_blank">` tags before rendering, and the Markdown component is configured with `sanitize_html=False`. Both elements are required; removing either breaks clickable sources.

### 1.8 Selenium scraping fragility

Both scrapers (`masslegalhelp_legal_tactics_scraper.py` and `malegislature_final_scrapper.py`) drive headless Chrome through Selenium. This was the single largest source of "works on my machine" issues:

- ChromeDriver and Chrome version mismatches.
- masslegalhelp.org pages where some chapters expose `h2[id]` anchors and others do not, requiring a full-page fallback.
- Polite-delay tuning: too short and the DOM was not fully rendered when read; too long and a complete re-scrape took thirty minutes or more.

A pure `requests` + BeautifulSoup scraper would be faster and more portable, but parts of the source sites render content via JavaScript, which is why Selenium was adopted.

### 1.9 No automated evaluation

We did not build a formal evaluation harness. Quality was judged by hand-running roughly twenty questions and reading the outputs. As a result, many of the design decisions above rest on intuition rather than measurement, and this section is shorter than it should be.

---

## 2. Lessons Learned

If we were to begin the project again with the benefit of hindsight, we would do the following differently.

1. **Build the evaluation harness first, not last.** A 50-question gold set with expected sources, run automatically on every change, would have saved more time than any retrieval tweak we made. Most of the dead ends in §1 would have been killed in a day rather than a week.

2. **Scrape rather than ingest PDFs.** For any source that publishes an HTML version with stable anchors, scrape the HTML. Per-section URLs are dramatically more valuable to the user than per-document URLs, and HTML extraction is markedly cleaner than PDF parsing.

3. **Pick one language model and commit.** The `LLM` abstraction (`llm_interface.py`, `openai_llm.py`, `ollama_llm.py`) was over-engineered for a project that, in practice, only ever shipped on OpenAI. We would defer that abstraction until a second backend genuinely works.

4. **Keep prompts simple until evidence demands otherwise.** Our prompt grew section by section as new behaviours were added. By the end it ran to roughly seventy lines and small models could not follow it. Begin with the smallest prompt that produces the required structure, and add sections only when an evaluation shows a need.

5. **Treat role-awareness as a prompting concern, not a retrieval concern.** Filtering on role at retrieval time hurt recall in our tests. The role's primary job is to shape *how* the answer is written; retrieval should pull the best legal passages regardless of role, and the LLM should frame them for the audience.

6. **Cross-encoder reranking is worth the latency.** Of all the retrieval changes we made, adding the reranker produced the largest single improvement in answer quality. We would add it first next time, not last.

7. **MMR plus per-source retrieval beats a single combined store.** Keeping Legal Tactics and the MGL as separate Chroma stores, retrieving from each with MMR, and merging through a reranker yielded substantially more balanced citations than any single combined store ever did.

8. **Hard-code less; configure more.** The `LEGAL_TACTICS_URLS` filename-to-URL dictionary in `scraping_load.py` is a relic of the PDF era and is brittle. Anything that maps "this source → this canonical URL" should live in document metadata at ingestion time, not in application code.

9. **Be honest about dead code.** The staging prompt, the PDF path, and the Ollama backend remain in the repository even though none are used by `legal_app.py`. Future maintainers will assume they work. Either delete dead code or annotate it clearly as unused.

10. **Treat the disclaimer as load-bearing.** The current prompt forces the disclaimer at the end of every answer, and that is the right call. Resist any future change that makes the disclaimer optional, dynamic, or "smart"; it is a legal and safety feature, not a UX feature.

---

## 3. Future Recommendations

The following are concrete next steps for the next cohort, organised by priority.

### 3.1 Foundational (the project is not really finished without these)

- **Automated evaluation suite.** A YAML or JSON file of 50 to 100 questions, each annotated with expected source URLs and key facts that must appear in the answer. A script that runs the full pipeline on every question and reports retrieval@k, citation accuracy, and (cheaply) an LLM-judge score for answer quality. Wire it into CI.
- **Conversation memory.** At present every question is independent; `legal_app.py` does not pass prior turns to the LLM. Add multi-turn history so users can ask follow-ups such as "what about for a month-to-month tenancy?" without restating context. The follow-up buttons currently re-submit the question text but the model has no memory of the previous turn.
- **Source freshness tracking.** Store the scrape date in each document's metadata, surface that date alongside each citation, and add a `refresh_sources.py` script that re-runs both scrapers and rebuilds both Chroma stores. Massachusetts law changes; stale answers are worse than no answer.

### 3.2 High-value features

- **Spanish and Portuguese language support.** Massachusetts has substantial Spanish- and Portuguese-speaking renter populations. Detect the input language, generate the answer in the same language, and keep the legal citations in English. masslegalhelp.org already has some translated content; scrape what exists.
- **Authenticated user sessions and saved answers.** Allow users to sign in, save answers to a personal "case folder," and reopen them later. Particularly useful for tenants assembling documentation before a hearing.
- **Referral handoff to legal aid.** When a question crosses from "general information" into "you need a lawyer" territory — eviction notices, court dates, lead poisoning of a child — surface a clear referral panel: Greater Boston Legal Services, MetroWest Legal Services, MassLegalHelp's lawyer-finder, and similar resources.
- **Document upload and grounded Q&A.** Allow a user to upload their lease, eviction notice, or 14-day notice, and ask questions grounded against that specific document in addition to the corpus. This is the single most frequently requested feature in user interviews.

### 3.3 Quality-of-life improvements

- **Streaming responses.** The structured Markdown output is long. Streaming it section by section would substantially improve perceived latency. Gradio supports this natively.
- **Structured follow-up extraction.** Follow-ups are currently parsed from the LLM's output by regex, which is fragile when the model uses smart quotes or skips the section. Move follow-up generation to a structured tool/JSON output.
- **Statute-level citations.** When an answer references "MGL Chapter 186, Section 15B," the citation should deep-link to the exact section URL on malegislature.gov. The MGL scraper already captures `section_url`; the prompt simply needs to be taught to surface it inline.
- **Hybrid retrieval (BM25 + dense).** Cross-encoder reranking already helps, but a BM25 first stage would catch keyword-heavy queries (such as statute numbers) that dense embeddings handle poorly.
- **Larger reranker.** Replace `cross-encoder/ms-marco-MiniLM-L-6-v2` with a stronger reranker (e.g. `bge-reranker-large`) once evaluation numbers justify the additional latency.
- **Telemetry.** Anonymous logs of: question, retrieved sources, whether the user clicked a follow-up, whether the user clicked a citation. Without this we are flying blind on what users actually find useful.
- **Containerisation and deployment.** Currently `legal_app.py` launches Gradio on `0.0.0.0:7861` from a developer's laptop. Add a Dockerfile, a `docker-compose.yml` that pre-builds the Chroma stores on first run, and a deployment guide for Hugging Face Spaces or a small VPS.

### 3.4 Longer-term research directions

- **Fine-tune a small open-weights model on the corpus.** Once the evaluation suite exists, distil GPT-4o-mini's behaviour into a Llama-3.x-8B or Mistral-7B fine-tune. If quality holds, the project becomes runnable without an OpenAI key — the original goal of the Ollama backend, finally achieved properly.
- **Expand beyond housing.** The same architecture (scrape → chunk → dual-store → MMR → rerank → role-aware prompt) generalises naturally to consumer protection, family law, and immigration. The infrastructure is general; only the corpus and the prompt need to change.

---

*Last updated by the cohort that built the dual-store, reranked, role-balanced retrieval pipeline now in this repository. If anything in this document contradicts the code, trust the code — and update this document.*