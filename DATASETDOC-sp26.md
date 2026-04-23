# DATASETDOC-sp26
## BU School of Law: Legal Chatbot - Dataset Documentation (Spring 2026)

This document provides comprehensive dataset documentation for the BU Spark legal chatbot project. It covers project context, data sources, collection methods, preprocessing, quality controls, intended use, maintenance, and extension guidance for future teams.

---

## Project Information

### What is the project name?
- **BU School of Law: Legal Chatbot**

### What is the link to the project GitHub repository?
- [GitHub repo page](https://github.com/BU-Spark/ml-bu-legal-agent)

### What is the link to the project Google Drive folder?
- [Drive folder](https://drive.google.com/drive/u/1/folders/1Dho_O-Tp5ILZbLZWTNPj4Oaw60HJ8vhN)

### In your own words, what is this project about? What is the goal of this project?
- This project designs and prototypes an AI-powered legal assistance chatbot for people facing eviction and housing-related legal issues in Massachusetts.
- The chatbot uses large language models (LLMs) plus retrieval from curated legal sources to provide accurate, understandable, citation-backed legal information.
- Core goals include:
  - improving access to legal information for users without formal legal training,
  - supporting user understanding of legal process steps and obligations,
  - generating grounded, source-cited responses based on Massachusetts housing law materials.

### Who is the client for the project?
- **BU Law Consumer Economic Justice Clinic**
- The Consumer Economic Justice Clinic is a 12-credit, year-long experiential course where students analyze causes of economic injustice while representing low-income consumers in civil matters.
- [Read more about the client](https://www.bu.edu/law/experiential-learning/clinics/consumer-economic-justice-clinic/)

### Who are the client contacts for the project?
- Jade Brown - jbrown20@bu.edu
- Orestes Rellos - orellos@bu.edu
- John Cloherty - jcloher1@bu.edu
- Russ Wilcox (expert in residence) - russ@artifexai.io

### What class was this project part of?
- **Spark Machine Learning Practicum: CDS DS 594**

---

## Dataset Information

### What datasets were used in this project?
This project uses a blended legal corpus assembled from textbook-derived data and web-scraped legal sources.

- Legal tactics textbook data:
  - [Legal tactics textbook repository folder](https://github.com/BU-Spark/ml-bu-legal-agent/tree/dev/data)
- Web scraping source list:
  - [Web scraping data links document](https://docs.google.com/document/d/1VKD4x6PPNmpP7nMCUztWVns8A_8ouhdFdDawQiDH7QE/edit?tab=t.0)
- Primary web sources currently scraped:
  - `masslegalhelp.org` (Legal Tactics chapters and section-level content)
  - `malegislature.gov` (Massachusetts General Laws sections)

### What keywords/tags describe the dataset?
- Domains of application: NLP, Retrieval-Augmented Generation, Text Understanding, Legal QA, Summarization, Conversational AI.
- Civic domains: Civic Tech, Housing, Education, Legal Access, Consumer Justice.

---

## Dataset Motivation

### For what purpose was the dataset created? Was there a specific task or gap?
- The dataset was created to support an AI legal assistant focused on Massachusetts housing law.
- The main task is grounded legal question answering with citations and role-aware framing (general, tenant, landlord perspectives).
- The dataset addresses a practical gap: legal materials are complex and difficult to navigate for many users. This corpus structures and indexes those materials so the chatbot can return plain-language responses tied to original legal sources.
- The dataset is designed for iterative refresh as statutes and legal guidance evolve over time.

---

## Dataset Composition

### What do the instances represent? What format are they in?
- Primary instance type: **document sections** (legal text segments).
- Data format: **text-based structured records**.
- Typical record fields include:
  - `section_name`
  - `section_url`
  - `section_text`
  - source metadata used during indexing and retrieval

### Are there multiple instance types?
- Yes, content is drawn from different source families:
  - practical legal guidance text (MassLegalHelp Legal Tactics content),
  - statutory law text (Massachusetts General Laws via Malegislature),
  - textbook-derived legal content used in prior and current iterations.

### Are there errors, noise, or redundancies?
- Some redundancy is expected and acceptable:
  - Legal Tactics guidance may overlap with statutory content from Massachusetts law pages.
  - Similar concepts may appear across textbook and website content.
- Web scraping introduces possible noise risks (navigation text, duplicated fragments, broken links), which are handled through cleaner scripts and filtering thresholds.

### Is the dataset self-contained?
- No. The dataset relies on external public websites and textbook materials.
- Ongoing maintenance is required to ensure:
  - source URLs remain valid,
  - scraping selectors remain compatible with source page structure,
  - indexed content is refreshed when legal content is updated.

---

## Collection Process

### What mechanisms or procedures were used to collect the data?
- **Web scraping (current core approach for both major legal sources):**
  - `masslegalhelp.org` scraping pipeline for Legal Tactics chapters and section anchors.
  - `malegislature.gov` scraping pipeline for Massachusetts General Laws chapter/section content.
- **Textbook-derived ingestion:**
  - Legal tactics textbook files and related processing scripts.
- **Embedding/indexing stage (post-collection):**
  - OpenAI embedding API is used to build Chroma vector stores for retrieval.

### How were collection mechanisms validated?
- Manual and practical validation through:
  - reviewing scraped outputs for section completeness and readability,
  - checking that citations link to expected source sections,
  - testing chatbot responses for source relevance and legal coherence,
  - review discussions with client stakeholders and project advisors.

---

## Preprocessing, Cleaning, and Labeling

### Was preprocessing/cleaning/labeling performed?
- Yes. The pipeline includes explicit cleaning and normalization scripts.
- Examples:
  - whitespace and formatting normalization,
  - removal of short/invalid sections,
  - stripping navigation/junk page text where applicable,
  - preserving section URLs for citation traceability,
  - converting cleaned records into document objects for embedding.

### Was role-related tagging applied?
- Role-aware behavior is supported in retrieval and prompting (`tenant`, `landlord`, `general`).
- Historical textbook preprocessing includes keyword-based role tagging.
- Prior role keyword examples include:
  - Tenant: `tenant rights`, `rent control`, `eviction protections`, `lease termination`
  - Landlord: `landlord duties`, `property maintenance`, `rent collection`, `eviction process`
- Current and future teams should review and refine these taxonomies to improve retrieval precision and fairness.

### Is preprocessing code available?
- Yes. Relevant scraping/cleaning/indexing scripts are in the repository, including:
  - `masslegalhelp_legal_tactics_scraper.py`
  - `masslegalhelp_legal_tactics_cleaner.py`
  - `masslegalhelp_legal_tactics_to_chroma.py`
  - `malegislature_primary_law_extraction.py`
  - `malegislature_final_scrapper.py`
  - `malegislature_final_cleaner.py`
  - `malegislatur_data_to_chroma.py`
  - textbook/pdf processing scripts in repository data-processing paths

---

## Uses

### What tasks has this dataset been used for?
- Retrieval-Augmented legal question answering.
- Role-sensitive response generation for housing-law users.
- Citation-backed conversational legal guidance.
- Testing and evaluation of legal relevance, clarity, and response grounding.

### How does the chatbot use this dataset at runtime?
- User query is embedded and matched by similarity against Chroma vector stores.
- Top relevant legal chunks are retrieved, reranked, and passed to the LLM prompt context.
- The LLM produces a response constrained by retrieved legal content, with source citations.

### What might impact future uses?
- Legal updates require periodic rescraping and re-indexing.
- Source page structure changes may break scraper selectors.
- Role-tag vocabularies and retrieval thresholds may need iterative tuning.
- Dataset extension to additional legal domains may require schema and evaluation updates.

---

## Distribution

### What access type should this dataset have?
- **External open access**, based on client discussions and current source licensing assumptions.
- Teams should still verify source terms of use and attribution requirements for each upstream source during future releases.

---

## Maintenance and Extension

### If others want to extend or contribute, how can they do so?
- Follow the existing ingestion pipelines and data contracts:
  1. Update source URLs and scraper logic as needed.
  2. Run scraping and cleaner scripts.
  3. Rebuild both vector stores.
  4. Validate output quality with representative legal test prompts.

### Recommended maintenance cadence
- Refresh web-scraped corpora periodically (e.g., monthly/quarterly) or immediately after major legal updates.
- Re-run the full ingestion pipeline whenever:
  - source structure changes,
  - major prompt/retrieval behavior changes,
  - textbook/source updates are added.

### Known extension opportunities
- Replace older textbook content with newer legal tactics editions when available.
- Expand source coverage to additional trusted Massachusetts legal resources.
- Improve role tagging with more robust taxonomy and evaluation.
- Add automated data-quality checks (schema checks, URL validity checks, duplicate detection, and regression test prompts).

---

## Risks, Assumptions, and Compliance Notes

- This dataset supports legal information, not legal advice.
- External source availability is assumed; broken links are an expected maintenance risk.
- Changes in website HTML structure can degrade scraper performance and require updates.
- Teams should avoid exposing sensitive user documents if adding user-upload corpora in future workflows.

---

## Summary

The BU School of Law Legal Chatbot dataset is a continuously maintained legal-text corpus built from:
- web-scraped `masslegalhelp.org` content,
- web-scraped `malegislature.gov` statutory content,
- and textbook-derived legal tactics materials.

Its core value is enabling grounded, citation-backed, role-aware housing-law assistance. The dataset is intentionally designed for regular refresh and extension as legal content evolves.
