# 🛡️ Production Local Corrective RAG (CRAG) with Self-Correction & Fallback

An enterprise-grade, privacy-first **Corrective Retrieval-Augmented Generation (CRAG)** system built completely on local open-weight models (`Llama 3.1 8B` & `nomic-embed-text`) orchestrated via **LangGraph**, **ChromaDB**, and **BM25 Hybrid Search**.

Unlike naive RAG pipelines that silently hallucinate when retrieved context is irrelevant or noisy, this system inserts an automated document evaluation and web fallback layer before generation, alongside an automated hallucination check after synthesis.

---

## 🏗️ System Architecture
text
[ User Query ]
│
▼
[ Hybrid Retrieval Node ]
┌───────────┴───────────┐
▼                       ▼
Dense (Chroma)      Sparse (BM25)
└───────────┬───────────┘
▼
[ Reciprocal Rank Fusion ]
│
▼
[ Document Grader Node ] ── (Evaluates Relevance)
│
┌───────┴───────────────────┐
│ (Irrelevant)              │ (Relevant)
▼                           │
[ Web Search Node (DDG) ]         │
│                           │
└───────────┬───────────────┘
▼
[ Generation Node ]
│
▼
[ Hallucination Grader ]
│
┌───────────┴───────────┐
│ (Grounded)            │ (Hallucinated)
▼                       ▼
[ Output ]          [ Loop / Regenerate ]

---

## 🔑 Key Engineering Highlights

- **100% Local Inference:** Powered entirely by Ollama (`llama3.1:8b` and `nomic-embed-text`), eliminating third-party API costs and preserving data privacy.
- **Hybrid Retrieval (Dense + Sparse):** Overcomes dense-only semantic blindspots by combining vector similarity search (ChromaDB) with lexical keyword matching (BM25) fused via **Reciprocal Rank Fusion (RRF)** ($k=60$).
- **LangGraph State Machine:** Built using cyclical state graph routing rather than brittle sequential chains, enabling resilient failure recovery.
- **Automated Fallback:** When internal document grades fall below threshold, queries dynamically fall back to external web search (`DuckDuckGo`).
- **Self-Correction & Hallucination Guardrails:** Implements an automated verification pass post-generation to ensure outputs are strictly grounded in retrieved evidence.

---

## 🚀 Quickstart

### 1. Prerequisites
- [Ollama](https://ollama.com/) installed
- Python 3.12+

### 2. Pull Local Models
bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b

### 3. Installation
bash
git clone https://github.com/indhuprasanna/crag-local-system.git
cd crag-local-system
python -m venv .venv
On Windows:
..venv\Scripts\Activate.ps1

On macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

### 4. Run the Streamlit Interface
bash
streamlit run app.py

