# Enterprise AI Agent

An enterprise-ready AI orchestration platform extending the base RAG pipeline into a multi-step research agent. The platform supports complex query planning, semantic caching, conflict auditing, multilingual translations, and analytics tracking.

---

## 1. System Architecture

```text
                    USER
                      │
                      ▼
             Enterprise AI Agent UI
                  (Streamlit)
                      │
                      ▼
            Language Detection / Router
                      │
                      ▼
              Semantic Cache Check
                 │           │
              CACHE HIT    CACHE MISS
                 │           │
                 │           ▼
                 │     Research Planner
                 │           │
                 │           ▼
                 │     Sub-question
                 │      Decomposition
                 │           │
                 │           ▼
                 │     Multi-query Retrieval
                 │           │
                 │     ┌─────┴─────┐
                 │     ▼           ▼
                 │  ChromaDB      BM25
                 │  Dense         Sparse
                 │     │           │
                 │     └─────┬─────┘
                 │           ▼
                 │       RRF Fusion
                 │           │
                 │           ▼
                 │     Cross-Encoder
                 │       Reranking
                 │           │
                 │           ▼
                 │    Evidence Validation
                 │           │
                 │     ┌─────┴─────┐
                 │     ▼           ▼
                 │  Conflicts    Information
                 │  Detection      Gaps
                 │     │           │
                 │     └─────┬─────┘
                 │           ▼
                 │      LLM Synthesis
                 │           │
                 └────────────┤
                              ▼
                    Enterprise Response
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
             Findings      Evidence       Sources
                           Matrix
                              │
                              ▼
                        Telemetry
```

---

## 2. Core Capabilities

*   **Sub-query Decomposition**: Breaks down complex business queries into 2-3 focused sub-queries for broader document-range coverage.
*   **Semantic Caching**: Checks cosine-similarity distance against query embeddings using a `0.92` threshold. Prefers a standard Redis client, falling back gracefully to an in-memory RAM dictionary if Redis is down.
*   **Multilingual Support**: Prompt-engineered routing detecting and translating inputs/outputs between **English**, **Tamil**, and **Hindi**.
*   **Conflict & Gap Audits**: Explicitly scans metadata variables (date, version, category, priority) to flag metric discrepancies and point out information gaps.
*   **Telemetry tracking**: Logs execution latencies, sub-queries, cache states, and user feedback ratings into `evaluation_log.json`.

---

## 3. Getting Started

### Prerequisites

*   Python 3.10+
*   Local Redis instance (optional, fallback in place)
*   Groq API Key

### Configuration

Setup your credentials inside a local `.env` file:
```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here
REDIS_URL=redis://localhost:6379
```

### Installation

```bash
# Create python virtual environment
python -m venv venv
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

### Running Tests
Execute the integration test suite covering document parses, multilingual routing, semantic cache fallbacks, and agent plans:
```bash
python scripts/run_integration_tests.py
```

### Running the UI
Launch the Streamlit workspace:
```bash
python -m streamlit run app.py
```

---

## 4. Scalability Roadmap: 100 to 100,000+ Records

To scale the current workspace from local storage to high-throughput enterprise workloads, replace the local structures with distributed systems:

1.  **Distributed File Parsing & Ingestion**:
    *   Replace local file transfers with **Amazon S3** or google-cloud storage.
    *   Decompress document tasks into asynchronous backend scripts using **Celery** workers.
2.  **Cluster Vector & Sparse Searches**:
    *   Transition Chroma DB to a managed search cluster (e.g. pgvector, Qdrant or Pinecone).
    *   Offload BM25 calculations to a scalable indexer search engine (e.g., Elasticsearch or OpenSearch) to handle queries concurrently.
3.  **Active Cache Cluster**:
    *   Deploy Redis in highly-available replication modes (AWS ElastiCache or Redis Enterprise) with metadata partitions.
4.  **Multi-Tenancy Isolation**:
    *   Filter Chroma metadata tags using strict customer IDs (`tenant_id = ID`) to guarantee document security.
