# Mortimer Agent

A Python RAG (Retrieval-Augmented Generation) assistant that ingests technical PDF documents and answers questions with structured, cited JSON responses.

## Architecture

```
PDF Loader (PyMuPDF)
    |
    v
Text Chunker (LangChain RecursiveCharacterTextSplitter)
    |
    v
Embedder (sentence-transformers / all-MiniLM-L6-v2)  -->  Vector Store (ChromaDB)
                                                                   |
                                              User Query  -->  Retrieval (top-k)
                                                                   |
                                                           Prompt Builder
                                                                   |
                                                           LLM (OpenAI gpt-4o-mini)
                                                                   |
                                                           RAGResponse (Pydantic)
```

**Pipeline steps:**
1. PDFs are downloaded and extracted page-by-page via PyMuPDF
2. Text is split into overlapping chunks with section metadata preserved
3. Chunks are embedded locally and stored in a persistent ChromaDB vector store
4. A user query is embedded, and the top-k most relevant chunks are retrieved
5. Retrieved context and the question are formatted into a prompt
6. The LLM generates an answer and returns it as a validated Pydantic model
7. The response is serialized as JSON

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for package management
- An OpenAI API key

### Installation

```bash
git clone github.com/johntrecker/mortimer-agent
cd mortimer-agent
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
uv sync
```

### Ingest Documents

Ingest the default three arxiv papers (Attention is All You Need, GPT-3, GPT-4):

```bash
uv run mortimer ingest
```

Or ingest custom PDF URLs:

```bash
uv run mortimer ingest https://arxiv.org/pdf/2512.19466 https://arxiv.org/pdf/2502.04426
```

### Ask Questions

```bash
uv run mortimer ask "What attention mechanism does the Transformer architecture use?"
```

### Reset the Vector Store

```bash
uv run mortimer reset
```

## Example Queries and Outputs

### Query 1

```bash
uv run mortimer ask "What is the main contribution of the Transformer architecture?"
```

```json
{
  "question": "What is the main contribution of the Transformer architecture?",
  "answer": "The Transformer architecture's main contribution is replacing recurrent and convolutional layers entirely with a self-attention mechanism. This allows for significantly more parallelization during training and achieves superior translation quality. The model relies solely on attention to compute representations of input and output, without using sequence-aligned RNNs or convolutions.",
  "sources": [
    "[1] (Source: 1706.03762.pdf, p.1, Section: Abstract)",
    "[2] (Source: 1706.03762.pdf, p.2, Section: Introduction)"
  ]
}
```

### Query 2

```bash
uv run mortimer ask "How many parameters does GPT-3 have?"
```

```json
{
  "question": "How many parameters does GPT-3 have?",
  "answer": "GPT-3 has 175 billion parameters. The paper presents autoregressive language models at this scale and larger, demonstrating strong few-shot performance across many NLP tasks.",
  "sources": [
    "[1] (Source: 2005.14165.pdf, p.1, Section: Abstract)",
    "[3] (Source: 2005.14165.pdf, p.4, Section: 2. Approach)"
  ]
}
```

## Configuration

Copy `.env.example` to `.env` and configure:

```env
OPENAI_API_KEY=sk-your-key-here

# Optional overrides
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=data/chroma_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_TOP_K=5
PDF_DOWNLOAD_DIR=data/pdfs
```

## Project Structure

```
mortimer-agent/
├── src/mortimer/
│   ├── cli.py                  # Click CLI (ingest, ask, reset)
│   ├── config.py               # pydantic-settings configuration
│   ├── models/
│   │   └── schemas.py          # Pydantic v2 data models (all frozen)
│   ├── ingestion/
│   │   ├── loader.py           # PDF download (HTTPS-only) + PyMuPDF extraction
│   │   └── chunker.py          # RecursiveCharacterTextSplitter + metadata
│   ├── retrieval/
│   │   ├── embedder.py         # sentence-transformers wrapper
│   │   └── vector_store.py     # ChromaDB persistence + cosine similarity search
│   ├── generation/
│   │   ├── prompt.py           # Prompt template builder
│   │   └── llm_client.py       # OpenAI JSON-mode client -> RAGResponse
│   └── pipeline/
│       └── rag.py              # Orchestrates ingest + query flows
├── tests/
│   ├── unit/                   # Mocked unit tests per module
│   └── integration/            # End-to-end pipeline tests (mocked LLM)
├── prompts/
│   ├── rag_system.txt          # System prompt
│   └── rag_user.txt            # User prompt template
├── data/
│   ├── pdfs/                   # Downloaded PDFs (gitignored)
│   └── chroma_db/              # ChromaDB persistence (gitignored)
├── pyproject.toml
└── .env.example
```

## Development

### Run Tests

```bash
uv run pytest
uv run pytest --cov --cov-report=term   # with coverage
uv run pytest tests/unit/               # unit tests only
uv run pytest tests/integration/        # integration tests only
```

### Linting

```bash
uv run ruff check src/ tests/
```

## Design Decisions

### PDF Parsing: PyMuPDF
PyMuPDF (`fitz`) offers the best text extraction speed and quality for academic papers among Python PDF libraries. It handles multi-page documents reliably and exposes per-page access which is used to preserve page number metadata on each chunk.

### Chunking: LangChain RecursiveCharacterTextSplitter
Only the text splitter utility from LangChain is used, not the full chain/agent framework. This avoids LangChain lock-in while using its battle-tested chunking logic. The recursive splitter respects natural text boundaries (paragraphs, sentences) before falling back to character splits.

### Embeddings: sentence-transformers (local)
`all-MiniLM-L6-v2` runs entirely locally. This eliminates per-embedding API costs, works offline after the initial model download (~90 MB), and has excellent quality/speed for semantic search over technical text. The tradeoff is a slightly longer first-run initialization time.

### Vector Store: ChromaDB
ChromaDB provides persistent local storage, built-in cosine similarity search, and metadata filtering with a clean Python API. Ingestion is idempotent: `has_document()` checks whether a source is already indexed before re-processing, and `add_chunks()` uses upsert semantics.

### LLM: OpenAI gpt-4o-mini (JSON mode)
`gpt-4o-mini` is used via OpenAI's `response_format={"type": "json_object"}` mode, ensuring the model returns valid JSON. The response is then validated through the `RAGResponse` Pydantic model, providing a hard schema guarantee on the output structure.

### Immutability
All Pydantic models use `ConfigDict(frozen=True)`. Data flows through the pipeline as new immutable objects rather than mutated state, which makes the pipeline straightforward to test and reason about.

### Security
- PDF downloads are restricted to HTTPS URLs only (no HTTP, no `file://`)
- Downloads are capped at 50 MB with a 30-second timeout
- Local file paths are restricted to the configured `PDF_DOWNLOAD_DIR`
- Question inputs are validated (non-empty, max 2000 characters)
- API key is stored as `pydantic.SecretStr` and never logged

## Future Features Roadmap

### Phase A: REST API (FastAPI)
- `POST /ingest` and `POST /query` endpoints
- Async pipeline conversion
- OpenAPI documentation

### Phase B: Conversation Memory
- Per-session conversation history appended to prompt context
- Sliding window to stay within token limits

### Phase C: Retrieval Quality Evaluation
- RAGAS framework integration (faithfulness, answer relevance, context precision)
- `mortimer eval` CLI command with a golden question set

### Phase D: Reranking and Prompt Engineering
- Cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`) of top-k results
- Prompt versioning and A/B testing

### Phase E: Query Caching
- Hash-based exact match cache for repeated queries
- Semantic similarity cache with TTL expiration
- Optional Redis backend

## License

MIT
