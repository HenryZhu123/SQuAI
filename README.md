## SQuAI: Scientific Question-Answering with Multi-Agent Retrieval-Augmented Generation

SQuAI is a scalable and trustworthy **multi-agent Retrieval-Augmented Generation (RAG)** system for scientific question answering (QA). It is designed to address the challenges of answering complex, open-domain scientific queries with high relevance, verifiability, and transparency. This project is introduced in our CIKM 2025 demo paper:

Link to: [Demo Video](https://www.youtube.com/watch?v=aGDrtsiZDQA&feature=youtu.be)

### Requirements

- Python 3.8+
- PyTorch 2.0.0+
- CUDA-compatible GPU (recommended for embedding / FAISS workloads)

### Installation

0. Load Module for Swig (HPC / cluster setups only; skip on a normal Linux machine)

```bash
ml release/24.04 GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2
```

1. Install libleveldb-dev (needed for LevelDB-backed full-text storage when not using PostgreSQL)

```bash
sudo apt-get install libleveldb-dev
```

2. Clone the repository:

```bash
git clone git@github.com:faerber-lab/SQuAI.git
cd SQuAI
```

3. Create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

### Dataset and index locations (`config.py`)

Runtime paths are defined in `config.py`. The **data root** (`MAIN_DATA_DIR`) is resolved as follows:

1. If the environment variable **`SQUAI_DATA_DIR`** is set to a non-empty string, that path is used (absolute path recommended).
2. Otherwise, the default is **`/home/ubuntu/SQuAI/squai_data`** (change in `config.py` if your layout differs).

Under `MAIN_DATA_DIR`, the code expects:

| Path | Role |
|------|------|
| `faiss_index/` | Dense (E5) FAISS index |
| `bm25_retriever/` | BM25 index |
| `full_text_db` | Full-text store (LevelDB / SQLite-compatible layout; see `sqlite_compat.py`) |
| `{MAIN_DATA_DIR}_extended_data` | `DATA_DIR` for extended assets (see `config.py`) |

**Optional — HPC workspace discovery:** Some scripts use `get_paths.get_main_data_dir()`, which may resolve a path via `ws_list` or fallback paths under `/projects/...` or `/data/horse/ws/...`. If you use the standard `config.py` entry points (e.g. `main.py`), set **`SQUAI_DATA_DIR`** so the data root is explicit and portable.

### PostgreSQL (optional full-text KV)

If the full-text backend should use **PostgreSQL** instead of only the local `full_text_db` tree, configure the DSN in `config.py` and/or via environment variables (recommended for secrets):

| Variable | Purpose |
|----------|---------|
| **`SQUAI_FULLTEXT_PG_DSN`** | PostgreSQL connection string, e.g. `postgresql://user:pass@localhost:5432/squai`. Overrides `FULLTEXT_PG_DSN_DEFAULT` in `config.py`. |
| **`SQUAI_FULLTEXT_PG_TABLE`** | Table name for key/value full-text rows (default: `squai_fulltext_kv`). Overrides `FULLTEXT_PG_TABLE_DEFAULT` in `config.py`. |

When `FULLTEXT_PG_DSN` is non-empty (after resolving env + defaults), `sqlite_compat.open_db()` and related code use PostgreSQL-backed storage; see `postgres_kv.py` and `scripts/build_full_text_db.py` for ingestion examples. **`psycopg2`** is required (see `requirements.txt`).

### Paperclip-backed Agent2 (optional)

By default, Agent2 uses the existing local retriever output (abstract-focused flow). You can enable an optional path that uses the local `paperclip` CLI for Agent2 candidate papers and chunk retrieval.

- Requires `paperclip` to be installed and available in `PATH`.
- When enabled, Agent2 candidates come from `paperclip search`, paper content from `paperclip cat /papers/<id>/content.lines`, and chunk selection uses hybrid sparse+dense scoring.
- Default behavior is unchanged unless you explicitly enable it.

Environment variables:

- `SQUAI_PAPERCLIP_AGENT2=1` enable Paperclip path for Agent2.
- `SQUAI_PAPERCLIP_SEARCH_TOP_K` number of papers from `paperclip search` (default `5`).
- `SQUAI_PAPERCLIP_CHUNK_SIZE` chunk size in chars (default `1400`).
- `SQUAI_PAPERCLIP_CHUNK_OVERLAP` chunk overlap in chars (default `250`).
- `SQUAI_PAPERCLIP_AGENT2_TOP_CHUNKS` top chunks per paper for Agent2 prompt (default `3`).
- `SQUAI_PAPERCLIP_SEARCH_TIMEOUT_SEC` / `SQUAI_PAPERCLIP_CAT_TIMEOUT_SEC` CLI timeouts.

Example:

```bash
SQUAI_PAPERCLIP_AGENT2=1 SQUAI_PAPERCLIP_SEARCH_TOP_K=5 python run_SQuAI.py --single_question "How is CRISPR delivered in vivo?"
```

### Running SQuAI

SQuAI can be run on a single question or a batch of questions from a JSON/JSONL file.

#### Process a Single Question

```bash
python run_SQuAI.py --model tiiuae/Falcon3-10B-Instruct --n 0.5 --alpha 0.65 --top_k 20 --single_question "Your question here?"
```

#### Process Questions from a Dataset

```bash
python run_SQuAI.py --model tiiuae/Falcon3-10B-Instruct --n 0.5 --alpha 0.65 --top_k 20 --data_file your_questions.jsonl --output_format jsonl
```

#### Parameters

- `--model`: Model name or path (default: "tiiuae/falcon-3-10b-instruct")
- `--n`: Adjustment factor for adaptive judge bar (default: 0.5)
- `--alpha`: Weight for semantic search vs. keyword search (0-1, default: 0.65)
- `--top_k`: Number of documents to retrieve (default: 20)
- `--data_file`: File containing questions in JSON or JSONL format
- `--single_question`: Process a single question instead of a dataset
- `--output_format`: Output format - json, jsonl, or debug (default: jsonl)
- `--output_dir`: Directory to save results (default: "results")

#### HTTP API (FastAPI)

The service in `main.py` exposes `POST /split` and `POST /ask`. Example:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Use environment variables such as **`SQUAI_LLM_MODEL`**, **`DEEPSEEK_API_KEY`** / **`FALCON_API_KEY`** (see `key_resolver.py`), and optionally **`uvicorn_port`** for the host/port sidecar file.

### System Architecture

SQuAI consists of four key agents working collaboratively to deliver accurate, faithful, and verifiable answers:

1. **Agent 1: Decomposer**  
   Decomposes complex user queries into simpler, semantically distinct sub-questions. This step ensures that each aspect of the question is treated with focused retrieval and generation, enabling precise evidence aggregation.

2. **Agent 2: Generator**  
   For each sub-question, this agent processes retrieved documents to generate structured **Question–Answer–Evidence (Q-A-E)** triplets. These triplets form the backbone of transparent and evidence-grounded answers.

3. **Agent 3: Judge**  
   Evaluates the relevance and quality of each Q-A-E triplet using a learned scoring mechanism. It filters out weak or irrelevant documents based on confidence thresholds, dynamically tuned to the difficulty of each query.

4. **Agent 4: Answer Generator**  
   Synthesizes a final, coherent answer from filtered Q-A-E triplets. Critically, it includes **fine-grained in-line citations** and citation context to enhance trust and verifiability. Every factual statement is explicitly linked to one or more supporting documents.

### Retrieval Engine

The agents are supported by a **hybrid retrieval system** that combines:

- **Sparse retrieval** (BM25) for keyword overlap and exact matching.
- **Dense retrieval** (E5 embeddings) for semantic similarity.

The system interpolates scores from both methods to maximize both lexical precision and semantic coverage.

```math
S_{hybrid}(d) = \alpha \cdot S_{sparse}(d) + (1 - \alpha) \cdot S_{dense}(d)
```

\(\alpha = 0.65\), based on empirical tuning. This slightly favors dense retrieval while retaining complementary signals from sparse methods, ensuring both semantic relevance and precision.

### User Interface

SQuAI includes an interactive web-based UI built with **Streamlit** and backed by a **FastAPI** server. Key features include:

- A simple input form for entering scientific questions.
- Visualization of decomposed sub-questions.
- Toggle between sparse, dense, and hybrid retrieval modes.
- Adjustable settings for document filtering thresholds and top-k retrieval.
- Display of generated answers with **fine-grained in-line citations**.
- Clickable references linking to original arXiv papers.

### Benchmarks & Evaluation

We evaluate SQuAI using three QA datasets designed to test performance across varying complexity levels:

- **LitSearch**: Real-world literature review queries from computer science.
- **unarXive Simple**: General questions with minimal complexity.
- **unarXive Expert**: Highly specific and technical questions requiring deep evidence grounding.

Evaluation metrics (via [DeepEval](https://deepeval.com)) include:

- **Answer Relevance** – How well the answer semantically matches the question.
- **Contextual Relevance** – How well the answer integrates retrieved evidence.
- **Faithfulness** – Whether the answer is supported by cited sources.

SQuAI improves combined scores by up to **12%** in faithfulness compared to a standard RAG baseline.

### Dataset & Resources

- **unarXive 2024**: Full-text arXiv papers with structured metadata, section segmentation, and citation annotations. [Hugging Face Dataset](https://huggingface.co/datasets/ines-besrour/unarxive_2024)
- **QA Triplet Benchmark**: 1,000 synthetic question–answer–evidence triplets for reproducible evaluation.

### Server deployment (optional)

Some deployments use the following; adjust to your environment:

- **`$HOME/data_dir`**: Legacy pattern — a file in `$HOME` named `data_dir` whose contents may point to a FAISS workspace path. Prefer **`SQUAI_DATA_DIR`** for clarity.
- **How to restart the service if it isn't working:** `sudo systemctl restart squai-frontend.service`
- **Data is not being copied:** Check if `/etc/dont_copy` exists.

### Citation

If you use this repository, please cite:

```bibtex
@inproceedings{Besrour2025SQuAI,
author = {Besrour, Ines and He, Jingbo and Schreieder, Tobias and F\"{a}rber, Michael},
title = {SQuAI: Scientific Question-Answering with Multi-Agent Retrieval-Augmented Generation},
year = {2025},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi = {10.1145/3746252.3761471},
booktitle = {Proceedings of the 34th ACM International Conference on Information and Knowledge Management},
pages = {6603--6608},
location = {Seoul, Republic of Korea},
series = {CIKM '25}
}
```
