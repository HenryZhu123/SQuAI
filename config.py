import os
import torch

# Linux: directory containing faiss_index/, bm25_retriever/, full_text_db/
# Default layout: project at /home/ubuntu/SQuAI with data in squai_data/
# Override with env SQUAI_DATA_DIR (same as get_paths.get_main_data_dir).
_MAIN_DEFAULT = "/home/ubuntu/SQuAI/squai_data"
_raw = (os.environ.get("SQUAI_DATA_DIR") or "").strip()
MAIN_DATA_DIR = os.path.abspath(_raw) if _raw else _MAIN_DEFAULT

EMBEDDING_MODEL = "intfloat/e5-large-v2"
MODEL_FORMAT = "sentence_transformers"
EMBEDDING_DIM = 1024
USE_GPU = torch.cuda.is_available()

# Configuration paths
DATA_DIR = f"{MAIN_DATA_DIR}_extended_data"
E5_INDEX_DIR = os.path.join(MAIN_DATA_DIR, "faiss_index")
BM25_INDEX_DIR = os.path.join(MAIN_DATA_DIR, "bm25_retriever")
DB_PATH = os.path.join(MAIN_DATA_DIR, "full_text_db")

# PostgreSQL full-text KV (paper_id -> blob). Used when FULLTEXT_PG_DSN is non-empty.
# Env SQUAI_FULLTEXT_PG_DSN overrides the default string below (recommended for secrets).
# Example: FULLTEXT_PG_DSN_DEFAULT = "postgresql://user:pass@localhost:5432/squai"
FULLTEXT_PG_DSN_DEFAULT = ""
_pg_dsn_env = (os.environ.get("SQUAI_FULLTEXT_PG_DSN") or "").strip()
FULLTEXT_PG_DSN = _pg_dsn_env or (FULLTEXT_PG_DSN_DEFAULT or "").strip()

# Table for key/value rows. Env SQUAI_FULLTEXT_PG_TABLE overrides default.
FULLTEXT_PG_TABLE_DEFAULT = "squai_fulltext_kv"
_pg_table_env = (os.environ.get("SQUAI_FULLTEXT_PG_TABLE") or "").strip()
FULLTEXT_PG_TABLE = _pg_table_env or FULLTEXT_PG_TABLE_DEFAULT
