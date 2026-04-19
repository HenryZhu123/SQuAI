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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


# Optional Paperclip path for Agent2 input documents.
# Disabled by default to preserve existing behavior.
PAPERCLIP_AGENT2_ENABLED = _env_bool("SQUAI_PAPERCLIP_AGENT2", False)
PAPERCLIP_SEARCH_TOP_K = int((os.environ.get("SQUAI_PAPERCLIP_SEARCH_TOP_K") or "5").strip())
PAPERCLIP_SEARCH_TIMEOUT_SEC = int(
    (os.environ.get("SQUAI_PAPERCLIP_SEARCH_TIMEOUT_SEC") or "20").strip()
)
PAPERCLIP_CAT_TIMEOUT_SEC = int(
    (os.environ.get("SQUAI_PAPERCLIP_CAT_TIMEOUT_SEC") or "30").strip()
)
PAPERCLIP_CHUNK_SIZE = int((os.environ.get("SQUAI_PAPERCLIP_CHUNK_SIZE") or "1400").strip())
PAPERCLIP_CHUNK_OVERLAP = int(
    (os.environ.get("SQUAI_PAPERCLIP_CHUNK_OVERLAP") or "250").strip()
)
PAPERCLIP_AGENT2_TOP_CHUNKS = int(
    (os.environ.get("SQUAI_PAPERCLIP_AGENT2_TOP_CHUNKS") or "3").strip()
)
PAPERCLIP_CAT_WORKERS = int((os.environ.get("SQUAI_PAPERCLIP_CAT_WORKERS") or "4").strip())
PAPERCLIP_CACHE_MAX_ITEMS = int(
    (os.environ.get("SQUAI_PAPERCLIP_CACHE_MAX_ITEMS") or "32").strip()
)
PAPERCLIP_CACHE_MAX_BYTES = int(
    (os.environ.get("SQUAI_PAPERCLIP_CACHE_MAX_BYTES") or str(40 * 1024 * 1024)).strip()
)
