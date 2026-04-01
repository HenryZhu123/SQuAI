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
