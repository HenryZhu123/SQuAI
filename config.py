import os
import torch

# Windows: directory containing faiss_index/, bm25_retriever/, full_text_db/
# Change this if your indices and DB are stored elsewhere.
MAIN_DATA_DIR = r"D:\SQuAI\SQuAI-main\squai_data"

EMBEDDING_MODEL = "intfloat/e5-large-v2"
MODEL_FORMAT = "sentence_transformers"
EMBEDDING_DIM = 1024
USE_GPU = torch.cuda.is_available()

# Configuration paths
DATA_DIR = f"{MAIN_DATA_DIR}_extended_data"
E5_INDEX_DIR = os.path.join(MAIN_DATA_DIR, "faiss_index")
BM25_INDEX_DIR = os.path.join(MAIN_DATA_DIR, "bm25_retriever")
DB_PATH = os.path.join(MAIN_DATA_DIR, "full_text_db")
