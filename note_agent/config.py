import os
from dotenv import load_dotenv

# =========================================
# Environment & Constants
# =========================================
load_dotenv()
assert os.getenv(
    "OPENAI_API_KEY"
), "환경 변수 OPENAI_API_KEY 가 필요합니다 (.env 설정)."
assert os.getenv(
    "OPENAI_API_KEY"
), "환경 변수 OPENAI_API_KEY 가 필요합니다 (.env 설정)."


PERSIST_DIR = os.getenv("PERSIST_DIR") or "./rag_store"
RESULTS_DIR = os.getenv("RESULTS_DIR") or "./results"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"
LLM_MODEL = os.getenv("LLM_MODEL") or "gpt-4o-mini"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or "900")
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or "120")
RETRIEVE_K = int(os.getenv("RETRIEVE_K") or "3")
TEMPL_COMPLETE = float(os.getenv("TEMPL_COMPLETE") or "0.2")
TEMP_SUMMARY = float(os.getenv("TEMP_SUMMARY") or "0.3")

DB_URL = os.getenv("DB_URL", "sqlite:///note_agent/data/note_agent.db")
