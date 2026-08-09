import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# GEMINI_API_KEY is intentionally not read here: this module is imported before
# load_dotenv() runs in some entrypoints, so llm_client.py reads it lazily instead.
#
# Model choice: "gemini-flash-latest" now resolves to "gemini-3.6-flash", whose
# free tier is capped at 20 generate_content requests/day on this account -
# unworkable once evals run in CI on every push. "gemini-2.5-flash*" is 404
# (deprecated for new users on this account) and "gemini-2.0-flash*" is a
# permanent 0-quota free tier here. "gemini-flash-lite-latest" has a workable
# free-tier quota and, as a bonus, has no "thinking" step eating output tokens
# (it also rejects any thinking_config at all - don't pass one).
GENERATION_MODEL = "gemini-flash-lite-latest"
EMBEDDING_MODEL = "models/gemini-embedding-001"

RAW_DATA_DIR = Path(os.environ.get("RAW_DATA_DIR", str(BASE_DIR / "data" / "raw")))

# Defaults outside OneDrive-synced Documents: Python's own file APIs fail to create
# new directories inside this repo's location on this machine (WinError 2). Override
# via env for Docker/deploy targets where that constraint doesn't apply.
CHROMA_DATA_DIR = Path(
    os.environ.get("CHROMA_DATA_DIR", r"C:\venvs\agente_ai_data\chroma_data")
)
CHROMA_COLLECTION_NAME = "functional_medicine"

LOG_DIR = Path(os.environ.get("LOG_DIR", r"C:\venvs\agente_ai_data\logs"))

RETRIEVAL_TOP_K = 5
MAX_OUTPUT_TOKENS = 500
FAITHFULNESS_THRESHOLD = 4.0
MAX_REVISIONS = 2
