from pathlib import Path


# Project directories
BASE_DIR = Path(__file__).resolve().parent

DOCUMENTS_DIR = BASE_DIR / "documents"
DATA_DIR = BASE_DIR / "data"

DATABASE_PATH = DATA_DIR / "knowledge.db"


# Minimum similarity required to create
# a relationship between two documents.
#
# 0.0 = connect everything
# 1.0 = practically impossible
#
# 0.10 - 0.20 is a good starting range.
SIMILARITY_THRESHOLD = 0.10


# Ignore extremely short words.
MIN_WORD_LENGTH = 2


# Ignore words that appear in too many documents.
# This prevents generic words from dominating relationships.
MAX_DOCUMENT_FREQUENCY_RATIO = 0.80