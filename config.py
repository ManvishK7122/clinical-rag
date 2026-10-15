# Central place for all adjustable settings.
# Change values here instead of hunting through the code.

DATA_DIR = "data"
DEFAULT_FILENAME = "Sample_ICU_note.pdf"

SIMILARITY_TOP_K = 6       # how many document chunks to retrieve per question
CHUNK_PREVIEW_LENGTH = 300  # how many characters to show in debug previews

ANSWER_MODEL = "gpt-4o-mini"   # model that writes the answers
ANSWER_TEMPERATURE = 0         # 0 = most consistent answers between runs

# Patient snapshot and trial matching
EXTRACT_MODEL = "gpt-4o-mini"  # model for snapshot extraction and criteria checks
OUTPUT_DIR = "outputs"         # snapshots and match reports are saved here
TRIALS_STATUS = "RECRUITING"   # only search trials that are currently enrolling
TRIALS_PAGE_SIZE = 10          # how many trials to fetch per search
TRIALS_DIR = "evals/trials"    # frozen copies of search results, so evals are repeatable

# Reliability and cost tracking
MAX_RETRIES = 3          # retry a failed API call this many times (with backoff)
REQUEST_TIMEOUT = 60     # seconds before an API call is abandoned
# USD per 1M tokens: (input, output). Check current prices on OpenAI's pricing page.
PRICE_PER_1M = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}
