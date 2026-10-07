# Central place for all adjustable settings.
# Change values here instead of hunting through the code.

DATA_DIR = "data"
DEFAULT_FILENAME = "Sample_ICU_note.pdf"

SIMILARITY_TOP_K = 6       # how many document chunks to retrieve per question
CHUNK_PREVIEW_LENGTH = 300  # how many characters to show in debug previews

ANSWER_MODEL = "gpt-4o-mini"   # model that writes the answers
ANSWER_TEMPERATURE = 0         # 0 = most consistent answers between runs