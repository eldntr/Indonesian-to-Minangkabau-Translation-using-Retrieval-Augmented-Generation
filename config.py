# Konfigurasi untuk model SentenceTransformer
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# Path ke dataset CSV
CSV_FILE_PATH = "datasetv2/train.csv"

# Ambang batas skor similaritas kosinus untuk pencarian
SIMILARITY_THRESHOLD = 0.4

# Konfigurasi untuk LLM
LLM_MODEL = "google/gemini-2.0-flash-001"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"