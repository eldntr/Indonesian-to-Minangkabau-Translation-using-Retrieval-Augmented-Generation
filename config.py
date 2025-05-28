# Konfigurasi untuk model SentenceTransformer
MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# Path ke dataset CSV
CSV_FILE_PATH = "dataset/train.csv"

# Ambang batas skor similaritas kosinus untuk pencarian
SIMILARITY_THRESHOLD = 0.4

# Konfigurasi untuk LLM
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"