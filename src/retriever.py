import pandas as pd
import re
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticRetriever:
    """
    Kelas untuk melakukan pencarian semantik pada dataset kalimat paralel.
    """
    def __init__(self, model_name: str, csv_file_path: str):
        """
        Inisialisasi retriever.

        Args:
            model_name (str): Nama model SentenceTransformer yang akan digunakan.
            csv_file_path (str): Path ke file CSV.
        """
        self.model_name = model_name
        self.model = self._load_sbert_model()
        
        self.df = self._load_data(csv_file_path)
        self.word_to_sentence_map = {}
        self.vocab_list = []
        self.corpus_embeddings = np.array([])

        if not self.df.empty:
            self._preprocess_data()
            self._generate_corpus_embeddings()
        else:
            print("Peringatan: DataFrame kosong, tidak ada data yang diproses.")

    def _load_sbert_model(self):
        """Memuat model SentenceTransformer."""
        try:
            print(f"Memuat model SentenceTransformer: {self.model_name}...")
            model = SentenceTransformer(self.model_name)
            print("Model berhasil dimuat.")
            return model
        except Exception as e:
            print(f"Gagal memuat model SentenceTransformer '{self.model_name}': {e}")
            return None

    def _load_data(self, csv_file_path: str):
        """Memuat data dari file CSV atau mengembalikan error jika gagal."""
        try:
            print(f"Mencoba memuat data dari: {csv_file_path}")
            return pd.read_csv(csv_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{csv_file_path}' tidak ditemukan.")
        except Exception as e:
            raise Exception(f"Error saat memuat file CSV '{csv_file_path}': {e}")

    def _preprocess_text(self, text: str) -> str:
        """Membersihkan teks: lowercase dan hapus tanda baca."""
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'[^\w\s]', '', text)
            return text
        return ""

    def _preprocess_data(self):
        """Melakukan pra-pemrosesan pada kolom 'indonesian' dan membangun kosakata."""
        print("Memulai pra-pemrosesan data...")
        self.df['indonesian_preprocessed'] = self.df['indonesian'].apply(self._preprocess_text)
        
        vocabulary_set = set()
        for _, row in self.df.iterrows():
            words = row['indonesian_preprocessed'].split()
            for word in words:
                if not word: continue
                vocabulary_set.add(word)
                if word not in self.word_to_sentence_map:
                    self.word_to_sentence_map[word] = []
                self.word_to_sentence_map[word].append({
                    "indonesian": row['indonesian'],
                    "minangkabau": row['minangkabau']
                })
        self.vocab_list = sorted(list(vocabulary_set))
        print(f"Pra-pemrosesan selesai. Ukuran kosakata: {len(self.vocab_list)} kata unik.")

    def _save_embeddings(self, embeddings: np.ndarray, file_name: str):
        """Menyimpan embedding ke file."""
        model_dir = "model"
        os.makedirs(model_dir, exist_ok=True)
        file_path = os.path.join(model_dir, file_name)
        np.save(file_path, embeddings)
        print(f"Embedding disimpan ke: {file_path}")

    def _load_embeddings(self, file_name: str) -> np.ndarray:
        """Memuat embedding dari file."""
        model_dir = "model"
        file_path = os.path.join(model_dir, file_name)
        if os.path.exists(file_path):
            print(f"Memuat embedding dari file: {file_path}")
            return np.load(file_path)
        return np.array([])

    def _generate_corpus_embeddings(self):
        """Membuat atau memuat embedding untuk seluruh kosakata."""
        embeddings_file = f"{self.model_name}_embeddings.npy"
        
        self.corpus_embeddings = self._load_embeddings(embeddings_file)
        if self.corpus_embeddings.size > 0:
            return

        if not self.vocab_list or not self.model:
            print("Kosakata atau model tidak tersedia, embedding tidak dibuat.")
            return

        print("Membuat embedding untuk kosakata...")
        try:
            self.corpus_embeddings = self.model.encode(self.vocab_list, show_progress_bar=True)
            self._save_embeddings(self.corpus_embeddings, embeddings_file)
            print("Pembuatan embedding korpus selesai.")
        except Exception as e:
            print(f"Error saat membuat embedding korpus: {e}")
            self.corpus_embeddings = np.array([])

    def retrieve(self, query: str, similarity_threshold: float) -> dict:
        """
        Mencari setiap kata dalam query secara semantik dan mengembalikan contoh kalimat.
        """
        if not self.model or self.corpus_embeddings.size == 0:
            print("Model atau embedding korpus tidak tersedia. Pencarian dibatalkan.")
            return {}

        query_words = self._preprocess_text(query).split()
        results = {}

        for word in query_words:
            if not word: continue
            
            query_embedding = self.model.encode([word])
            similarities = cosine_similarity(query_embedding, self.corpus_embeddings)
            most_similar_idx = np.argmax(similarities)
            
            if similarities[0, most_similar_idx] >= similarity_threshold:
                found_word = self.vocab_list[most_similar_idx]
                all_examples = self.word_to_sentence_map.get(found_word, [])
                
                if all_examples:
                    results[word] = {
                        "found_word_in_corpus": found_word,
                        "similarity_score": float(similarities[0, most_similar_idx]),
                        "retrieved_example": all_examples[0]
                    }
        return results