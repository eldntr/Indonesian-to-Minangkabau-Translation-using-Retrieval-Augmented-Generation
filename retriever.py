import pandas as pd
import re
import io # Digunakan untuk data string jika file CSV tidak ada
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticRetriever:
    """
    Kelas untuk melakukan pencarian semantik pada dataset kalimat paralel.
    """
    def __init__(self, csv_file_path=None, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        Inisialisasi retriever.

        Args:
            csv_file_path (str, optional): Path ke file CSV. 
                                          Jika None, akan menggunakan data contoh internal.
            model_name (str): Nama model SentenceTransformer yang akan digunakan.
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
            print("Pastikan Anda memiliki koneksi internet saat pertama kali menjalankan untuk mengunduh model,")
            print("atau pastikan model tersedia secara lokal.")
            return None

    def _load_data(self, csv_file_path):
            """Memuat data dari file CSV atau mengembalikan error jika gagal."""
            if csv_file_path:
                try:
                    print(f"Mencoba memuat data dari: {csv_file_path}")
                    df = pd.read_csv(csv_file_path)
                    print("Data berhasil dimuat dari file.")
                    return df
                except FileNotFoundError:
                    error_message = f"File '{csv_file_path}' tidak ditemukan."
                    print(error_message)
                    raise FileNotFoundError(error_message)
                except Exception as e:
                    error_message = f"Error saat memuat file CSV '{csv_file_path}': {e}"
                    print(error_message)
                    raise Exception(error_message)
            else:
                error_message = "Path ke file CSV tidak diberikan dan data internal tidak tersedia."
                print(error_message)
                raise ValueError(error_message)

    def _preprocess_text(self, text):
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
        for index, row in self.df.iterrows():
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

    def _generate_corpus_embeddings(self):
        """Membuat embedding untuk seluruh kosakata."""
        if not self.vocab_list:
            print("Kosakata kosong, tidak ada embedding yang dibuat.")
            return
        if not self.model:
            print("Model tidak dimuat, tidak dapat membuat embedding.")
            return

        print("Membuat embedding untuk kosakata...")
        try:
            self.corpus_embeddings = self.model.encode(self.vocab_list, show_progress_bar=True)
            print("Pembuatan embedding korpus selesai.")
        except Exception as e:
            print(f"Error saat membuat embedding korpus: {e}")
            self.corpus_embeddings = np.array([]) # Pastikan tetap array numpy kosong

    def retrieve(self, query, similarity_threshold=0.5):
        """
        Mencari setiap kata dalam query secara semantik.
        Mengembalikan satu contoh kalimat paling relevan untuk setiap kata query.
        """
        if not self.model:
            print("Model tidak tersedia. Tidak dapat melakukan pencarian.")
            return {}
        if not self.vocab_list or self.corpus_embeddings.size == 0:
            print("Kosakata atau embedding korpus tidak tersedia. Tidak dapat melakukan pencarian.")
            return {}

        query_words = self._preprocess_text(query).split()
        results = {}

        for word in query_words:
            if not word: continue
            
            try:
                query_embedding = self.model.encode([word])
            except Exception as e:
                print(f"Error saat membuat embedding untuk kata query '{word}': {e}")
                continue

            similarities = cosine_similarity(query_embedding, self.corpus_embeddings)
            most_similar_idx = np.argmax(similarities)
            
            if similarities[0, most_similar_idx] >= similarity_threshold:
                found_word = self.vocab_list[most_similar_idx]
                all_examples_for_found_word = self.word_to_sentence_map.get(found_word, [])
                
                if all_examples_for_found_word:
                    selected_example = all_examples_for_found_word[0] 
                    results[word] = {
                        "found_word_in_corpus": found_word,
                        "similarity_score": float(similarities[0, most_similar_idx]),
                        "retrieved_example": selected_example
                    }
        return results

def for_query_summary_and_answer(original_query, retrieved_data_list):
        if not retrieved_data_list:
            return f"No relevant information was found for the query: \"{original_query}\". Please try another query or check the database."

        context_str = "Here is the information related to Indonesian words with example sentences in Indonesian-Minangkabau:\n\n"
        for item in retrieved_data_list:
            context_str += f"- For the word \"{item['original_query_word']}\":\n"
            # context_str += f"  - Similar word found in the corpus: \"{item['found_word_in_corpus']}\" (Score: {item['similarity_score']:.2f})\n"
            context_str += f"  - Example Sentence in Indonesian: \"{item['retrieved_example']['indonesian']}\"\n"
            context_str += f"  - Example Sentence in Minangkabau: \"{item['retrieved_example']['minangkabau']}\"\n\n"

        prompt = f"""
{context_str}
Your Task:
1. Note that the given words are in Indonesian.
2. Each word has an example sentence in Indonesian and its translation in Minangkabau.
3. Translate the following Indonesian sentence: "{original_query}" into Minangkabau.
Provide the answer directly as the translated sentence:
"""
        return prompt
    
# --- Blok Utama untuk Pengujian ---
if __name__ == "__main__":
    # Inisialisasi Retriever
    # Ganti dengan path ke file CSV Anda jika ada, atau biarkan None untuk menggunakan data internal.
    retriever = SemanticRetriever(csv_file_path="dataset/train.csv")

    if retriever.model and (retriever.vocab_list or retriever.corpus_embeddings.size > 0) :
        # Contoh Query
        # Query ini menggunakan kata "bangunan ibadah", yang tidak ada di teks asli.
        # Teks asli mengandung kata "masjid" dan "makam".
        # query_pengguna = "arsitektur bangunan ibadah tiongkok"
        query_pengguna = "skki juga memiliki program gelar ganda ambisius dengan sejumlah universitas ternama di dunia"
        
        print(f"\nMelakukan pencarian untuk query: \"{query_pengguna}\"")
        hasil_pencarian = retriever.retrieve(query_pengguna, similarity_threshold=0.4)

        if not hasil_pencarian:
            print(f"\nTidak ada hasil yang ditemukan untuk query: \"{query_pengguna}\"")
        else:
            print(f"\n--- Hasil Pencarian Semantik untuk Query: \"{query_pengguna}\" ---")
            
            list_data_untuk_prompt_summary = []

            for kata_query, data_hasil in hasil_pencarian.items():
                list_data_untuk_prompt_summary.append({
                    "original_query_word": kata_query,
                    **data_hasil # Menyalin semua key dari data_hasil
                })

            # Membuat dan menampilkan prompt untuk summary keseluruhan query
            if list_data_untuk_prompt_summary:
                prompt_summary_keseluruhan = for_query_summary_and_answer(
                    query_pengguna,
                    list_data_untuk_prompt_summary
                )
                print("\n\n--- Prompt untuk LLM (Summary Keseluruhan Query) ---")
                print(prompt_summary_keseluruhan)
                print("=" * 60)
    else:
        print("Retriever tidak dapat diinisialisasi dengan benar (model atau data tidak tersedia). Pengujian dibatalkan.")