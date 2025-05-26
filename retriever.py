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
        """Memuat data dari file CSV atau menggunakan data contoh internal."""
        if csv_file_path:
            try:
                print(f"Mencoba memuat data dari: {csv_file_path}")
                df = pd.read_csv(csv_file_path)
                print("Data berhasil dimuat dari file.")
                return df
            except FileNotFoundError:
                print(f"File '{csv_file_path}' tidak ditemukan.")
            except Exception as e:
                print(f"Error saat memuat file CSV '{csv_file_path}': {e}")
        
        print("Menggunakan data contoh internal karena file tidak disediakan atau gagal dimuat.")
        csv_data_internal = """indonesian,minangkabau
"saat ini, makam syekh sihalahan telah diberi cungkup dan dibuatkan bangunan yang disusun dengan bata berplester dengan ukuran 4 x 3 meter","kiniko, makam syekh sihalahan alah diagia cungkup dan dibuekan bangunan nan disusun jo bata berplester dengan ukuran 4 x 3 meter"
"majelis islam tinggi adalah badan tertinggi yang bertanggung jawab untuk urusan komunitas muslim di wilayah sumatera barat, badan ini awalnya dibentuk dengan tujuan 1sebagai suatu badan penasehat yang terdiri dari umat muslim.majelis islam tinggi atau biasa disingkat dengan sebutan mit didirikan dengan tujuan untuk menghimpun perjuangan umat islam yang ada diminangkabau.majelis islam tinggi ini didirikan oleh seorang syeikh yang bernama syeikh muhammad jamil jambek q.v.beliau banyak mengadakan kegiatan pembaharuan dan pemurnian mengenai islam melalui ceramahceramah dan dakwah secara lisan serta menulis bukubuku yang menolak amalan tarekat secara berlebihan.anggota dari majelis islam tinggi ini sudah banyak tergabung dalam majelis iskam tinggi,diantaranya tokoh ulama besar yakni haji abdul karim amrullah juga dikenali sebagai haji rasul atau inyiak doto, 18791945 merupakan antara tokoh lain yang mengasaskan sekolah islam moden yang pertama sumatera thawalib 1919 yakni yang berada di padang panjang bersamasama dengan tokoh haji abdullah ahmad 18781933.","majelis islam tinggi adolah badan tinggi nan bertanggung jawab panuah untuak urusan komunitas muslim di sumatera barat, badan iko awalnya dibantuak 1sabagai suatu badan nan manasehati nan tadiri dari umek muslim.majelis islam tinggi atau biaso awak singkat namonyo jo sabutan mit dibantuak untuak mangumpuan pajuangan umat islam nan ado di diminangkabau.majelis islam tinggi iko didirian dek salah satu syeikh nan tanamo banamo syeikh muhammad jamil jambek q.v.baliau banyak meadokan acara baru dan itu murni untuak islam melalui ceramahceramah dan dakwah yang dilakukan secara muluik ka muluik dan menulis bukubuku nan manolak amalan tarekat secara berlebihan.anggota dari majelis islam tinggi iko alah banyak tagabuang dalam majelis iskam tinggi,diantaranyo tokoh ulama besar yaitu haji abdul karim amrullah dikenal juo sebagai haji rasul atau inyiak doto, 18791945 yang merupakan antara tokoh lain yang barazaskan sakola islam moden nan partamo sumatera thawalib 1919 yaitu barado di padang panjang basamosamo jo tokoh haji abdullah ahmad 18781933."
"masjid ini memiliki arsitektur tiongkok dan minangkabau.","masjid iko punyo arsitektur dari cino jo minangkabau."
"""
        return pd.read_csv(io.StringIO(csv_data_internal))

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

class PromptGenerator:
    """
    Kelas untuk menghasilkan prompt yang akan diproses oleh LLM.
    """
    @staticmethod
    def for_sentence_translation_elaboration(query_word, found_word, similarity, indonesian_sentence, minangkabau_sentence):
        """
        Membuat prompt untuk meminta elaborasi atau verifikasi terjemahan kalimat.
        """
        prompt = f"""Analisis Pasangan Kalimat Indonesia-Minang berikut:

Kata Kunci dari Query Pengguna: "{query_word}"
Kata Terkait yang Ditemukan di Basis Data: "{found_word}" (Skor Kemiripan: {similarity:.2f})

Kalimat Bahasa Indonesia:
"{indonesian_sentence}"

Kalimat Bahasa Minang (Terjemahan yang Ditemukan):
"{minangkabau_sentence}"

Tugas Anda:
1.  Evaluasi keakuratan dan kenaturalan terjemahan dari Bahasa Indonesia ke Bahasa Minang tersebut.
2.  Jika ada, berikan alternatif terjemahan yang mungkin lebih baik atau lebih umum digunakan dalam konteks Bahasa Minang.
3.  Berikan sedikit penjelasan mengenai konteks penggunaan kalimat Minang tersebut atau nuansa makna yang mungkin ada.
4.  Jika kata kunci query ("{query_word}") dan kata yang ditemukan ("{found_word}") berbeda, jelaskan mengapa sistem mungkin menganggapnya mirip secara semantik.

Mohon berikan jawaban yang jelas dan informatif.
"""
        return prompt

    @staticmethod
    def for_query_summary_and_answer(original_query, retrieved_data_list):
        """
        Membuat prompt untuk merangkum temuan atau menjawab query berdasarkan data yang diambil.
        """
        if not retrieved_data_list:
            return f"Tidak ada informasi relevan yang ditemukan untuk query: \"{original_query}\". Mohon coba query lain atau periksa basis data."

        context_str = "Informasi yang berhasil diambil dari basis data:\n\n"
        for item in retrieved_data_list:
            context_str += f"- Untuk kata kunci query \"{item['original_query_word']}\":\n"
            context_str += f"  - Ditemukan kata mirip di korpus: \"{item['found_word_in_corpus']}\" (Skor: {item['similarity_score']:.2f})\n"
            context_str += f"  - Contoh Kalimat Indonesia: \"{item['retrieved_example']['indonesian']}\"\n"
            context_str += f"  - Contoh Kalimat Minang: \"{item['retrieved_example']['minangkabau']}\"\n\n"

        prompt = f"""Query Pengguna:
"{original_query}"

Konteks yang Ditemukan:
{context_str}
Tugas Anda:
1.  Berdasarkan konteks di atas, berikan jawaban atau penjelasan yang relevan terhadap query pengguna ("{original_query}").
2.  Jika memungkinkan, rangkum informasi penting dari contoh-contoh kalimat yang ditemukan.
3.  Jika ada ambiguitas atau informasi yang kurang, sebutkan.

Mohon berikan jawaban yang komprehensif dan langsung menjawab pertanyaan pengguna.
"""
        return prompt

# --- Blok Utama untuk Pengujian ---
if __name__ == "__main__":
    # Inisialisasi Retriever
    # Ganti dengan path ke file CSV Anda jika ada, atau biarkan None untuk menggunakan data internal.
    # Contoh: retriever = SemanticRetriever(csv_file_path="dataset/train.csv")
    retriever = SemanticRetriever(csv_file_path=None) 

    if retriever.model and (retriever.vocab_list or retriever.corpus_embeddings.size > 0) :
        # Contoh Query
        # Query ini menggunakan kata "bangunan ibadah", yang tidak ada di teks asli.
        # Teks asli mengandung kata "masjid" dan "makam".
        # query_pengguna = "arsitektur bangunan ibadah tiongkok"
        query_pengguna = "siapa syekh sihalahan"
        
        print(f"\nMelakukan pencarian untuk query: \"{query_pengguna}\"")
        hasil_pencarian = retriever.retrieve(query_pengguna, similarity_threshold=0.4)

        if not hasil_pencarian:
            print(f"\nTidak ada hasil yang ditemukan untuk query: \"{query_pengguna}\"")
        else:
            print(f"\n--- Hasil Pencarian Semantik untuk Query: \"{query_pengguna}\" ---")
            
            list_data_untuk_prompt_summary = []

            for kata_query, data_hasil in hasil_pencarian.items():
                print(f"\nKata Kunci dari Query: '{kata_query}'")
                print(f"  -> Ditemukan kata paling mirip di korpus: '{data_hasil['found_word_in_corpus']}' (Skor Kemiripan: {data_hasil['similarity_score']:.2f})")
                
                contoh = data_hasil['retrieved_example']
                print("  -> Contoh Kalimat Terpilih:")
                print(f"    - Indo  : {contoh['indonesian']}")
                print(f"    - Minang: {contoh['minangkabau']}")
                
                # Menyiapkan data untuk prompt summary
                list_data_untuk_prompt_summary.append({
                    "original_query_word": kata_query,
                    **data_hasil # Menyalin semua key dari data_hasil
                })

                # Membuat dan menampilkan prompt untuk elaborasi kalimat
                prompt_elaborasi = PromptGenerator.for_sentence_translation_elaboration(
                    kata_query,
                    data_hasil['found_word_in_corpus'],
                    data_hasil['similarity_score'],
                    contoh['indonesian'],
                    contoh['minangkabau']
                )
                print("\n  --- Prompt untuk LLM (Elaborasi Kalimat) ---")
                print(prompt_elaborasi)
                print("-" * 50)

            # Membuat dan menampilkan prompt untuk summary keseluruhan query
            if list_data_untuk_prompt_summary:
                prompt_summary_keseluruhan = PromptGenerator.for_query_summary_and_answer(
                    query_pengguna,
                    list_data_untuk_prompt_summary
                )
                print("\n\n--- Prompt untuk LLM (Summary Keseluruhan Query) ---")
                print(prompt_summary_keseluruhan)
                print("=" * 60)
    else:
        print("Retriever tidak dapat diinisialisasi dengan benar (model atau data tidak tersedia). Pengujian dibatalkan.")