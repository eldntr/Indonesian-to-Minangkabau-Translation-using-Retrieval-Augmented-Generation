# main.py

from src.retriever import SemanticRetriever
from src.utils import generate_translation_prompt
import config

def main():
    """
    Fungsi utama untuk menjalankan aplikasi pencarian semantik.
    """
    print("--- Memulai Aplikasi Penerjemah Semantik ---")
    
    try:
        # 1. Inisialisasi Retriever dengan konfigurasi
        retriever = SemanticRetriever(
            model_name=config.MODEL_NAME,
            csv_file_path=config.CSV_FILE_PATH
        )
    except (FileNotFoundError, Exception) as e:
        print(f"Gagal menginisialisasi retriever: {e}")
        return

    # Periksa apakah retriever berhasil diinisialisasi
    if not retriever.model or retriever.corpus_embeddings.size == 0:
        print("Retriever tidak dapat diinisialisasi dengan benar. Aplikasi berhenti.")
        return

    # 2. Contoh Query Pengguna
    query_pengguna = "skki juga memiliki program gelar ganda ambisius dengan sejumlah universitas ternama di dunia"
    
    print(f"\nMelakukan pencarian untuk query: \"{query_pengguna}\"")
    
    # 3. Lakukan pencarian
    hasil_pencarian = retriever.retrieve(query_pengguna, similarity_threshold=config.SIMILARITY_THRESHOLD)

    # 4. Proses dan tampilkan hasil
    if not hasil_pencarian:
        print(f"\nTidak ada hasil yang ditemukan untuk query: \"{query_pengguna}\"")
    else:
        print(f"\n--- Hasil Pencarian Semantik Ditemukan ---")
        
        list_data_untuk_prompt = [
            {"original_query_word": kata_query, **data_hasil}
            for kata_query, data_hasil in hasil_pencarian.items()
        ]

        # 5. Buat prompt untuk LLM
        prompt_final = generate_translation_prompt(
            query_pengguna,
            list_data_untuk_prompt
        )
        
        print("\n--- Prompt untuk LLM (Terjemahan Keseluruhan Query) ---")
        print(prompt_final)
        print("=" * 60)

if __name__ == "__main__":
    main()