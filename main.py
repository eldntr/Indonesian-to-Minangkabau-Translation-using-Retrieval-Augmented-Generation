# main.py

from src.retriever import SemanticRetriever
from src.utils import generate_translation_prompt
from src.llm_handler import send_prompt_to_llm
from src.evaluation_metrics import calculate_bleu, calculate_rouge, calculate_chrf
import config
from dotenv import load_dotenv
import os

def main():
    """
    Fungsi utama untuk menjalankan aplikasi pencarian semantik.
    """
    # Muat variabel lingkungan dari file .env
    load_dotenv()
    
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
    query_pengguna = "tempat pemuatan iklan di lembaran ketiga dan empat"
    kunci_jawaban = "tampek pamuatan iklan di lembaran katigo dan ampek"
    
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

        # 6. Kirim prompt ke LLM menggunakan API OpenRouter
        response = send_prompt_to_llm(prompt_final)
        print("\n--- Jawaban dari LLM ---")
        print(response)

        # 7. Evaluasi berbagai metrik
        if response:
            # BLEU Score
            bleu_score = calculate_bleu(kunci_jawaban, response)
            print(f"\n--- BLEU Score ---\n{bleu_score}")

            # ROUGE Scores
            rouge_scores = calculate_rouge(kunci_jawaban, response)
            print(f"\n--- ROUGE Scores ---")
            print(f"ROUGE-1: {rouge_scores['rouge-1']}")
            print(f"ROUGE-2: {rouge_scores['rouge-2']}")
            print(f"ROUGE-L: {rouge_scores['rouge-l']}")

            # ChrF Score
            chrf_score = calculate_chrf(kunci_jawaban, response)
            print(f"\n--- ChrF Score ---\n{chrf_score}")

if __name__ == "__main__":
    main()