import pandas as pd
from tqdm import tqdm
import time
from dotenv import load_dotenv
import os
import numpy as np

# Impor modul dan fungsi yang sudah ada dari proyek Anda
from src.retriever import SemanticRetriever
from src.utils import generate_translation_prompt
from src.llm_handler import send_prompt_to_llm
from src.evaluation_metrics import calculate_bleu, calculate_ter, calculate_chrf, calculate_meteor
import config

# --- KONFIGURASI ---
# Path ke dataset pengujian. Pastikan file ini ada dan memiliki kolom 'indonesia' dan 'minang'.
TEST_DATA_PATH = 'dataset/test.csv'

# Path untuk file output
OUTPUT_DIR = 'results'
RESULT_CSV_PATH = os.path.join(OUTPUT_DIR, 'result.csv')
EVALUATION_SUMMARY_PATH = os.path.join(OUTPUT_DIR, 'total_evaluation.txt')

# Pengaturan untuk penanganan API
MAX_RETRIES = 3  # Jumlah maksimum percobaan ulang jika API gagal
RETRY_DELAY_SECONDS = 5  # Waktu tunggu (detik) sebelum mencoba lagi
REQUEST_DELAY_SECONDS = 1  # Jeda antar permintaan untuk menghindari rate limiting

def process_and_evaluate_corpus():
    """
    Fungsi untuk memproses seluruh data dari CSV, menerjemahkan, mengevaluasi,
    dan menyimpan hasilnya.
    """
    # Muat variabel lingkungan
    load_dotenv()
    
    print("--- Memulai Proses Evaluasi Korpus ---")

    # Pastikan direktori output ada
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Inisialisasi Retriever
    try:
        retriever = SemanticRetriever(
            model_name=config.MODEL_NAME,
            csv_file_path=config.CSV_FILE_PATH  # Menggunakan korpus train untuk retriever
        )
        print("Semantic Retriever berhasil diinisialisasi.")
    except Exception as e:
        print(f"Gagal menginisialisasi retriever: {e}")
        return

    # 2. Muat data uji
    try:
        df_test = pd.read_csv(TEST_DATA_PATH)
        # Pastikan kolom yang diperlukan ada
        if 'indonesian' not in df_test.columns or 'minangkabau' not in df_test.columns:
            print(f"Error: File '{TEST_DATA_PATH}' harus memiliki kolom 'indonesia' dan 'minang'.")
            return
        print(f"Berhasil memuat {len(df_test)} baris data dari '{TEST_DATA_PATH}'.")
    except FileNotFoundError:
        print(f"Error: File data uji tidak ditemukan di '{TEST_DATA_PATH}'.")
        return

    # 3. Proses setiap baris data
    results_list = []
    
    print("\n--- Memulai Proses Penerjemahan dan Evaluasi ---")
    try:
        for index, row in tqdm(df_test.iterrows(), total=df_test.shape[0], desc="Menerjemahkan"):
            query_pengguna = row['indonesian']
            kunci_jawaban = row['minangkabau']
            
            terjemahan = None
            for attempt in range(MAX_RETRIES):
                try:
                    # Langkah A: Dapatkan contoh relevan dari korpus training
                    hasil_pencarian = retriever.retrieve(query_pengguna, similarity_threshold=config.SIMILARITY_THRESHOLD)
                    
                    list_data_untuk_prompt = [
                        {"original_query_word": kata_query, **data_hasil}
                        for kata_query, data_hasil in hasil_pencarian.items()
                    ]

                    # Langkah B: Buat prompt
                    prompt_final = generate_translation_prompt(query_pengguna, list_data_untuk_prompt)

                    # Langkah C: Kirim ke LLM
                    response = send_prompt_to_llm(prompt_final)
                    
                    if response:
                        terjemahan = response
                        break  # Berhasil, keluar dari loop retry
                    else:
                        print(f"\nPercobaan {attempt + 1} gagal untuk baris {index}: Menerima respons kosong dari LLM.")
                        
                except Exception as e:
                    print(f"\nError pada baris {index}, percobaan {attempt + 1}/{MAX_RETRIES}: {e}")

                # Tunggu sebelum mencoba lagi
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY_SECONDS)
            
            # Jika semua percobaan gagal, catat sebagai error
            if terjemahan is None:
                print(f"Gagal memproses baris {index} setelah {MAX_RETRIES} percobaan.")
                terjemahan = "ERROR_TRANSLATION"

            # 4. Hitung skor evaluasi
            bleu_score = calculate_bleu(kunci_jawaban, terjemahan)
            meteor_score = calculate_meteor(kunci_jawaban, terjemahan)
            ter_score = calculate_ter(kunci_jawaban, terjemahan)
            chrf_score = calculate_chrf(kunci_jawaban, terjemahan)
            
            # 5. Simpan hasil untuk baris ini
            results_list.append({
                'indonesia': query_pengguna,
                'minang_ground_truth': kunci_jawaban,
                'hasil_terjemahan': terjemahan,
                'bleu_score': bleu_score,
                'meteor_score': meteor_score,
                'ter_score': ter_score,
                'chrf_score': chrf_score
            })
            
            # Update CSV and evaluation summary after each row
            df_results = pd.DataFrame(results_list)
            df_results.to_csv(RESULT_CSV_PATH, index=False, encoding='utf-8')
            
            avg_bleu = df_results['bleu_score'].mean()
            avg_meteor = df_results['meteor_score'].mean()
            avg_ter = df_results['ter_score'].mean()
            avg_chrf = df_results['chrf_score'].mean()
            
            summary_text = (
                "--- Rangkuman Total Evaluasi ---\n\n"
                f"Jumlah data yang dievaluasi: {len(df_results)}\n\n"
                f"Rata-rata BLEU Score     : {avg_bleu:.4f} (Semakin tinggi semakin baik)\n"
                f"Rata-rata METEOR Score   : {avg_meteor:.4f} (Semakin tinggi semakin baik)\n"
                f"Rata-rata TER Score      : {avg_ter:.4f} (Semakin RENDAH semakin baik)\n"
                f"Rata-rata ChrF Score     : {avg_chrf:.4f} (Semakin tinggi semakin baik)\n"
            )

            with open(EVALUATION_SUMMARY_PATH, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            
            # Jeda antar permintaan untuk menghindari rate limit API
            time.sleep(REQUEST_DELAY_SECONDS)
    except KeyboardInterrupt:
        print("\nProses dihentikan oleh pengguna. Menyimpan hasil sementara...")
        if results_list:
            df_results = pd.DataFrame(results_list)
            df_results.to_csv(RESULT_CSV_PATH, index=False, encoding='utf-8')
            print(f"Hasil sementara disimpan di: {RESULT_CSV_PATH}")
            
            avg_bleu = df_results['bleu_score'].mean()
            avg_meteor = df_results['meteor_score'].mean()
            avg_ter = df_results['ter_score'].mean()
            avg_chrf = df_results['chrf_score'].mean()
            
            summary_text = (
                "--- Rangkuman Total Evaluasi ---\n\n"
                f"Jumlah data yang dievaluasi: {len(df_results)}\n\n"
                f"Rata-rata BLEU Score     : {avg_bleu:.4f} (Semakin tinggi semakin baik)\n"
                f"Rata-rata METEOR Score   : {avg_meteor:.4f} (Semakin tinggi semakin baik)\n"
                f"Rata-rata TER Score      : {avg_ter:.4f} (Semakin RENDAH semakin baik)\n"
                f"Rata-rata ChrF Score     : {avg_chrf:.4f} (Semakin tinggi semakin baik)\n"
            )

            with open(EVALUATION_SUMMARY_PATH, 'w', encoding='utf-8') as f:
                f.write(summary_text)
            print(f"Rangkuman sementara disimpan di: {EVALUATION_SUMMARY_PATH}")
        return

    print("\n--- Proses Selesai ---")

if __name__ == "__main__":
    process_and_evaluate_corpus()