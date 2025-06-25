import pandas as pd
import sacrebleu
import sys

def calculate_corpus_metrics(csv_file_path):
    """
    Menghitung ulang metrik evaluasi (BLEU, CHRF++, TER, spBLEU) untuk seluruh korpus
    dari file CSV hasil evaluasi.

    Args:
        csv_file_path (str): Path menuju file .csv yang berisi hasil.
    """
    try:
        # 1. Membaca file CSV menggunakan pandas
        df = pd.read_csv(csv_file_path)
        print(f"✅ Berhasil memuat {len(df)} baris dari '{csv_file_path}'")
    except FileNotFoundError:
        print(f"❌ Error: File tidak ditemukan di '{csv_file_path}'", file=sys.stderr)
        return

    # 2. Memastikan kolom yang dibutuhkan ada
    required_columns = ['hasil_terjemahan', 'minang_ground_truth']
    if not all(col in df.columns for col in required_columns):
        print(f"❌ Error: CSV harus memiliki kolom {required_columns}", file=sys.stderr)
        return
        
    # 3. Menyiapkan daftar hipotesis dan referensi
    # sacrebleu mengharapkan referensi dalam format list of lists,
    # karena satu hipotesis bisa memiliki beberapa referensi.
    hypotheses = df['hasil_terjemahan'].astype(str).tolist()
    references = [df['minang_ground_truth'].astype(str).tolist()] # Perhatikan format [[ref1, ref2, ...]]

    print("\n🔄 Menghitung ulang skor untuk seluruh korpus...")

    # 4. Menghitung metrik menggunakan sacrebleu
    # .score memberikan skor dalam skala 0-100 untuk BLEU dan TER
    # .score untuk CHRF++ sudah dalam skala 0-100 secara default
    
    # Menghitung BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    
    # Menghitung CHRF++ (default dari sacrebleu.corpus_chrf adalah CHRF++)
    chrf = sacrebleu.corpus_chrf(hypotheses, references)
    
    # Menghitung TER
    ter = sacrebleu.corpus_ter(hypotheses, references)

    # Menghitung spBLEU (sentence-level BLEU)
    spbleu_scores = []
    for hyp, ref in zip(hypotheses, references[0]):
        spbleu = sacrebleu.sentence_bleu(hyp, [ref])
        spbleu_scores.append(spbleu.score)
    avg_spbleu = sum(spbleu_scores) / len(spbleu_scores)

    # 5. Mencetak hasil
    print("\n--- 📊 Hasil Perhitungan Ulang Metrik Korpus 📊 ---")
    print(f"🔵 BLEU Score   : {bleu.score:.2f}")
    print(f"⚪️ CHRF++ Score : {chrf.score:.2f}")
    print(f"🔴 TER Score    : {ter.score:.2f} (Translation Edit Rate, lebih rendah lebih baik)")
    print(f"🟢 spBLEU Score : {avg_spbleu:.2f} (Rata-rata BLEU per kalimat)")
    print("-------------------------------------------------")
    print("\nCatatan: METEOR tidak dihitung ulang karena memerlukan setup NLTK dan ")
    print("perhitungan corpus-levelnya tidak distandarisasi di sacrebleu.")


# --- JALANKAN SCRIPT ---
if __name__ == "__main__":
    # Ganti dengan nama file CSV Anda yang sebenarnya
    # Anggap file CSV Anda bernama 'sample_results.csv' dan berada di folder yang sama
    csv_file = 'translation_results_with_scores (1).csv'
    calculate_corpus_metrics(csv_file)