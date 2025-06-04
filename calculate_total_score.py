import csv
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Fungsi untuk membaca file CSV dan menghitung total BLEU score
def calculate_total_bleu_score(csv_file):
    total_bleu_score = 0
    count = 0

    # Membuka file CSV
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        # Pastikan kolom referensi dan prediksi ada di file CSV
        if 'minang_ground_truth' not in reader.fieldnames or 'hasil_terjemahan' not in reader.fieldnames:
            raise ValueError("CSV harus memiliki kolom 'minang_ground_truth' dan 'hasil_terjemahan'")
        
        # Iterasi setiap baris dalam file CSV
        for row in reader:
            reference = row['minang_ground_truth'].strip().split()  # Referensi (tokenized)
            prediction = row['hasil_terjemahan'].strip().split()  # Prediksi (tokenized)
            
            # Menghitung BLEU score untuk setiap pasangan referensi dan prediksi
            bleu_score = sentence_bleu([reference], prediction, smoothing_function=SmoothingFunction().method1)
            total_bleu_score += bleu_score
            count += 1

    # Mengembalikan total BLEU score dan rata-rata BLEU score
    return total_bleu_score, total_bleu_score / count if count > 0 else 0

# Contoh penggunaan
csv_file = 'results/result.csv'  # Ganti dengan nama file CSV Anda
try:
    total_bleu, average_bleu = calculate_total_bleu_score(csv_file)
    print(f"Total BLEU Score: {total_bleu}")
    print(f"Average BLEU Score: {average_bleu}")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")