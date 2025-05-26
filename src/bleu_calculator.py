from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(reference, candidate):
    """
    Menghitung BLEU score antara referensi (kunci jawaban) dan kandidat (hasil prediksi).
    
    Args:
        reference (str): Kalimat referensi (kunci jawaban).
        candidate (str): Kalimat kandidat (hasil prediksi).
    
    Returns:
        float: BLEU score.
    """
    reference_tokens = [reference.split()]  # Tokenisasi referensi
    candidate_tokens = candidate.split()    # Tokenisasi kandidat
    smoothing_function = SmoothingFunction().method1  # Gunakan smoothing
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing_function)
