from rouge_score import rouge_scorer

def calculate_rouge(reference, candidate):
    """
    Menghitung ROUGE scores antara referensi (kunci jawaban) dan kandidat (hasil prediksi).
    
    Args:
        reference (str): Kalimat referensi (kunci jawaban).
        candidate (str): Kalimat kandidat (hasil prediksi).
    
    Returns:
        dict: Skor ROUGE-1, ROUGE-2, dan ROUGE-L.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge-1": scores['rouge1'].fmeasure,
        "rouge-2": scores['rouge2'].fmeasure,
        "rouge-l": scores['rougeL'].fmeasure
    }
