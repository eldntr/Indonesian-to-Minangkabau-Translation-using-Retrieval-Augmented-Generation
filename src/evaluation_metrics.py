from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF

def calculate_bleu(reference, candidate):
    """
    Menghitung BLEU score antara referensi dan kandidat.
    
    Args:
        reference (str): Kalimat referensi.
        candidate (str): Kalimat kandidat.
    
    Returns:
        float: BLEU score.
    """
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing_function)

def calculate_rouge(reference, candidate):
    """
    Menghitung ROUGE scores antara referensi dan kandidat.
    
    Args:
        reference (str): Kalimat referensi.
        candidate (str): Kalimat kandidat.
    
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

def calculate_chrf(reference, candidate):
    """
    Menghitung ChrF score antara referensi dan kandidat.
    
    Args:
        reference (str): Kalimat referensi.
        candidate (str): Kalimat kandidat.
    
    Returns:
        float: ChrF score.
    """
    chrf = CHRF()
    return chrf.corpus_score([candidate], [[reference]]).score
