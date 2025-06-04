from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize
import nltk
from sacrebleu.metrics import CHRF
import pyter

# Pastikan data NLTK yang diperlukan telah diunduh
# Jalankan sekali di terminal Python Anda:
# >>> import nltk
# >>> nltk.download('punkt')
# >>> nltk.download('wordnet')
# >>> nltk.download('omw-1.4')


def calculate_bleu(reference, candidate):
    """
    Menghitung BLEU score antara referensi dan kandidat.
    """
    reference_tokens = [reference.split()]
    candidate_tokens = candidate.split()
    # Menggunakan smoothing function untuk menangani n-gram yang tidak ada di referensi
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothing_function)


def calculate_meteor(reference, candidate):
    """
    Menghitung METEOR score antara referensi dan kandidat.
    METEOR memerlukan tokenisasi.
    """
    # Tokenisasi diperlukan untuk METEOR agar dapat mencocokkan kata dasar dan sinonim
    reference_tokens = word_tokenize(reference)
    candidate_tokens = word_tokenize(candidate)
    return single_meteor_score(reference_tokens, candidate_tokens)


def calculate_ter(reference, candidate):
    """
    Menghitung Translation Edit Rate (TER) score.
    Skor yang lebih rendah lebih baik.
    """
    reference_tokens = reference.split()
    candidate_tokens = candidate.split()
    try:
        # Menghitung TER score
        ter_score = pyter.ter(candidate_tokens, reference_tokens)
        return ter_score
    except Exception as e:
        return f"Error calculating TER: {e}"


def calculate_chrf(reference, candidate):
    """
    Menghitung ChrF score antara referensi dan kandidat.
    """
    chrf = CHRF()
    # sacrebleu mengharapkan kandidat dalam list dan referensi dalam list of list
    return chrf.corpus_score([candidate], [[reference]]).score
