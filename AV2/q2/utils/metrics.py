from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score

def compute_bleu(expected, generated):
    """Calcula BLEU-1 / BLEU-2 simplificado para respostas curtas."""
    smoothie = SmoothingFunction().method1
    reference = expected.split()
    candidate = generated.split()

    if len(candidate) == 0:
        return 0.0

    bleu = sentence_bleu([reference], candidate, smoothing_function=smoothie)
    return float(bleu)


def compute_bertscore(expected, generated):
    """Calcula BERTScore (F1)."""
    P, R, F1 = bert_score([generated], [expected], lang="pt", verbose=False)
    return float(F1[0])