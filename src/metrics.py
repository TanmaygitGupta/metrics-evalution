"""
metrics.py
----------
Metric computation module for summarization evaluation.

Implements:
  - ROUGE-1, ROUGE-2, ROUGE-L  (via rouge-score)
  - BERTScore                   (via bert-score)
  - BARTScore                   (via transformers, facebook/bart-large-cnn)

Each function accepts (hypotheses, references) as lists of strings
and returns a dict of {metric_name: list_of_scores}.

Usage:
    from src.metrics import compute_rouge, compute_bertscore, compute_bartscore
"""

import logging
from typing import Optional

import torch
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from transformers import BartForConditionalGeneration, BartTokenizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ROUGE
# ---------------------------------------------------------------------------

def compute_rouge(
    hypotheses: list[str],
    references: list[str],
    metrics: tuple[str, ...] = ("rouge1", "rouge2", "rougeL"),
) -> dict[str, list[float]]:
    """
    Compute ROUGE scores for each hypothesis-reference pair.

    Args:
        hypotheses: List of system-generated summaries.
        references: List of reference summaries.
        metrics: ROUGE variants to compute.

    Returns:
        Dict mapping metric name to list of F1 scores (one per sample).
    """
    scorer = rouge_scorer.RougeScorer(list(metrics), use_stemmer=True)
    results: dict[str, list[float]] = {m: [] for m in metrics}

    for hyp, ref in zip(hypotheses, references):
        scores = scorer.score(ref, hyp)
        for m in metrics:
            results[m].append(round(scores[m].fmeasure, 4))

    logger.info("ROUGE computed for %d samples.", len(hypotheses))
    return results


# ---------------------------------------------------------------------------
# BERTScore
# ---------------------------------------------------------------------------

def compute_bertscore(
    hypotheses: list[str],
    references: list[str],
    model_type: str = "distilbert-base-uncased",
    batch_size: int = 8,
    device: Optional[str] = None,
) -> dict[str, list[float]]:
    """
    Compute BERTScore (Precision, Recall, F1) for each pair.

    Uses distilbert-base-uncased by default for speed on CPU.
    Switch to 'roberta-large' for research-grade accuracy.

    Args:
        hypotheses: System summaries.
        references: Reference summaries.
        model_type: HuggingFace model to use for BERTScore.
        batch_size: Batch size for encoding.
        device: 'cuda' or 'cpu'. Auto-detected if None.

    Returns:
        Dict with keys: bertscore_precision, bertscore_recall, bertscore_f1.
        Each value is a list of floats.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Computing BERTScore on device='%s', model='%s'.", device, model_type)

    P, R, F1 = bert_score_fn(
        cands=hypotheses,
        refs=references,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        verbose=False,
    )

    return {
        "bertscore_precision": [round(p.item(), 4) for p in P],
        "bertscore_recall":    [round(r.item(), 4) for r in R],
        "bertscore_f1":        [round(f.item(), 4) for f in F1],
    }


# ---------------------------------------------------------------------------
# BARTScore
# ---------------------------------------------------------------------------

class BARTScorer:
    """
    BARTScore implementation using facebook/bart-large-cnn.

    BARTScore measures the average log-likelihood assigned by a pre-trained
    BART model to the hypothesis given the reference (and vice versa).

    Reference: Yuan et al. (2021) — https://arxiv.org/abs/2106.11520
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize BARTScorer.

        Args:
            model_name: Pre-trained BART model to use.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading BARTScore model '%s' on '%s'.", model_name, self.device)

        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def score(
        self,
        hypotheses: list[str],
        references: list[str],
        max_length: int = 1024,
    ) -> list[float]:
        """
        Compute BARTScore for each (hypothesis, reference) pair.

        Score = average token log-likelihood of hypothesis given reference.

        Args:
            hypotheses: System summaries (candidates).
            references: Human reference summaries.
            max_length: Max token length.

        Returns:
            List of BARTScore floats (higher is better; typically negative).
        """
        scores = []
        for hyp, ref in zip(hypotheses, references):
            src_tokens = self.tokenizer(
                ref,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).to(self.device)

            tgt_tokens = self.tokenizer(
                hyp,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                output = self.model(
                    input_ids=src_tokens["input_ids"],
                    attention_mask=src_tokens["attention_mask"],
                    labels=tgt_tokens["input_ids"],
                )
            # Normalize by sequence length to get per-token log-likelihood
            loss = output.loss.item()
            scores.append(round(-loss, 4))  # Negate loss → higher is better

        logger.info("BARTScore computed for %d samples.", len(hypotheses))
        return scores


def compute_bartscore(
    hypotheses: list[str],
    references: list[str],
    model_name: str = "facebook/bart-large-cnn",
) -> dict[str, list[float]]:
    """
    Convenience wrapper for BARTScorer.

    Args:
        hypotheses: System summaries.
        references: Reference summaries.
        model_name: BART model name.

    Returns:
        Dict with key 'bartscore' mapping to list of scores.
    """
    scorer = BARTScorer(model_name=model_name)
    return {"bartscore": scorer.score(hypotheses, references)}
