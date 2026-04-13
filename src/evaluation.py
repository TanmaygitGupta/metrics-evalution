"""
evaluation.py
-------------
Orchestrates the full evaluation pipeline over the DUC 2004 dataset.

Steps:
  1. Load dataset
  2. Compute ROUGE, BERTScore, BARTScore for each sample
  3. Aggregate results into a pandas DataFrame
  4. Save results to CSV

Usage:
    from src.evaluation import run_evaluation
    df = run_evaluation()
"""

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.loader import load_dataset, get_pairs
from src.metrics import compute_rouge, compute_bertscore, compute_bartscore

logger = logging.getLogger(__name__)


def run_evaluation(
    dataset_path: str = "data/duc_2004_simulated.json",
    output_csv: str = "results/evaluation_results.csv",
    limit: Optional[int] = None,
    use_bartscore: bool = True,
    bertscore_model: str = "distilbert-base-uncased",
) -> pd.DataFrame:
    """
    Run the full evaluation pipeline and return results as a DataFrame.

    Args:
        dataset_path: Path to the JSON dataset file.
        output_csv: Where to save the CSV output.
        limit: Max samples to evaluate (None = all).
        use_bartscore: Whether to compute BARTScore (slow on CPU, ~2 min).
        bertscore_model: HuggingFace model for BERTScore.

    Returns:
        DataFrame with columns: id, reference, system, rouge1, rouge2,
        rougeL, bertscore_precision, bertscore_recall, bertscore_f1,
        bartscore (if enabled).
    """
    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    logger.info("Loading dataset from '%s'.", dataset_path)
    data = load_dataset(dataset_path)
    ids, references, systems = get_pairs(data, limit=limit)
    n = len(ids)
    logger.info("Evaluating %d samples.", n)

    records = {"id": ids, "reference": references, "system": systems}

    # ------------------------------------------------------------------ #
    # 2. ROUGE
    # ------------------------------------------------------------------ #
    print(f"\n[1/3] Computing ROUGE scores for {n} samples...")
    t0 = time.time()
    rouge_scores = compute_rouge(systems, references)
    records.update(rouge_scores)
    print(f"      Done in {time.time()-t0:.1f}s.")

    # ------------------------------------------------------------------ #
    # 3. BERTScore
    # ------------------------------------------------------------------ #
    print(f"[2/3] Computing BERTScore ({bertscore_model}) for {n} samples...")
    t0 = time.time()
    bert_scores = compute_bertscore(systems, references, model_type=bertscore_model)
    records.update(bert_scores)
    print(f"      Done in {time.time()-t0:.1f}s.")

    # ------------------------------------------------------------------ #
    # 4. BARTScore (optional)
    # ------------------------------------------------------------------ #
    if use_bartscore:
        print(f"[3/3] Computing BARTScore (facebook/bart-large-cnn) for {n} samples...")
        print("      Note: First run downloads ~1.6GB model. May take several minutes.")
        t0 = time.time()
        bart_scores = compute_bartscore(systems, references)
        records.update(bart_scores)
        print(f"      Done in {time.time()-t0:.1f}s.")
    else:
        print("[3/3] BARTScore skipped (use_bartscore=False).")

    # ------------------------------------------------------------------ #
    # 5. Build DataFrame & save
    # ------------------------------------------------------------------ #
    df = pd.DataFrame(records)

    # Round all numeric columns to 4 decimal places for readability
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(4)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n[✓] Results saved to '{output_csv}'.")

    return df


def print_summary(df: pd.DataFrame) -> None:
    """
    Print a human-readable summary of evaluation results.

    Args:
        df: Results DataFrame as returned by run_evaluation().
    """
    metric_cols = [c for c in df.columns if c not in ("id", "reference", "system")]
    print(f"\n{'='*60}")
    print(f"  Evaluation Summary ({len(df)} samples)")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*58}")
    for col in metric_cols:
        mean = df[col].mean()
        std  = df[col].std()
        mn   = df[col].min()
        mx   = df[col].max()
        print(f"  {col:<30} {mean:>8.4f} {std:>8.4f} {mn:>8.4f} {mx:>8.4f}")
    print(f"{'='*60}\n")
