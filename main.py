"""
main.py
-------
Entry point for the Summarization Metrics Evaluation pipeline.

Runs all four phases sequentially:
  Phase 1  → Generate dataset (if not already present)
  Phase 2  → Evaluation (ROUGE + BERTScore + BARTScore)
  Phase 3  → Analysis (correlations + visualizations)
  Phase 4  → Innovation (hybrid metric + error analysis + regression)

Usage:
    # Full pipeline (BARTScore included, takes ~5–10 min on first run):
    python main.py

    # Skip BARTScore for quick testing:
    python main.py --no-bartscore

    # Limit samples:
    python main.py --limit 10

    # Use different BERTScore model:
    python main.py --bertscore-model roberta-large
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Add project root to path ─────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def main(args: argparse.Namespace) -> None:
    """Run the full evaluation pipeline."""
    from data.duc_simulate import generate_dataset
    from src.evaluation import run_evaluation, print_summary
    from src.analysis import run_analysis
    from src.innovation import run_innovation

    print("=" * 65)
    print("  Summarization Metrics Evaluation — DUC 2004")
    print("  Phases: Data → Evaluation → Analysis → Innovation")
    print("=" * 65)

    # ─── Phase 1: Dataset ─────────────────────────────────────────────
    print("\n[Phase 1] Generating DUC 2004 Simulated Dataset...")
    dataset_path = "data/duc_2004_simulated.json"
    if not Path(dataset_path).exists():
        generate_dataset(output_path=dataset_path)
    else:
        print(f"  Dataset already exists at '{dataset_path}'. Skipping generation.")

    # ─── Phase 2: Evaluation ──────────────────────────────────────────
    print("\n[Phase 2] Running Evaluation Pipeline...")
    t_start = time.time()
    df = run_evaluation(
        dataset_path=dataset_path,
        output_csv="results/evaluation_results.csv",
        limit=args.limit,
        use_bartscore=not args.no_bartscore,
        bertscore_model=args.bertscore_model,
    )
    print_summary(df)

    # ─── Phase 3: Analysis ────────────────────────────────────────────
    print("\n[Phase 3] Running Analysis...")
    run_analysis(df, output_dir="results/")

    # ─── Phase 4: Innovation ──────────────────────────────────────────
    print("\n[Phase 4] Running Innovation Modules...")
    df_enriched = run_innovation(df, output_dir="results/")

    # ─── Final Summary ────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print(f"  ✓ Pipeline complete in {elapsed:.1f}s")
    print(f"  Results  → results/evaluation_results.csv")
    print(f"  Enriched → results/results_enriched.csv")
    print(f"  Plots    → results/*.png")
    print("=" * 65)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Summarization Metrics Evaluation — DUC 2004",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--no-bartscore",
        action="store_true",
        help="Skip BARTScore computation (faster, avoids 1.6GB model download).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit evaluation to the first N samples.",
    )
    parser.add_argument(
        "--bertscore-model",
        type=str,
        default="distilbert-base-uncased",
        metavar="MODEL",
        help="HuggingFace model for BERTScore (e.g., roberta-large for best quality).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
