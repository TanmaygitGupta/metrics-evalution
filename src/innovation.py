"""
innovation.py
-------------
Innovation Phase: two complementary improvements over baseline metrics.

INNOVATION A — Hybrid Metric
    Final Score = α * ROUGE-1 + β * ROUGE-2 + γ * BERTScore-F1 [+ δ * BARTScore_normalized]
    Weights are tunable; defaults are set from empirical analysis.
    The hybrid is robust to both lexical and semantic variation.

INNOVATION B — Metric Disagreement / Error Analysis
    Identifies samples where ROUGE and BERTScore significantly disagree,
    categorizing them as:
      - "Semantic Match" (high BERTScore, low ROUGE)   → ROUGE failure
      - "Lexical Match"  (high ROUGE, low BERTScore)   → BERTScore under-penalizes noise
      - "Agreement-High" / "Agreement-Low"

INNOVATION C — Regression Model
    Learns to predict a composite "pseudo-human" score from metric outputs
    using Ridge regression. Also provides feature importance analysis.

Usage:
    from src.innovation import compute_hybrid_metric, error_analysis, train_regression_model
"""

import logging
from typing import Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=1.1)


# ─────────────────────────────────────────────────────────────────────────────
# Innovation A: Hybrid Metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_hybrid_metric(
    df: pd.DataFrame,
    alpha: float = 0.25,   # ROUGE-1 weight
    beta:  float = 0.15,   # ROUGE-2 weight
    gamma: float = 0.60,   # BERTScore-F1 weight
    delta: float = 0.10,   # BARTScore weight (if available)
) -> pd.DataFrame:
    """
    Compute a weighted hybrid evaluation score.

    Formula:
        If bartscore present:
            hybrid = α*rouge1 + β*rouge2 + γ*bertscore_f1 + δ*bartscore_norm
        Else:
            hybrid = α'*rouge1 + β'*rouge2 + γ'*bertscore_f1
            (weights renormalized to sum to 1)

    BARTScore is normalized to [0,1] via min-max scaling before combining.

    Args:
        df: Results DataFrame from run_evaluation().
        alpha, beta, gamma, delta: Metric weights.

    Returns:
        DataFrame with an additional 'hybrid_score' column.
    """
    df = df.copy()
    has_bart = "bartscore" in df.columns

    if has_bart:
        # Normalize BARTScore to [0,1]
        scaler = MinMaxScaler()
        df["bartscore_norm"] = scaler.fit_transform(df[["bartscore"]])
        total = alpha + beta + gamma + delta
        df["hybrid_score"] = (
            alpha / total * df["rouge1"] +
            beta  / total * df["rouge2"] +
            gamma / total * df["bertscore_f1"] +
            delta / total * df["bartscore_norm"]
        ).round(4)
    else:
        total = alpha + beta + gamma
        df["hybrid_score"] = (
            alpha / total * df["rouge1"] +
            beta  / total * df["rouge2"] +
            gamma / total * df["bertscore_f1"]
        ).round(4)

    logger.info("Hybrid metric computed for %d samples (bartscore=%s).", len(df), has_bart)
    return df


def plot_hybrid_comparison(df: pd.DataFrame, output_dir: str = "results/") -> str:
    """
    Bar chart comparing hybrid_score vs individual metrics per sample.

    Args:
        df: DataFrame with hybrid_score column.
        output_dir: Where to save the figure.

    Returns:
        Path to the saved figure.
    """
    x = range(len(df))
    width = 0.20

    cols_to_plot = ["rouge1", "bertscore_f1", "hybrid_score"]
    if "bartscore_norm" in df.columns:
        cols_to_plot.insert(2, "bartscore_norm")

    colors = ["#4C72B0", "#C44E52", "#DA8BC3", "#55A868"]
    offsets = np.linspace(-width * len(cols_to_plot) / 2, width * len(cols_to_plot) / 2, len(cols_to_plot))

    fig, ax = plt.subplots(figsize=(16, 5))
    for i, (col, color, offset) in enumerate(zip(cols_to_plot, colors, offsets)):
        ax.bar(
            [xi + offset for xi in x],
            df[col],
            width,
            label=col,
            color=color,
            alpha=0.85,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["id"].str.replace("DUC2004_", "#"), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("Hybrid Score vs Individual Metrics per Sample", fontsize=13)
    ax.legend()
    fig.tight_layout()

    path = str(Path(output_dir) / "hybrid_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Innovation B: Error Analysis / Disagreement Detection
# ─────────────────────────────────────────────────────────────────────────────

def error_analysis(
    df: pd.DataFrame,
    rouge_col: str = "rouge1",
    bert_col: str = "bertscore_f1",
    threshold: float = 0.10,
    output_dir: str = "results/",
) -> pd.DataFrame:
    """
    Identify and categorize samples where ROUGE and BERTScore disagree.

    Categories:
        - "Semantic Match"  : BERTScore high, ROUGE low  → paraphrasing caught by BERT but missed by ROUGE
        - "Lexical Match"   : ROUGE high, BERTScore low  → lexical overlap without semantic match
        - "Agreement-High"  : Both metrics agree & high
        - "Agreement-Low"   : Both metrics agree & low
        - "Mixed"           : Neither strong disagreement pattern

    Args:
        df: Results DataFrame.
        rouge_col: ROUGE metric column to use.
        bert_col: BERTScore column to use.
        threshold: Minimum score delta to declare disagreement.
        output_dir: Where to save the disagreement scatter plot.

    Returns:
        DataFrame with added 'disagreement_category' and 'score_delta' columns.
    """
    df = df.copy()
    df["score_delta"] = (df[bert_col] - df[rouge_col]).round(4)

    mid_rouge = df[rouge_col].median()
    mid_bert  = df[bert_col].median()

    def _classify(row):
        delta = row["score_delta"]
        rouge = row[rouge_col]
        bert  = row[bert_col]
        if delta > threshold and bert > mid_bert:
            return "Semantic Match (ROUGE fails)"
        elif delta < -threshold and rouge > mid_rouge:
            return "Lexical Match (BERT over-credits)"
        elif rouge > mid_rouge and bert > mid_bert:
            return "Agreement-High"
        elif rouge < mid_rouge and bert < mid_bert:
            return "Agreement-Low"
        else:
            return "Mixed"

    df["disagreement_category"] = df.apply(_classify, axis=1)

    # Print summary
    print("\n[Error Analysis] Disagreement Categories:")
    print(df.groupby("disagreement_category")["id"].count().rename("count").to_string())

    # Scatter plot
    _plot_disagreement_scatter(df, rouge_col, bert_col, output_dir)
    return df


def _plot_disagreement_scatter(
    df: pd.DataFrame,
    rouge_col: str,
    bert_col: str,
    output_dir: str,
) -> None:
    """Create and save scatter plot colored by disagreement category."""
    palette = {
        "Semantic Match (ROUGE fails)":       "#C44E52",
        "Lexical Match (BERT over-credits)":  "#4C72B0",
        "Agreement-High":                     "#55A868",
        "Agreement-Low":                      "#DD8452",
        "Mixed":                              "#9B9B9B",
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    for category, grp in df.groupby("disagreement_category"):
        ax.scatter(
            grp[rouge_col],
            grp[bert_col],
            label=category,
            color=palette.get(category, "#9B9B9B"),
            s=100,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )

    # Annotate sample IDs
    for _, row in df.iterrows():
        ax.annotate(
            row["id"].replace("DUC2004_", "#"),
            (row[rouge_col], row[bert_col]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
            color="gray",
        )

    # Diagonal reference line (perfect agreement)
    lims = [
        min(df[rouge_col].min(), df[bert_col].min()) - 0.05,
        max(df[rouge_col].max(), df[bert_col].max()) + 0.05,
    ]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.4, label="Perfect Agreement")

    ax.set_xlabel(f"ROUGE-1 Score", fontsize=12)
    ax.set_ylabel(f"BERTScore-F1", fontsize=12)
    ax.set_title("Metric Disagreement Analysis\n(ROUGE-1 vs BERTScore-F1)", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    path = str(Path(output_dir) / "disagreement_analysis.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved disagreement scatter to '%s'.", path)


# ─────────────────────────────────────────────────────────────────────────────
# Innovation C: Regression Model
# ─────────────────────────────────────────────────────────────────────────────

def train_regression_model(
    df: pd.DataFrame,
    target_col: str = "hybrid_score",
    feature_cols: Optional[list[str]] = None,
    output_dir: str = "results/",
) -> dict:
    """
    Train a Ridge regression model to predict a target score from metric features.

    Uses Leave-One-Out cross-validation (suitable for small datasets).

    Args:
        df: DataFrame with metric columns (must include 'hybrid_score').
        target_col: Column to predict (default: hybrid_score as proxy).
        feature_cols: Input metric columns (default: rouge1, rouge2, rougeL, bertscore_*).
        output_dir: Where to save the feature importance plot.

    Returns:
        Dict with keys: model, scaler, loo_r2, loo_mse, predictions, feature_importances.
    """
    if feature_cols is None:
        candidates = ["rouge1", "rouge2", "rougeL",
                      "bertscore_precision", "bertscore_recall", "bertscore_f1"]
        if "bartscore" in df.columns:
            candidates.append("bartscore")
        feature_cols = [c for c in candidates if c in df.columns]

    X = df[feature_cols].values
    y = df[target_col].values

    # Feature scaling
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Leave-One-Out CV
    loo = LeaveOneOut()
    model = Ridge(alpha=1.0)
    y_pred_loo = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X_scaled):
        model.fit(X_scaled[train_idx], y[train_idx])
        y_pred_loo[test_idx] = model.predict(X_scaled[test_idx])

    # Final model on all data
    model.fit(X_scaled, y)

    # Metrics
    ss_res = np.sum((y - y_pred_loo) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    loo_r2  = round(1 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0
    loo_mse = round(np.mean((y - y_pred_loo) ** 2), 6)

    print(f"\n[Regression Model]")
    print(f"  Features         : {feature_cols}")
    print(f"  Target           : {target_col}")
    print(f"  LOO R²           : {loo_r2:.4f}")
    print(f"  LOO MSE          : {loo_mse:.6f}")

    feature_importances = dict(zip(feature_cols, model.coef_.round(4)))
    print(f"  Feature weights  : {feature_importances}")

    # Feature importance plot
    _plot_feature_importance(feature_importances, output_dir)

    # Predicted vs Actual plot
    _plot_predicted_vs_actual(y, y_pred_loo, output_dir)

    return {
        "model":                model,
        "scaler":               scaler,
        "loo_r2":               loo_r2,
        "loo_mse":              loo_mse,
        "predictions":          y_pred_loo.round(4).tolist(),
        "feature_importances":  feature_importances,
    }


def _plot_feature_importance(importances: dict, output_dir: str) -> None:
    """Bar chart of ridge regression feature coefficients."""
    fig, ax = plt.subplots(figsize=(8, 4))
    features = list(importances.keys())
    values   = list(importances.values())
    colors   = ["#C44E52" if v < 0 else "#55A868" for v in values]

    ax.barh(features, values, color=colors, alpha=0.85, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Ridge Coefficient")
    ax.set_title("Regression Model — Feature Importance", fontsize=13)
    fig.tight_layout()

    path = str(Path(output_dir) / "feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_predicted_vs_actual(y_true, y_pred, output_dir: str) -> None:
    """Scatter plot of predicted vs actual hybrid scores (LOO)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, color="#4C72B0", s=70, alpha=0.8, edgecolors="white")

    lims = [min(y_true.min(), y_pred.min()) - 0.02, max(y_true.max(), y_pred.max()) + 0.02]
    ax.plot(lims, lims, "r--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score (LOO)")
    ax.set_title("Regression: Predicted vs Actual\n(Leave-One-Out CV)", fontsize=12)
    ax.legend()
    fig.tight_layout()

    path = str(Path(output_dir) / "predicted_vs_actual.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main innovation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_innovation(df: pd.DataFrame, output_dir: str = "results/") -> pd.DataFrame:
    """
    Run all three innovations and save outputs.

    Args:
        df: Results DataFrame from run_evaluation().
        output_dir: Directory for saving outputs.

    Returns:
        Enriched DataFrame with hybrid_score and disagreement_category.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Innovation A: Hybrid Metric
    print("\n[Innovation A] Computing Hybrid Metric...")
    df = compute_hybrid_metric(df)
    hybrid_path = plot_hybrid_comparison(df, output_dir)
    print(f"[✓] Hybrid scores computed. Plot saved: {hybrid_path}")

    # Innovation B: Error Analysis
    print("\n[Innovation B] Running Error / Disagreement Analysis...")
    df = error_analysis(df, output_dir=output_dir)
    print("[✓] Disagreement analysis complete.")

    # Innovation C: Regression Model
    print("\n[Innovation C] Training Regression Model...")
    reg_results = train_regression_model(df, output_dir=output_dir)
    df["predicted_hybrid"] = reg_results["predictions"]
    print("[✓] Regression model trained.")

    # Save enriched CSV
    enriched_path = str(Path(output_dir) / "results_enriched.csv")
    df.to_csv(enriched_path, index=False)
    print(f"\n[✓] Enriched results saved to '{enriched_path}'.")

    return df