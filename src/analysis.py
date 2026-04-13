"""
analysis.py
-----------
Analysis module: correlation analysis and visualization.

Performs:
  - Pearson & Spearman correlation between all metrics
  - Correlation heatmap
  - Scatter plots (pairwise metric comparisons)
  - Distribution plots (histograms per metric)
  - Side-by-side bar chart comparing ROUGE vs BERTScore per sample

Usage:
    from src.analysis import run_analysis
    run_analysis(df, output_dir="results/")
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

# ── Plot style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
METRIC_PALETTE = {
    "rouge1":              "#4C72B0",
    "rouge2":              "#DD8452",
    "rougeL":              "#55A868",
    "bertscore_f1":        "#C44E52",
    "bertscore_precision": "#8172B2",
    "bertscore_recall":    "#937860",
    "bartscore":           "#DA8BC3",
}


# ─────────────────────────────────────────────────────────────────────────────
# Correlation Analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_correlations(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Compute Pearson and Spearman correlation matrices over all numeric columns.

    Args:
        df: Results DataFrame with metric columns.

    Returns:
        Dict with keys 'pearson' and 'spearman', each a correlation DataFrame.
    """
    metric_cols = _get_metric_cols(df)
    pearson  = df[metric_cols].corr(method="pearson")
    spearman = df[metric_cols].corr(method="spearman")
    return {"pearson": pearson, "spearman": spearman}


def print_correlations(corr_dict: dict[str, pd.DataFrame]) -> None:
    """
    Pretty-print Pearson and Spearman correlation matrices.

    Args:
        corr_dict: As returned by compute_correlations().
    """
    for name, matrix in corr_dict.items():
        print(f"\n{name.capitalize()} Correlation Matrix:")
        print(matrix.round(4).to_string())
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_dir: str = "results/",
    method: str = "pearson",
) -> str:
    """
    Save a correlation heatmap of all evaluation metrics.

    Args:
        df: Results DataFrame.
        output_dir: Directory to save the figure.
        method: 'pearson' or 'spearman'.

    Returns:
        Path to the saved figure.
    """
    metric_cols = _get_metric_cols(df)
    corr = df[metric_cols].corr(method=method)

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, True)  # Hide diagonal (always 1.0)

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(f"{method.capitalize()} Correlation Heatmap — Summarization Metrics", pad=12)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    out = _save_fig(fig, output_dir, f"heatmap_{method}.png")
    logger.info("Saved heatmap to '%s'.", out)
    return out


def plot_scatter_matrix(df: pd.DataFrame, output_dir: str = "results/") -> str:
    """
    Save a scatter-plot matrix (pairplot) of all metrics.

    Args:
        df: Results DataFrame.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    metric_cols = _get_metric_cols(df)
    g = sns.pairplot(
        df[metric_cols],
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "edgecolor": "white", "s": 60},
        diag_kws={"fill": True},
    )
    g.fig.suptitle("Scatter Plot Matrix — Summarization Metrics", y=1.01, fontsize=14)
    g.fig.tight_layout()

    out = _save_fig(g.fig, output_dir, "scatter_matrix.png")
    logger.info("Saved scatter matrix to '%s'.", out)
    return out


def plot_distributions(df: pd.DataFrame, output_dir: str = "results/") -> str:
    """
    Save distribution (KDE + histogram) plots for each metric.

    Args:
        df: Results DataFrame.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    metric_cols = _get_metric_cols(df)
    n = len(metric_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(metric_cols):
        color = METRIC_PALETTE.get(col, "#4C72B0")
        sns.histplot(
            df[col],
            ax=axes[i],
            kde=True,
            color=color,
            bins=10,
            alpha=0.7,
            edgecolor="white",
        )
        axes[i].set_title(col, fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Score")
        axes[i].set_ylabel("Count")
        axes[i].axvline(df[col].mean(), color="red", linestyle="--", linewidth=1.2, label=f"Mean={df[col].mean():.3f}")
        axes[i].legend(fontsize=9)

    # Hide any leftover axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Score Distributions — Summarization Metrics", fontsize=15, y=1.02)
    fig.tight_layout()

    out = _save_fig(fig, output_dir, "distributions.png")
    logger.info("Saved distributions to '%s'.", out)
    return out


def plot_rouge_vs_bertscore(df: pd.DataFrame, output_dir: str = "results/") -> str:
    """
    Plot ROUGE-1 vs BERTScore-F1 per sample to reveal disagreement.

    Args:
        df: Results DataFrame.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    if "bertscore_f1" not in df.columns:
        logger.warning("bertscore_f1 not found; skipping ROUGE-vs-BERTScore plot.")
        return ""

    x = range(len(df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    bars1 = ax.bar([i - width/2 for i in x], df["rouge1"],        width, label="ROUGE-1",       color="#4C72B0", alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], df["bertscore_f1"], width, label="BERTScore-F1",  color="#C44E52", alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["id"].str.replace("DUC2004_", "#"), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("ROUGE-1 vs BERTScore-F1 per Sample\n(Gaps indicate semantic vs lexical divergence)", fontsize=13)
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    fig.tight_layout()

    out = _save_fig(fig, output_dir, "rouge_vs_bertscore.png")
    logger.info("Saved ROUGE vs BERTScore chart to '%s'.", out)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis runner
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(df: pd.DataFrame, output_dir: str = "results/") -> dict:
    """
    Run the full analysis suite and save all outputs.

    Args:
        df: Results DataFrame from run_evaluation().
        output_dir: Directory for saving figures.

    Returns:
        Dict containing correlation matrices and figure paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Correlations
    print("\n[Analysis 1/4] Computing correlations...")
    corr = compute_correlations(df)
    print_correlations(corr)

    # Save correlation CSVs
    corr["pearson"].to_csv(f"{output_dir}/pearson_correlation.csv")
    corr["spearman"].to_csv(f"{output_dir}/spearman_correlation.csv")

    # 2. Heatmap
    print("[Analysis 2/4] Plotting correlation heatmap...")
    heatmap_path = plot_correlation_heatmap(df, output_dir, method="pearson")

    # 3. Distributions
    print("[Analysis 3/4] Plotting score distributions...")
    dist_path = plot_distributions(df, output_dir)

    # 4. ROUGE vs BERTScore
    print("[Analysis 4/4] Plotting ROUGE vs BERTScore per sample...")
    rouge_bert_path = plot_rouge_vs_bertscore(df, output_dir)

    # 5. Scatter matrix (can be slow for many metrics)
    scatter_path = plot_scatter_matrix(df, output_dir)

    print("\n[✓] Analysis complete. All plots saved to:", output_dir)

    return {
        "correlations": corr,
        "heatmap": heatmap_path,
        "distributions": dist_path,
        "rouge_vs_bertscore": rouge_bert_path,
        "scatter_matrix": scatter_path,
    }


def _get_metric_cols(df: pd.DataFrame) -> list[str]:
    """Return only numeric metric column names from a DataFrame."""
    skip = {"id", "reference", "system"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def _save_fig(fig: plt.Figure, output_dir: str, filename: str) -> str:
    """Save a matplotlib figure to disk and close it."""
    path = str(Path(output_dir) / filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
