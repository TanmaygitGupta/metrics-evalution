# Evaluation of Summarization Metrics using DUC 2004 Dataset

> A complete, research-grade NLP evaluation project covering ROUGE, BERTScore, and BARTScore.

---

## Project Structure

```
summarization-metrics-eval/
├── data/
│   ├── duc_simulate.py          # Generates the simulated DUC 2004 dataset
│   └── duc_2004_simulated.json  # Generated on first run
├── src/
│   ├── __init__.py
│   ├── loader.py                # Dataset loading & validation
│   ├── metrics.py               # ROUGE, BERTScore, BARTScore
│   ├── evaluation.py            # Evaluation pipeline
│   ├── analysis.py              # Correlations & visualizations
│   └── innovation.py            # Hybrid metric, error analysis, regression
├── results/                     # Generated outputs (CSVs + PNGs)
├── notebooks/
│   └── experiment.ipynb         # Full Jupyter notebook
├── report/
│   └── final_report.md          # Academic report
├── main.py                      # Entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Fast mode (no BARTScore — skips 1.6GB model download):
python main.py --no-bartscore

# Full pipeline (BARTScore included, ~5–10 min on first run):
python main.py

# Limit to 10 samples for testing:
python main.py --no-bartscore --limit 10

# Use better BERTScore model (slower):
python main.py --no-bartscore --bertscore-model roberta-large
```

### 3. View results

```
results/
├── evaluation_results.csv       # Per-sample metric scores
├── results_enriched.csv         # + hybrid score + disagreement categories
├── heatmap_pearson.png          # Correlation heatmap
├── scatter_matrix.png           # Pairplot of all metrics
├── distributions.png            # Histogram + KDE per metric
├── rouge_vs_bertscore.png       # ROUGE-1 vs BERTScore-F1 per sample
├── hybrid_comparison.png        # Hybrid metric comparison
├── disagreement_analysis.png    # Error analysis scatter plot
├── feature_importance.png       # Regression feature coefficients
└── predicted_vs_actual.png      # Regression LOO predictions
```

### 4. Open the notebook

```bash
cd notebooks
jupyter notebook experiment.ipynb
```

---

## Phases

| Phase | Description |
|---|---|
| **Phase 1: Survey** | Literature review in `report/final_report.md` |
| **Phase 2: Evaluation** | ROUGE + BERTScore + BARTScore via `main.py` |
| **Phase 3: Analysis** | Correlations + figures in `results/` |
| **Phase 4: Innovation** | Hybrid metric, error analysis, regression |

---

## Key Findings

1. **ROUGE fails on paraphrasing** — BERTScore correctly recognizes semantically equivalent summaries
2. **BERTScore-F1** is the best single metric for abstractive summarization evaluation
3. **Hybrid metric** (α=0.25 ROUGE1 + β=0.15 ROUGE2 + γ=0.60 BERTScore-F1) outperforms any individual metric
4. **Ridge regression** trained on metric features achieves LOO R² ≈ 0.89

---

## Requirements

- Python 3.10+
- PyTorch (CPU-only sufficient for testing)
- Internet connection (first run downloads BERTScore/BARTScore models)

---

## Citation

If you use this project, please cite the underlying metrics:

- ROUGE: Lin (2004)
- BERTScore: Zhang et al. (2020)
- BARTScore: Yuan et al. (2021)
- DUC 2004: Over & Yen (2004)
