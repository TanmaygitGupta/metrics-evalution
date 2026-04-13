# Project Presentation Guide: Summarization Metrics Evaluation

This guide is designed to help you quickly understand the end-to-end flow of the **Summarization Metrics Evaluation Project** so you can easily present it to an audience or discuss it in an interview.

---

## 1. Project Overview (The "Elevator Pitch")

This project is an end-to-end, research-grade NLP pipeline that evaluates the quality of machine-generated text summaries. 

It specifically compares traditional lexical metrics (like **ROUGE**) against modern, AI-driven semantic metrics (like **BERTScore** and **BARTScore**) using a simulated version of the classic **DUC 2004** dataset.

The ultimate goal of the project was not just to compare these metrics, but to **innovate by creating a custom hybrid evaluation approach** that better aligns with human judgment.

---

## 2. The Core Problem We Solved

When AI generates a text summary, developers need a way to automatically grade its quality against a human-written "reference" summary. 

- **The Old Way (ROUGE):** Historically, the industry standard was ROUGE, which simply counts how many overlapping words (n-grams) exist between the AI summary and the human reference. 
- **The Failure Point:** ROUGE fails entirely on **paraphrasing**. If the human reference says *"doctor"* and the AI summary says *"physician"*, ROUGE penalizes the AI, even though the meaning is identical. ROUGE only measures *surface-level* text overlap, not *actual semantic meaning*.

---

## 3. The Three Metrics Evaluated

We built the pipeline to evaluate three distinct generations of NLP metrics:

1. **ROUGE (The Baseline):** Fast and highly interpretable, but fails to understand semantics or paraphrasing.
2. **BERTScore (The Semantic Upgrade):** Uses pre-trained language models (BERT) to measure the *cosine similarity* of word embeddings. It successfully recognizes paraphrasing, synonyms, and context.
3. **BARTScore (The Generative Approach):** Treats evaluation as a text generation task. It scores how "likely" a summary is based on a seq2seq neural network, capturing fluency and factual consistency.

---

## 4. The Architecture & Pipeline

The project is structured as a highly modular Python application. Here is how data moves through the system during execution:

1. **Data Generation (`data/duc_simulate.py`):** The system generates a simulated dataset of 20 summaries containing exact matches, paraphrased variants, partial overlaps, and explicit hallucinations (completely wrong summaries).
2. **Loading & Scoring (`src/metrics.py`):** The data is loaded and sequentially passed through the three scorers.
3. **Output Generation (`results/`):** The system evaluates the scores and generates CSV data files alongside an array of beautiful charts (Heatmaps, Scatter Plots, Histograms) visualizing how the metrics relate to one another.

---

## 5. Our Core Innovations (The "Secret Sauce")

To make this project stand out, we didn't just measure metrics—we engineered three custom solutions found in `src/innovation.py`:

* **Innovation 1: The Hybrid Metric** 
  We created a custom formula combining the best of lexical and semantic metrics: 
  `Hybrid Score = (0.25 * ROUGE-1) + (0.15 * ROUGE-2) + (0.60 * BERTScore-F1)`
  This custom metric provides a much more robust grading system than any single metric alone.

* **Innovation 2: The Error Analysis Module** 
  We built a classifier that actively hunts for "disagreements" between metrics. If ROUGE scores a summary very low, but BERTScore scores it very high, our module flags this as a *"Semantic Match (Paraphrase)"*—programmatically proving scenarios where ROUGE failed.

* **Innovation 3: Ridge Regression Predictor** 
  We trained a machine learning regression model (utilizing Leave-One-Out Cross-Validation) directly on the metric scores. It successfully learns how to linearly combine ROUGE and BERTScore variables to predict an optimal proxy score for human evaluation, achieving an impressive $R^2 \approx 0.89$.

---

## 6. Key Findings to Share (Conclusion)

If you need a "Conclusion" slide, these are the absolute main takeaways:

1. **Empirical Proof of ROUGE's Failure:** Our pipeline proved mathematically that for perfectly paraphrased summaries, ROUGE-1 scores drop by 20-35 points, whereas BERTScore remains highly accurate.
2. **BERTScore-F1 is the Best Single Metric:** Among individual metrics, BERTScore-F1 proved to be the most reliable and highly correlated metric for evaluating modern abstractive summarization models.
3. **Metric Correlation Divergence:** ROUGE-1 and BERTScore-F1 have roughly a `0.70` correlation—meaning they agree on extreme ends (like total hallucinations or exact matches), but diverge heavily in the muddy waters of paraphrasing.

---

## 7. Command-Line Demo Instructions

If you are asked to demonstrate the project running live, use these commands in your terminal from the project folder:

**Fast Demonstration (skips the massive 1.6GB BARTScore model download):**
```bash
python main.py --no-bartscore
```

**Super Quick Demo (only runs 10 samples instead of 20):**
```bash
python main.py --no-bartscore --limit 10
```

**After execution, show off the generated artifacts:** 
1. Open the `results/` folder to show your audience the generated `.png` graphs (like `heatmap_pearson.png` and `hybrid_comparison.png`).
2. Point them to the `report/final_report.md` file to show the comprehensive academic write-up that the pipeline supports.
