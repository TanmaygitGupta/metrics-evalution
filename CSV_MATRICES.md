# Data Matrices: CSV Results Breakdown

This document provides a highly readable Markdown visualization of the matrices found inside your `results` directory. The long reference and system summary text columns have been hidden from the raw data tables to focus specifically on the numerical matrices.

---

## 1. Pearson Correlation Matrix (`pearson_correlation.csv`)
*This matrix shows the linear correlation between metrics. Notice the ~0.83 correlation between ROUGE-1 and BERTScore-F1, showing they agree overall but diverge on specific edge cases.*

| Metric | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore (P) | BERTScore (R) | BERTScore (F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ROUGE-1** | 1.0000 | 0.9187 | 0.9435 | 0.8440 | 0.8176 | 0.8391 |
| **ROUGE-2** | 0.9187 | 1.0000 | 0.9575 | 0.7260 | 0.7249 | 0.7332 |
| **ROUGE-L** | 0.9435 | 0.9575 | 1.0000 | 0.8383 | 0.8243 | 0.8398 |
| **BERTScore (P)** | 0.8440 | 0.7260 | 0.8383 | 1.0000 | 0.9585 | 0.9888 |
| **BERTScore (R)** | 0.8176 | 0.7249 | 0.8243 | 0.9585 | 1.0000 | 0.9903 |
| **BERTScore (F1)**| 0.8391 | 0.7332 | 0.8398 | 0.9888 | 0.9903 | 1.0000 |

---

## 2. Spearman Correlation Matrix (`spearman_correlation.csv`)
*This matrix shows the rank-order correlation. It tells a similar story to Pearson, confirming the stability of the metrics regardless of their exact numerical distribution.*

| Metric | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore (P) | BERTScore (R) | BERTScore (F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ROUGE-1** | 1.0000 | 0.8977 | 0.8661 | 0.7589 | 0.6942 | 0.7266 |
| **ROUGE-2** | 0.8977 | 1.0000 | 0.8909 | 0.8445 | 0.7479 | 0.8272 |
| **ROUGE-L** | 0.8661 | 0.8909 | 1.0000 | 0.8642 | 0.7815 | 0.8304 |
| **BERTScore (P)** | 0.7589 | 0.8445 | 0.8642 | 1.0000 | 0.8586 | 0.9429 |
| **BERTScore (R)** | 0.6942 | 0.7479 | 0.7815 | 0.8586 | 1.0000 | 0.9549 |
| **BERTScore (F1)**| 0.7266 | 0.8272 | 0.8304 | 0.9429 | 0.9549 | 1.0000 |

---

## 3. Raw Metric Results (`evaluation_results.csv`)
*These are the raw scores computed by the NLP pipelines for the 20 simulated samples.*

| ID | Category Type | ROUGE-1 | ROUGE-2 | ROUGE-L | BS-Precision | BS-Recall | BS-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| DUC2004_001 | Normal | 0.4528 | 0.1176 | 0.2642 | 0.8749 | 0.8647 | 0.8698 |
| DUC2004_002 | Normal | 0.5333 | 0.2326 | 0.3556 | 0.8926 | 0.9097 | 0.9011 |
| DUC2004_003 | Normal | 0.5000 | 0.1905 | 0.3182 | 0.8706 | 0.8762 | 0.8734 |
| DUC2004_004 | Normal | 0.4151 | 0.1569 | 0.3396 | 0.8851 | 0.8640 | 0.8744 |
| DUC2004_005 | Normal | 0.5238 | 0.3500 | 0.5238 | 0.9195 | 0.9138 | 0.9166 |
| DUC2004_006 | Normal | 0.4286 | 0.2500 | 0.4286 | 0.9119 | 0.8913 | 0.9015 |
| DUC2004_007 | Normal | 0.4186 | 0.1463 | 0.3256 | 0.8940 | 0.9112 | 0.9025 |
| DUC2004_008 | Normal | 0.3182 | 0.1429 | 0.3182 | 0.8883 | 0.9073 | 0.8977 |
| DUC2004_009 | Normal | 0.3462 | 0.0400 | 0.2692 | 0.8599 | 0.8448 | 0.8523 |
| DUC2004_010 | Normal | 0.2917 | 0.1304 | 0.2500 | 0.8902 | 0.8557 | 0.8726 |
| **DUC2004_011** | **Paraphrase** | **0.2162** | **0.0000** | **0.2162** | **0.8546** | **0.8643** | **0.8594** |
| **DUC2004_012** | **Paraphrase** | **0.2791** | **0.0488** | **0.2791** | **0.8656** | **0.8789** | **0.8722** |
| DUC2004_013 | Hallucination | 0.0444 | 0.0000 | 0.0444 | 0.6833 | 0.6932 | 0.6882 |
| DUC2004_014 | Hallucination | 0.1500 | 0.0000 | 0.1500 | 0.7720 | 0.7607 | 0.7663 |
| DUC2004_015 | Exact Match | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| DUC2004_016 | Exact Match | 0.6667 | 0.4500 | 0.6667 | 0.9415 | 0.9608 | 0.9511 |
| DUC2004_017 | Partial | 0.2791 | 0.0976 | 0.1860 | 0.8345 | 0.8268 | 0.8306 |
| DUC2004_018 | Partial | 0.2857 | 0.0000 | 0.2857 | 0.8768 | 0.8621 | 0.8693 |
| DUC2004_019 | Partial | 0.3556 | 0.0465 | 0.2667 | 0.8301 | 0.8021 | 0.8158 |
| DUC2004_020 | Partial | 0.5098 | 0.2041 | 0.3922 | 0.8753 | 0.8301 | 0.8521 |
*(Note how ROUGE-1 plummets on Samples 011 and 012, while BERTScore stays above ~0.85).*

---

## 4. Analytical Enriched Results (`results_enriched.csv`)
*This expands on the raw results by incorporating the calculated Hybrid Score, the Score Delta, and the Classifier's Disagreement Category.*

| ID | Score Delta | ROUGE-1 | BS-F1 | Hybrid Score | Predicted Hybrid | Disagreement Category |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| DUC2004_001 | 0.4170 | 0.4528 | 0.8698 | **0.6527** | 0.6405 | Mixed |
| DUC2004_002 | 0.3678 | 0.5333 | 0.9011 | **0.7089** | 0.6869 | Semantic Match (ROUGE fails) |
| DUC2004_003 | 0.3734 | 0.5000 | 0.8734 | **0.6776** | 0.6588 | Semantic Match (ROUGE fails) |
| DUC2004_004 | 0.4593 | 0.4151 | 0.8744 | **0.6520** | 0.6508 | Semantic Match (ROUGE fails) |
| DUC2004_005 | 0.3928 | 0.5238 | 0.9166 | **0.7334** | 0.7224 | Semantic Match (ROUGE fails) |
| DUC2004_006 | 0.4729 | 0.4286 | 0.9015 | **0.6855** | 0.6863 | Semantic Match (ROUGE fails) |
| DUC2004_007 | 0.4839 | 0.4186 | 0.9025 | **0.6681** | 0.6674 | Semantic Match (ROUGE fails) |
| DUC2004_008 | 0.5795 | 0.3182 | 0.8977 | **0.6396** | 0.6548 | Semantic Match (ROUGE fails) |
| DUC2004_009 | 0.5061 | 0.3462 | 0.8523 | **0.6039** | 0.6132 | Agreement-Low |
| DUC2004_010 | 0.5809 | 0.2917 | 0.8726 | **0.6160** | 0.6281 | Semantic Match (ROUGE fails) |
| DUC2004_011 | 0.6432 | 0.2162 | 0.8594 | **0.5697** | 0.5987 | Agreement-Low |
| DUC2004_012 | 0.5931 | 0.2791 | 0.8722 | **0.6004** | 0.6225 | Agreement-Low |
| DUC2004_013 | 0.6438 | 0.0444 | 0.6882 | **0.4240** | 0.4704 | Agreement-Low |
| DUC2004_014 | 0.6163 | 0.1500 | 0.7663 | **0.4973** | 0.5281 | Agreement-Low |
| DUC2004_015 | 0.0000 | 1.0000 | 1.0000 | **1.0000** | 0.8576 | Agreement-High |
| DUC2004_016 | 0.2844 | 0.6667 | 0.9511 | **0.8048** | 0.7779 | Semantic Match (ROUGE fails) |
| DUC2004_017 | 0.5515 | 0.2791 | 0.8306 | **0.5828** | 0.5907 | Agreement-Low |
| DUC2004_018 | 0.5836 | 0.2857 | 0.8693 | **0.5930** | 0.6173 | Agreement-Low |
| DUC2004_019 | 0.4602 | 0.3556 | 0.8158 | **0.5854** | 0.5907 | Agreement-Low |
| DUC2004_020 | 0.3423 | 0.5098 | 0.8521 | **0.6693** | 0.6535 | Mixed |
