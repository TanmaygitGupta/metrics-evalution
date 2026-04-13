# Edge Cases in Summarization Evaluation

In NLP evaluation, an "edge case" is a scenario where a metric behaves unpredictably or fails to align with human judgment. The simulated DUC 2004 dataset in this project was specifically engineered to trigger these edge cases to test the robustness of the metrics.

This document breaks down the primary edge cases you need to be aware of and how your evaluation framework handles them.

---

## 1. The Paraphrase Edge Case (The Most Critical)

**The Scenario:** The AI-generated summary conveys the *exact same meaning* as the human reference, but uses *completely different vocabulary*. 
* **Example Reference:** "Scientists discovered that regular physical exercise significantly reduces the risk of cardiovascular disease..."
* **Example AI Output:** "Research has shown that consistently working out lowers the chances of heart-related illnesses..." (Sample DUC2004_011)

**How Metrics React:**
* ❌ **ROUGE Fails:** ROUGE grades this incredibly poorly (ROUGE-1 drops to `~0.21`, ROUGE-2 drops to `0.00`) because there are almost no overlapping n-grams.
* ✅ **BERTScore Succeeds:** BERTScore detects the contextual and semantic similarity of the word embeddings, correctly rewarding the summary with a high `~0.86` F1 score.

**Why it matters:** As modern Large Language Models become highly "abstractive" (rather than just copy-pasting sentences), paraphrasing is the norm. ROUGE's failure on this edge case is the primary justification for this entire project.

---

## 2. The Keyword Salad Edge Case (The Lexical Trap)

**The Scenario:** The AI outputs a summary that contains almost all the correct "keywords" from the reference, but strings them together in a grammatically incorrect or semantically nonsensical way.

**How Metrics React:**
* ❌ **ROUGE Fails:** ROUGE-1 will give this a remarkably high score because it only checks if the words are present, not if they make sense.
* ⚠️ **BERTScore Struggles:** Because the word embeddings closely match the reference, BERTScore might still over-credit this summary, though its contextual nature dampens the score slightly.
* ✅ **BARTScore Succeeds:** Because BARTScore evaluates the text as a *sequence generation* task, it will notice that the text lacks fluency and grammatically makes no sense, penalizing it heavily via low log-likelihood.

---

## 3. The Hallucination Edge Case (The Baseline Test)

**The Scenario:** The AI confidently generates a beautifully formatted, highly fluent summary... about a completely incorrect topic. 
* **Example Reference:** Talking about the UN Security Council debating the Iraq invasion.
* **Example AI Output:** Talking about stock markets surging to record highs. (Sample DUC2004_013)

**How Metrics React:**
* ✅ **Both Metrics Succeed:** ROUGE-1 drops to `~0.04` and BERTScore drops to its baseline lowest tier (`~0.68`). 

**Why it matters:** We need to prove that "Semantic" models like BERTScore aren't just handing out high scores simply because the English is fluent. This edge case proves the metrics can successfully detect complete factual deviations.

---

## 4. The Extractive Exact-Match Edge Case

**The Scenario:** The AI model is purely "extractive," meaning it literally copy-pastes a sentence directly from the source text, which happens to perfectly match the human reference. (Sample DUC2004_015)

**How Metrics React:**
* ✅ **Both Metrics Succeed:** Both ROUGE and BERTScore will issue a perfect `1.0` score across the board. 

**Why it matters:** It provides a calibration ceiling. It proves the math is working properly at the upper limit.

---

## How Your Framework Solves These Edge Cases

If asked during a presentation how you address these edge cases, refer to the **Error Analysis Module** (`disagreement_category` column in your CSVs).

Your pipeline actively hunts for these edge cases by calculating the `Score Delta` (the difference between normalized BERTScore and ROUGE). If the delta is massive, your code automatically flags the summary as a *"Semantic Match (ROUGE fails)"* or *"Lexical Match (BERT over-credits)"*. 

Instead of trusting a single metric, your framework uses the **Hybrid Score**, balancing the mathematical risks of each individual metric's edge cases to provide a much safer, human-aligned final grade.
