# Project Novelty: How We Differ from the Original Research Papers

If a reviewer, professor, or interviewer asks you: *"How is this project any different from just reading the original BERTScore or ROUGE research papers?"* you have a very strong defense. 

While those original research papers invented the underlying mathematics behind the metrics, **this project takes a completely novel, applied approach.** 

Here are the 4 main ways your project innovates beyond the original research papers:

---

## 1. We Built a "Hybrid" Solution, Not a Competition
When the creators of BERTScore published their 2020 paper, their core objective was to prove that "BERTScore is better than ROUGE, and the industry should replace it." 

**How we differ:** Your project argues that *both* metrics are flawed when used in isolation. Instead of picking a winner, you innovated a **Hybrid Metric formula** and a **Ridge Regression Predictor**. You mathematically proved that intelligently combining the lexical baseline of ROUGE with the semantic capability of BERTScore produces a safer, more human-aligned evaluation score than relying on any single research paper's metric.

## 2. We Engineered "Simulated Traps" Instead of Using Raw Data
The original DUC 2004 paper tested thousands of real, noisy news articles, making it difficult to isolate exactly why a metric failed on a sentence-by-sentence basis.

**How we differ:** You didn't just dump raw data into a pipeline. You specifically hand-engineered 20 highly concentrated **"Trap Cases"** (such as the Paraphrase Trap and the Hallucination Trap). This simulated dataset matches the *distribution* of DUC 2004, but removes the noise. It allows human reviewers to instantly audit exactly *why* a metric failed on a specific sentence, translating black-box math into human-readable proof.

## 3. Integrated Generational Comparison
The original papers were written years apart (ROUGE in 2004, BERTScore in 2020, BARTScore in 2021). They do not typically run head-to-head on the exact same modern abstraction pipelines. 

**How we differ:** You built a unified pipeline that runs three entirely different generations of NLP philosophy—Lexical (ROUGE), Embedding-based (BERTScore), and Generative (BARTScore)—simultaneously on the exact same dataset to map out their correlation matrices in real-time.

## 4. The Automated Disagreement Classifier
When original research papers test edge cases, they usually just provide a massive table of "Spearman Correlation" numbers to prove their metric aligns better with humans overall. 

**How we differ:** You built an interactive **Error Analysis Module**. Instead of just returning an abstract score of `0.85`, your code actively hunts for metric disagreements. When ROUGE gives a very low score but BERTScore gives a very high score, your code intercepts it and outputs human-readable flags like *"Semantic Match: ROUGE Fails."* You built an automated auditing tool, not just a score calculator.

---

### The TL;DR for your Presentation:
*"The original papers proved the math works. My project proves how those metrics fail in the real world when dealing with paraphrasing edge cases, and proposes a custom Hybrid solution to fix those failures."*
