# Responsible AI Assignment 2 - Toxicity Moderation Pipeline

This repository contains a complete end-to-end implementation of Assignment 2 in Jupyter notebooks (`part1.ipynb` to `part5.ipynb`) plus the production-style moderation pipeline code (`pipeline.py`).

The work covers:
- baseline model training and threshold tuning,
- bias audit across demographic cohorts,
- adversarial robustness testing,
- fairness mitigation strategies, and
- deployment-oriented multi-layer moderation design.

## Repository Contents

- `part1.ipynb` - Baseline DistilBERT training, evaluation, and operating-threshold selection
- `part2.ipynb` - Bias audit across cohorts with AIF360 fairness metrics
- `part3.ipynb` - Adversarial attacks (evasion and poisoning) and risk analysis
- `part4.ipynb` - Fairness mitigations (reweighing + threshold optimization) and trade-off study
- `part5.ipynb` - Production moderation pipeline simulation with calibration and routing
- `pipeline.py` - Three-layer moderation pipeline (regex pre-filter + calibrated model + review routing)

## Dataset and Split Protocol

- Dataset: `jigsaw-unintended-bias-train.csv`
- Labeling rule: `label = 1` if `toxic >= 0.5`, else `0`
- Stratified subset used in the assignment:
  - 120,000 total rows
  - 100,000 train
  - 20,000 evaluation
- Class balance (from notebook output):
  - Training toxic ratio: `0.0800`
  - Evaluation toxic ratio: `0.0799`

## Part-by-Part Requirements Coverage and Results

### Part 1 - Baseline Model + Threshold Selection

Implemented:
- Fine-tuned DistilBERT toxicity classifier
- Evaluated on held-out evaluation split
- Swept decision thresholds to select operating point

Results:
- Accuracy: `0.9498`
- Macro F1: `0.8087`
- AUC-ROC: `0.9524`
- Threshold sweep (Macro F1):
  - `0.3 -> 0.8086`
  - `0.4 -> 0.8143` (best)
  - `0.5 -> 0.8087`
  - `0.6 -> 0.7961`
  - `0.7 -> 0.7734`

Selected operating threshold: `0.4` (best macro-F1 trade-off).

### Part 2 - Bias Audit and Fairness Metrics

Implemented:
- Cohort audit:
  - High-Black cohort: `black >= 0.5`
  - Reference cohort: `black < 0.1 AND white >= 0.5`
- Computed confusion-matrix metrics by cohort
- Calculated disparate impact and AIF360 group fairness metrics

Results:
- Cohort sizes:
  - High-Black: `149`
  - Reference: `187`
- High-Black vs Reference:
  - TPR: `0.3778` vs `0.4423`
  - FPR: `0.1154` vs `0.0741`
  - FNR: `0.6222` vs `0.5577`
  - Precision: `0.5862` vs `0.6970`
- FPR Disparate Impact ratio: `1.5577`
- Statistical Parity Difference: `0.0182`
- Equal Opportunity Difference: `-0.0645`

Key finding: the baseline model over-flags the High-Black cohort (higher false positives).

### Part 3 - Adversarial Robustness (Evasion + Poisoning)

Implemented:
- Character-level evasion attack against toxic comments
- Data poisoning experiment (5% corruption in training set)
- Comparative degradation analysis

Results:
- Evasion Attack Success Rate (ASR): `0.9880`
- Avg toxic confidence before attack: `0.9077`
- Avg toxic confidence after attack: `0.0259`
- Clean vs Poisoned model:
  - F1: `0.8142` vs `0.7950`
  - FNR: `0.3627` vs `0.3734`

Key finding: real-time evasion is substantially more dangerous operationally than this level of training-time poisoning.

### Part 4 - Fairness Mitigation and Incompatibility Analysis

Implemented:
- Mitigation 1: reweighing
- Mitigation 2: post-processing threshold optimization (equalized-odds style)
- Comparative fairness-performance trade-off evaluation
- Formal incompatibility discussion (demographic parity vs equalized odds under unequal base rates)

Results:
- Baseline:
  - Overall F1: `0.8142`
  - High-Black FPR: `0.1154`
  - Reference FPR: `0.0741`
  - SPD: `0.0182`
  - EOD: `-0.0645`
- Reweighing:
  - Overall F1: `0.8053`
  - High-Black FPR: `0.1635`
  - Reference FPR: `0.1704`
  - SPD: `-0.0283`
  - EOD: `-0.1175`
- Threshold Optimization:
  - Overall F1: `0.5926`
  - High-Black FPR: `0.0000`
  - Reference FPR: `0.0000`
  - SPD: `-0.0092`
  - EOD: `-0.0427`

Key finding: stronger fairness constraints can dramatically reduce utility (F1), highlighting unavoidable trade-offs.

### Part 5 - Production Moderation Pipeline Simulation

Implemented:
- Three-layer architecture in `pipeline.py`:
  1. Regex-based hard blocklist pre-filter
  2. Calibrated model scoring (Isotonic Regression)
  3. Decision routing:
     - `allow` if probability `< 0.4`
     - `review` if `0.4 <= probability < 0.6`
     - `block` if probability `>= 0.6`
- Demonstrated pipeline behavior on 1,000 sampled comments

Results:
- Regex hard-block hits: `0` in sampled 1,000
- Auto-actioned (allow/block): `984 / 1000` (`98.4%`)
- Human review queue: `16 / 1000` (`1.6%`)
- Auto-actioned subset metrics:
  - Macro F1: `0.7656`
  - Toxic precision: `0.7021`
  - Toxic recall: `0.4648`
- Review queue composition:
  - Toxic: `6` (`37.5%`)
  - Non-toxic: `10` (`62.5%`)

Key finding: the system is highly automatable but conservative on toxic recall under strict block thresholds.

## How to Run

The notebooks are authored for Google Colab + Google Drive.

1. Upload project files to Drive (same structure).
2. Ensure dataset file exists at project root:
   - `jigsaw-unintended-bias-train.csv`
3. Open each notebook in order:
   - `part1.ipynb` -> `part2.ipynb` -> `part3.ipynb` -> `part4.ipynb` -> `part5.ipynb`
4. Execute all cells top-to-bottom in each notebook.

## Reproducibility Notes

- Stratified splits use fixed random seeds (`random_state=42`) throughout.
- Models and checkpoints are saved under `saved_model/` paths in the notebooks.
- Some fairness/adversarial outputs depend on exact library versions and environment state.

## Final Takeaways

- Baseline toxicity detection is strong on aggregate metrics.
- Measurable bias appears across demographic cohorts.
- Evasion attacks are highly effective without robust input defenses.
- Fairness mitigation improves some parity objectives but incurs utility costs.
- A layered moderation architecture with calibrated uncertainty routing is practical for deployment at scale.
