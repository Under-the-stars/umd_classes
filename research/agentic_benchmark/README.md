# Agentic Benchmark: Fair Comparison and Full Study

Author: Shobha Gupta | DATA605 Spring 2026 | Issue #435

## Question

Does a multi-agent LangGraph pipeline with a Reviewer node produce more reliable results than a single-pass pipeline and AutoGluon, especially when the data is corrupted?

## Short answer

The Reviewer does not improve accuracy. It does make the pipeline safer. It finds every injected impossible value, and it flagged an injected leaked feature in 10 of 10 runs, where the systems without it scored a false MCC of 1.0. On clean data the fixed Reviewer matches the no-reviewer score on Heart Disease and Amazon. On NYC Taxi it scores lower on MCC.

## Why this folder exists

My first results said LangGraph won. When I audited them, the comparison was not fair: the systems used different models and different splits, the corruption had no random seed, the test set was tiny, and models were chosen by F1, which rewards predicting everyone as positive. I rebuilt the benchmark so every system is tested the same way. Claims from the first version (for example "only LangGraph detected leakage" and "+16.6% under noise") are not supported and should be ignored.

## Setup

- Systems: always positive (baseline), single agent, graph without Reviewer, graph with the old Reviewer, graph with the fixed Reviewer, and AutoGluon (see the note below).
- Every system gets the same stratified split, the same features and the same candidate models. The model is chosen by cross-validated MCC on training data only.
- Corruption is seeded and applied to training data only: label flips (5, 15, 30%) plus impossible values (2, 5, 5%). The corrupted rows are recorded, so the Reviewer can be scored as a detector.
- Seeds: Heart Disease 10, NYC Taxi 3, Amazon Reviews 3. Samples: 5000 rows for NYC Taxi and Amazon, 303 rows for Heart.
- Metrics: MCC (Matthews Correlation Coefficient: 0 is chance, 1 is perfect), balanced accuracy, specificity, F1, recall, precision, AUC. F1 is reported for comparison with the first version, but it is unsafe alone, because an always-positive model already scores a high F1.
- Train and test scores are both recorded. Train scores are measured on the corrupted training data, so they are not a clean overfitting measure.

## The fixed Reviewer

- Drops rows with impossible values (bounds written by hand per dataset).
- Flags a leaked feature in two ways: a single feature that separates the classes almost perfectly, and a score-jump check (a very high score that collapses when one feature is removed).
- Flags rows that look mislabeled, but removes them only if a nested cross-validation test shows that removing them helps. In every run so far, it never passed that test, so these rows were reported, not removed.
- If the model is degenerate (predicts almost everyone positive, or almost no one), it retries with a different strategy: class weights, a tuned threshold, or another model.
- The old Reviewer removed suspected wrong labels without checking that it helped, which cost accuracy on Heart Disease.

## Results (held-out test set, mean over seeds)

MCC, with F1 in brackets. Always positive has MCC 0.000 everywhere, and F1 0.629 (Heart), 0.889 (Taxi) and 0.850 (Amazon). The single agent and the graph without Reviewer use the same models on the same data, so their scores are identical.

| Dataset | Label noise | Single agent / graph, no Reviewer | Graph + old Reviewer | Graph + fixed Reviewer |
|---|---|---|---|---|
| Heart Disease | 0% | 0.646 (0.797) | 0.638 (0.793) | 0.650 (0.799) |
| Heart Disease | 5% | 0.654 (0.801) | 0.632 (0.790) | 0.654 (0.799) |
| Heart Disease | 15% | 0.651 (0.798) | 0.562 (0.756) | 0.628 (0.787) |
| Heart Disease | 30% | 0.504 (0.723) | 0.443 (0.702) | 0.466 (0.710) |
| Amazon Reviews | 0% | 0.587 (0.898) | 0.587 (0.898) | 0.587 (0.898) |
| Amazon Reviews | 5% | 0.574 (0.893) | 0.577 (0.895) | 0.577 (0.895) |
| Amazon Reviews | 15% | 0.545 (0.882) | 0.553 (0.886) | 0.553 (0.886) |
| Amazon Reviews | 30% | 0.532 (0.889) | 0.532 (0.889) | 0.532 (0.889) |
| NYC Taxi (payment_type dropped) | 0% | 0.202 (0.894) | 0.172 (0.841) | 0.129 (0.806) |
| NYC Taxi (payment_type dropped) | 5% | 0.206 (0.894) | 0.156 (0.860) | 0.156 (0.860) |
| NYC Taxi (payment_type dropped) | 15% | 0.206 (0.894) | 0.157 (0.840) | 0.157 (0.840) |
| NYC Taxi (payment_type dropped) | 30% | 0.206 (0.894) | 0.130 (0.780) | 0.130 (0.780) |

Standard deviations over seeds are in `results_full/full_all.csv`. On Heart Disease they are about 0.1 MCC, so the differences between systems at 0% to 15% noise are within the spread.

Fixed Reviewer minus no Reviewer (MCC): Heart Disease +0.003, -0.001, -0.023, -0.038; Amazon 0.000, +0.003, +0.008, 0.000; NYC Taxi -0.073, -0.050, -0.049, -0.076 (for 0, 5, 15, 30% noise).

### AutoGluon

AutoGluon is not part of the `full_*.csv` tables above. An earlier run (60 second limit, medium quality, different seeds) gave Heart Disease MCC 0.642 (clean), 0.605 (15%), 0.468 (30%); Amazon 0.588, 0.573, 0.517; NYC Taxi 0.139 (clean), 0.125 (15%). That puts it level with the single agent on Heart and Amazon and lower on Taxi. Those numbers are from a different run and are not directly comparable. Results of the same-split AutoGluon run, if available, are in `results_full/full_autogluon.csv`.

## How well the Reviewer found the corrupted data

| Dataset | Impossible values found | Flipped labels flagged (fixed Reviewer) | Flipped labels found (old Reviewer) |
|---|---|---|---|
| Heart Disease | all (precision 1.0, recall 1.0) | precision 0.41, recall 0.52 | precision 0.50, recall 0.30 |
| Amazon Reviews | all | precision 0.60, recall 0.54 | none |
| NYC Taxi | all | precision 0.40, recall 0.29 | none |
| Injected leak (Heart) | not applicable | not applicable | not applicable |

Leak test: a near-copy of the label was added as a feature. The single agent and the graph without Reviewer scored MCC 1.000. Both Reviewers reported the honest score of about 0.65, and the fixed Reviewer flagged the leaky feature in 10 of 10 runs.

The fixed Reviewer also flags many rows on clean data (about 270 on Amazon, 28 on Heart, 616 on Taxi). These are mostly naturally hard rows, so a flag means "worth a look", not "wrong label".

## What this shows

1. On accuracy there is no clear winner. LangGraph, the single agent and AutoGluon are within each other's spread on Heart Disease and Amazon, and the simple models beat AutoGluon on Taxi.
2. The Reviewer does not improve accuracy. The fixed version matches no Reviewer on clean data for Heart and Amazon, and is slightly lower on Heart at 15% and 30% noise, within the spread.
3. On NYC Taxi the Reviewer lowers MCC and F1 but raises balanced accuracy (0.569 vs 0.527 on clean data), because its degenerate-model retry switches to a balanced model. This is a different trade-off, not a clear improvement or loss.
4. The Reviewer is useful for data integrity. It finds all injected impossible values and catches an injected leak where the other systems silently report a perfect score.
5. It misses the natural `payment_type` leak in NYC Taxi. In an earlier run, keeping that one column lifted every system from MCC 0.21 to 0.91, and no system removed it.
6. F1 is a poor comparison metric here. On Taxi, always positive scores F1 0.889 against 0.894 for the best system. MCC and balanced accuracy separate real models from the baseline.
7. Rules route the graph. The LLM only writes the plan note and the summary, so these runs do not test LLM decision making.

## Limits

- The single agent is a stand-in for a one-pass assistant. I did not run ChatGPT.
- Taxi and Amazon use 3 seeds and 5000 rows; Heart uses 10 seeds and 303 rows. Small differences are not reliable.
- The leak test uses an injected feature that is easy to spot by design.
- The Reviewer's impossible-value bounds are written by hand per dataset.
- The fixed Reviewer was tuned on separate seeds (100 to 104) of Heart Disease only. Taxi and Amazon were not used for tuning.
- Train scores are measured on corrupted training data, so a train-test gap mixes overfitting with label noise.
- The Reviewer's check for removing mislabeled rows never passed its safety test in these runs, so no labels were removed.

## Files

- `agentic_bench.py`: the benchmark harness (splits, systems, Reviewer, corruption, metrics).
- `agentic_viz.py`: plots and tables.
- `full_study.ipynb`: the end-to-end notebook (EDA, one walkthrough run per dataset with SHAP, full benchmark, charts, leak test).
- `run_heart.py`, `run_heart_ag.py`, `run_amazon_ag.py`, `run_taxi.py`, `run_taxi_ag.py`, `run_benchmark.py`, `run_leak.py`: scripts from the first fair benchmark.
- `results/`: raw CSVs from the first fair benchmark.
- `results_full/`: raw CSVs and plots from the full study above.

## How to run

1. Install: `pip install langgraph langchain-ollama scikit-learn pandas numpy matplotlib seaborn shap vaderSentiment pyarrow jupyter` (add `autogluon.tabular` for AutoGluon).
2. Put the data where the notebook expects it (Heart Disease CSV, NYC Taxi parquet, Amazon Beauty CSV) and edit the paths in the first code cell.
3. Optional: start Ollama (`ollama serve`, `ollama pull llama3.2`) and set `USE_LLM = True` for LLM summaries.
4. Open `full_study.ipynb` and run all cells. Set `QUICK=1` in the environment first for a short test. Results are written to `outputs/`.
