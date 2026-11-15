# Benchmarking Agentic Reasoning for Data Science: Full Project Playbook

Nov 15, 2026 · @shobha

## 1. Goal and research question

This project asks one question: does a multi-agent LangGraph pipeline with a Reviewer node produce more reliable results than a single-pass pipeline and AutoGluon, especially when the data is corrupted? It started as a DATA605 (Spring 2026) project, issue #435.

**What I compare**

- **Always positive:** predicts the majority class for every row. The floor every system must beat.
- **Single agent:** one linear pass. Fill missing values, pick the best of three models, predict once. A stand-in for a one-pass assistant (I did not run ChatGPT).
- **Graph without Reviewer:** the same pipeline as a LangGraph graph, with the Reviewer switched off. This isolates what the graph structure alone adds.
- **Graph with Reviewer:** the full cyclic pipeline, in two versions: the old Reviewer and the fixed Reviewer.
- **AutoGluon:** an automatic stacked-model baseline with a 60 second limit.

**On which data**

Three datasets: Heart Disease (Cleveland, 303 rows), NYC Taxi tip prediction, and Amazon Beauty review sentiment. Each is run clean and with seeded corruption: wrong labels and impossible values, plus one test with an injected leaked feature.

**What "reliable" means here**

Not only a higher score. A reliable pipeline should also notice when the data is broken (impossible values, wrong labels, a feature that leaks the answer) and refuse to report a fake result. The study measures both: test accuracy, and how well the Reviewer catches the faults I plant.

## 2. What was wrong with the first version

My first results said LangGraph beat AutoGluon and the single agent. When I audited them, the comparison was not fair, so the claims could not be trusted and the whole analysis had to be rerun.

| Problem in the first version | Why it breaks the comparison |
| --- | --- |
| Different models and different splits per system | A score difference could come from the data split or the model choice, not the pipeline |
| Corruption had no random seed | The corrupted data could not be reproduced, and each system saw different damage |
| Heart Disease test set was about 61 rows | One extra correct prediction moves accuracy by roughly 1.6 points, so gaps were noise |
| Best model chosen by F1, on the test set | F1 rewards predicting "positive" for everyone, and choosing on the test set leaks the answer |
| AutoGluon ran on a different split | Its score was not comparable with the others |
| `payment_type` dropped by hand in NYC Taxi | It hid a real leak, then the notebook claimed only LangGraph "caught" it |
| Resampling before the train/test split | Duplicated minority rows ended up in both train and test |
| Missing values filled with the whole dataset's mode | A small leak from test rows into training |
| Reviewer retry did nothing | The retry reran the same step with the same inputs, so the loop added no value |
| Hard-coded results dictionary in the notebook | The numbers shown were typed in, not produced by the run |

**Claims I had to remove**

- "+16.6% under noise" for LangGraph over the single agent.
- "Only LangGraph detected data leakage" in NYC Taxi.
- "LangGraph outperforms AutoGluon on Heart Disease" (F1 88% vs 73%).
- A drop from 0.94 to 0.56 used as proof that the Reviewer matters.

**Why F1 alone was the biggest trap**

On NYC Taxi about 80% of trips are tipped. A model that predicts "tipped" for every trip gets F1 of 0.889 and learns nothing. The best real system gets 0.894. F1 cannot tell them apart. MCC (Matthews Correlation Coefficient: 0 is chance, 1 is perfect) gives the always-positive model exactly 0, so it shows the truth.

## 3. The rules of the fair benchmark

Every flaw in section 2 got one rule. These rules are what make the numbers comparable, so keep them if the project is ever rebuilt.

1. **Same split for everyone.** One stratified 80/20 train/test split per seed, shared by every system.
2. **Same features and same candidate models.** Random forest, logistic regression and histogram gradient boosting for the single agent and the graph.
3. **Choose the model on training data only.** Three-fold cross-validated MCC on the training set. Never F1, never the test set.
4. **Seeded corruption, training data only.** The test set always stays clean. The exact corrupted row ids are recorded, so the Reviewer can be scored as a detector.
5. **Impute inside the pipeline, fit on train.** Missing values are filled from training rows only.
6. **Retries must change something.** Each Reviewer retry tries a new strategy, or the graph stops and says nothing changed.
7. **Many seeds, report mean and standard deviation.** Differences smaller than the spread are called ties.
8. **Always include the always-positive baseline.** It shows how much of a score is real skill.
9. **Everything comes from CSV files.** Tables and charts are built from saved results, never typed in.
10. **Report F1 next to MCC and balanced accuracy.** F1 stays for continuity with the first version, but is never used to choose a model or declare a winner.

## 4. The datasets, and how each one was cleaned and used

Three datasets with different sizes and different difficulty, so the Reviewer is tested in a tiny, a noisy and a nearly unpredictable setting.

|  | Heart Disease | NYC Taxi | Amazon Beauty reviews |
| --- | --- | --- | --- |
| Source | Cleveland heart disease CSV (`heart_disease_cleveland.csv`) | NYC yellow taxi trips, January 2024 (`yellow_tripdata_2024-01.parquet`) | Amazon Reviews 2023, All\_Beauty category (McAuley Lab, Hugging Face) |
| Rows used | 303 (242 train, 61 test) | 5,000 sampled (4,000 train, 1,000 test) | 5,000 sampled (4,000 train, 1,000 test) |
| Target | `target`: heart disease yes/no (164 no, 139 yes) | `tipped`: tip amount above 0 (about 80% yes) | `sentiment`: rating 4 or 5 = positive (about 74% yes) |
| Why chosen | Tiny and clinical: small test set, easy to overfit | Large and nearly unpredictable without the leaky column | Text-derived features, moderate difficulty |

**Heart Disease**

- 13 features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal.
- 6 missing cells (in `ca` and `thal`). They are not filled up front. The pipeline fills them from training rows only: the mode for columns with 10 or fewer distinct values, otherwise the median.
- Impossible-value bounds the Reviewer checks: age 1 to 120, chol 50 to 700, trestbps 50 to 260.

**NYC Taxi**

1. Load the January 2024 parquet file (about 2.96 million raw rows).
2. Compute `trip_duration` in minutes from the pickup and dropoff times.
3. Keep realistic trips only: fare 5 to 500, duration 3 to 180 minutes, distance 0.1 to 100, at least one passenger.
4. Create `tipped` = tip amount above 0, then drop `tip_amount`.
5. Keep 7 features: passenger\_count, trip\_distance, fare\_amount, trip\_duration, PULocationID, DOLocationID, RatecodeID.
6. Sample 5,000 rows with a fixed seed (42).

`payment_type` is left out on purpose. Cash tips are not recorded in this data, so cash trips always show a tip of 0, and the column almost gives away the label. In an earlier run, keeping this one column lifted every system from MCC 0.21 to 0.91. It is the natural leak that no system caught. The Reviewer's bounds for taxi: fare 3 to 500, passengers 1 to 8, distance 0.1 to 100, duration 1 to 180.

**Amazon Beauty reviews**

1. Start from the All\_Beauty reviews file saved as `amazon_beauty_50k.csv` (columns: rating, text, verified, title, helpful\_vote).
2. Drop rows with no text, title or rating, then sample 5,000 rows (seed 42).
3. Create `sentiment` = rating of 4 or more.
4. Build 9 numeric features: text\_length, title\_length, word\_count, exclaim\_count, upper\_ratio, vader\_text, vader\_title (VADER sentiment scores of the text and the title), helpful\_vote and verified.
5. The Reviewer's bounds: text length 1 to 20,000, title length 1 to 500.

**The order of operations (the same for every dataset)**

1. Load and shape the data (above).
2. Add a row id to every row, so any row can be traced later.
3. Split into train and test with a seed (stratified 80/20).
4. Corrupt the training set only, with the same seed. The test set is never touched.
5. Hand the same train and test frames to every system.

## 5. Architecture

The pipeline is a cyclic LangGraph StateGraph with five nodes. The Reviewer sits in the middle of a loop: it can send the graph back to the EDA Analyst (bad data) or the ML Architect (bad model), or pass to the Report Writer.

&#91;embedded content: architecture · 5 nodes, 2 retry loops\]

Read it left to right on the top row, then down into the Reviewer. The two lines going back up are the retry loops. The line going left is the pass route to the report.

**What each node does**

| Node | Reads | Does | Writes |
| --- | --- | --- | --- |
| Planner | Training frame | Counts rows, flags imbalance (minority under 35%) and missing values, optionally asks the LLM for a one-line plan | `plan`, log entry |
| EDA Analyst | Train and test frames, pending fixes | Fills missing values from training rows only. On a retry it drops rows with impossible values, drops rows flagged as mislabeled (if approved) and drops leaky features | Cleaned frames, lists of dropped rows and features |
| ML Architect | Cleaned frames, current strategy | Chooses the best of three models by 3-fold CV MCC on train, fits it, predicts the test set. Strategies: default, class weight, tuned threshold, alternative model | Test metrics, train metrics, model, history |
| Quality Reviewer | Train frame, metrics, history | Runs the checks in section 6 and picks the route | `review_decision`, pending fixes, retry count |
| Report Writer | Everything above | Writes the final report (optionally with an LLM summary) | `final_report` |

**Routing**

- `pass` goes to the Report Writer, then the graph ends.
- `retry_eda` sends the graph back to the EDA Analyst with a list of fixes (impossible values, leaked features, labels).
- `retry_ml` sends it back to the ML Architect with the next untried strategy.
- The graph stops after 3 retries, or when a retry changes nothing, and the report says so.

**Design choices worth remembering**

- **Shared state:** one typed dictionary holds the data, the pending fixes, the retry counter, the history of attempts, the metrics and the log. Every node reads from it and writes back to it.
- **Rules route, the LLM writes:** the routing is rule-based. The optional LLM (llama3.2 through Ollama) writes the plan note and the summary only. A `strategy_mode` switch lets the LLM pick the retry strategy from the untried list, falling back to rules, but the reported runs did not use it. Describe the system as a hybrid, not as an LLM-driven agent.
- **Why a cycle:** a linear pipeline cannot act on what it finds. The loop lets the Reviewer change the inputs and try again, which is the one thing the single agent cannot do.
- **Switching it off:** the same graph with the Reviewer set to always pass is the "graph without Reviewer" baseline.

## 6. The Reviewer: what it checks, why it was rebuilt, what it does now

The Reviewer is the node the whole project is about. It runs after the ML Architect and either passes the result on or sends the graph back with a reason.

**Checks in the first fair version (v1)**

| Check | How it works | If it fires |
| --- | --- | --- |
| Impossible values | Hand-written bounds per dataset (for example age 1 to 120) | `retry_eda`: EDA drops those rows |
| Leaked feature | A single feature with AUC above 0.95 on its own | `retry_eda`: the feature is dropped |
| Wrong labels | Out-of-fold gradient boosting disagrees with the given label (probability under 0.1); acts only if more than 5% of rows are flagged | `retry_eda`: the flagged rows are dropped |
| Degenerate model | Balanced accuracy under 0.55, or recall above 0.97 with specificity under 0.10, or recall under 0.70 | `retry_ml` with the next untried strategy: class weights, then a tuned threshold, then another model |

Safety rules: at most 3 retries, a graph recursion limit of 40, and the graph stops if a retry changes nothing.

**Why v1 had to be rebuilt**

- Its label check removed good rows from a training set of only about 240. On Heart Disease with 15% noise the Reviewer scored MCC 0.562 against 0.651 without it.
- The gate never fired on Amazon or Taxi, so label noise there went unhandled.
- It missed natural leaks, which only show up when a high score collapses after one feature is removed.

**What the fixed Reviewer does**

- **Same:** impossible values, single-feature leaks and the degenerate-model retry work as in v1.
- **New leak check (score jump):** if cross-validated MCC is above 0.85 and dropping one feature lowers it by more than 0.30, that feature is flagged as a leak. It only runs when the score is suspiciously high, so normal data is untouched.
- **New label rule, do no harm:** it tries cutoffs 0.1, 0.2 and 0.3, and removes flagged rows only if repeated cross-validation shows a gain of at least 0.05 MCC and the removal wins in at least 75% of folds. Otherwise the rows are only reported, not removed. In the final run this never passed, so no labels were removed.

**A lesson from building it**

My first version of this test looked good but was fooled. It chose rows to drop using a detector that had already seen the validation rows, so "cleaning" seemed to help even on clean data. The fix was nested cross-validation: inside each fold, the detector sees only that fold's training rows. After that, on clean Heart data the Reviewer scored the same as no Reviewer, which is the correct result.

**How it was tuned**

Only on separate seeds (100 to 104) of Heart Disease. The reported seeds were never used for tuning. Taxi and Amazon were not used for tuning at all.

## 7. Baselines, corruption design and metrics

**Baselines**

- **Always positive:** every row predicted as the positive class. MCC is exactly 0, so any real system has to beat it.
- **Single agent:** `run_single_agent` fills missing values, picks the best of the three models by cross-validated MCC, predicts once. No data cleaning, no checks.
- **Graph without Reviewer:** the graph with the Reviewer switched off. Its scores equal the single agent's, which confirms the graph itself adds no accuracy.
- **AutoGluon:** `run_autogluon` on the same frames, 60 second limit, medium quality preset, balanced accuracy as its metric.

**How the data is corrupted (training set only)**

| Condition | Label noise | Impossible values |
| --- | --- | --- |
| Clean | 0% | 0% |
| Mild | 5% | 2% |
| Medium | 15% | 5% |
| Heavy | 30% | 5% |

- **Label noise:** a random share of training labels are flipped (0 becomes 1, 1 becomes 0).
- **Impossible values:** a random column among the bounded ones is set to -9999 in a random share of rows.
- **Seeded:** the random generator uses the seed, and the flipped and corrupted row ids are saved, so the Reviewer can be scored as a detector (precision and recall of what it found).
- **Leak test:** a feature that is a near-copy of the label (label times 0.9 plus a little noise) is added to both train and test. Any system that does not catch it scores close to 1.0 without any real skill.

**Metrics, and why each one is there**

| Metric | Why |
| --- | --- |
| MCC | The main score. 0 is chance, 1 is perfect, and always-positive gets exactly 0 |
| Balanced accuracy | Average of recall and specificity, so it cannot be gamed by the majority class |
| F1 | Kept for continuity with the first version. Not used to choose models |
| Recall and specificity | Show whether a model predicts everyone positive |
| AUC | How well the scores rank cases, independent of the threshold |
| Train and test scores | Train scores are measured on the corrupted training data, so a train-test gap mixes overfitting with label noise |

## 8. Results of the final full run

The Reviewer does not improve accuracy, but it makes the pipeline safer: it finds every injected impossible value and caught the injected leak in 10 of 10 runs. Heart Disease used 10 seeds, NYC Taxi and Amazon used 3, all on the held-out test set.

**Test MCC, mean over seeds (F1 in brackets)**

Always positive scores MCC 0.000 everywhere, with F1 0.629 (Heart), 0.889 (Taxi) and 0.850 (Amazon). The single agent and the graph without Reviewer are identical.

| Dataset | Label noise | Single agent / graph, no Reviewer | Graph + old Reviewer | Graph + fixed Reviewer |
| --- | --- | --- | --- | --- |
| Heart Disease | 0% | 0.646 (0.797) | 0.638 (0.793) | 0.650 (0.799) |
| Heart Disease | 5% | 0.654 (0.801) | 0.632 (0.790) | 0.654 (0.799) |
| Heart Disease | 15% | 0.651 (0.798) | 0.562 (0.756) | 0.628 (0.787) |
| Heart Disease | 30% | 0.504 (0.723) | 0.443 (0.702) | 0.466 (0.710) |
| Amazon Reviews | 0% | 0.587 (0.898) | 0.587 (0.898) | 0.587 (0.898) |
| Amazon Reviews | 5% | 0.574 (0.893) | 0.577 (0.895) | 0.577 (0.895) |
| Amazon Reviews | 15% | 0.545 (0.882) | 0.553 (0.886) | 0.553 (0.886) |
| Amazon Reviews | 30% | 0.532 (0.889) | 0.532 (0.889) | 0.532 (0.889) |
| NYC Taxi | 0% | 0.202 (0.894) | 0.172 (0.841) | 0.129 (0.806) |
| NYC Taxi | 5% | 0.206 (0.894) | 0.156 (0.860) | 0.156 (0.860) |
| NYC Taxi | 15% | 0.206 (0.894) | 0.157 (0.840) | 0.157 (0.840) |
| NYC Taxi | 30% | 0.206 (0.894) | 0.130 (0.780) | 0.130 (0.780) |

On Heart Disease the spread between seeds is about 0.1 MCC, so differences at 0% to 15% noise are ties.

**AutoGluon**

Not rerun on the same splits. In an earlier run (different seeds, 60 seconds) it scored Heart MCC 0.642 clean, 0.605 at 15% and 0.468 at 30%; Amazon 0.588, 0.573 and 0.517; Taxi 0.139 clean and 0.125 at 15%. Treat those as not directly comparable.

**What the Reviewer found**

| Dataset | Impossible values found | Flipped labels flagged (fixed) | Flipped labels found (old) |
| --- | --- | --- | --- |
| Heart Disease | all (precision 1.0, recall 1.0) | precision 0.41, recall 0.52 | precision 0.50, recall 0.30 |
| Amazon Reviews | all | precision 0.60, recall 0.54 | none |
| NYC Taxi | all | precision 0.40, recall 0.29 | none |

Leak test: the single agent and the graph without Reviewer scored MCC 1.000, which is fake. Both Reviewers reported the honest 0.65, and the fixed Reviewer flagged the leaky feature in 10 of 10 runs.

**What this means**

1. **No clear accuracy winner.** The systems tie within the spread on Heart and Amazon, and the simple models beat AutoGluon on Taxi.
2. **The old Reviewer lost accuracy.** The fix removed most of that damage, but the Reviewer is still slightly below no Reviewer at 15% and 30% noise on Heart (within the spread).
3. **Taxi is a different trade-off.** The fixed Reviewer lowers MCC but raises balanced accuracy (0.569 against 0.527 on clean data), because its degenerate-model retry switches to a balanced model.
4. **The Reviewer's value is data integrity.** It catches impossible values and obvious leaks, and prevents a fake perfect score.
5. **It misses natural leaks.** Nothing removed `payment_type` in the earlier taxi run.
6. **F1 hides the truth.** On Taxi, predicting "tipped" for everyone scores F1 0.889, the best real system scores 0.894.
7. **Flags are not proof.** The Reviewer flags many rows even on clean data (about 270 on Amazon, 28 on Heart, 616 on Taxi), mostly naturally hard rows.

## 9. Step by step: how to rebuild this project from scratch

Follow these in order. Each step ends with something you can check before moving on.

**Part A: Environment**

1. Create a Python 3.11 environment (I used a conda environment named `agent-bench`).
2. Install: `pip install langgraph langchain-ollama scikit-learn pandas numpy matplotlib seaborn shap vaderSentiment pyarrow jupyter`. Add `autogluon.tabular` only for the AutoGluon baseline.
3. Install Ollama, run `ollama serve`, and `ollama pull llama3.2`. Check with `ollama run llama3.2 "say hello"`. Ollama is optional: it only writes the plan note and the summary text.
4. If you work in a notebook, install packages with `%pip install ...` inside it. The notebook kernel can be a different environment from your terminal.

**Part B: The data (section 4)**

1. Download the three datasets and put them where the notebook expects them.
2. Write one loader per dataset that returns a frame and the name of the target column.
3. Check: shapes, positive rates (Heart 0.46, Taxi 0.80, Amazon 0.74) and missing cells (Heart has 6).

**Part C: The fair-benchmark foundation (section 3)**

1. Write the metric function: accuracy, balanced accuracy, recall, specificity, precision, F1, MCC, AUC, PR-AUC.
2. Write `add_rid` (a row id on every row) and a seeded stratified `make_split`.
3. Write the three candidate models: random forest, logistic regression with scaling, histogram gradient boosting. Add a balanced-class-weight option.
4. Write the model chooser: three-fold cross-validated MCC on the training set only.
5. Write the seeded corruption function (label flips and -9999 impossible values, training only, saving the row ids) and the leaked-feature function.
6. Check: the same seed always gives the same split and the same corrupted rows.

**Part D: Baselines**

1. `always_positive`: predict 1 for every row.
2. `run_single_agent`: impute, choose a model, predict once.
3. Check: on clean data the single agent beats always positive on MCC.

**Part E: The LangGraph pipeline (section 5)**

1. Define the shared state as a typed dictionary (train, test, target, pending fixes, history, metrics, retry counters, log).
2. Write the Planner, EDA Analyst, ML Architect, Reviewer and Report Writer nodes.
3. Wire them: planner, eda, ml, reviewer, then a conditional edge to `retry_eda`, `retry_ml` or `pass` (which goes to the report writer and then the end).
4. Set `recursion_limit` to 40 and a retry cap of 3.
5. Check: with the Reviewer switched off, the graph returns exactly the single agent's score.

**Part F: The Reviewer (section 6)**

1. Impossible-value bounds per dataset.
2. Single-feature leak check (AUC above 0.95).
3. Degenerate-model check and the retry strategy list: default, class weight, threshold, alternative model.
4. Check the leak test: with the leaked feature added, the Reviewer should report about 0.65 on Heart and the plain systems 1.0.
5. Only then add the label-noise check, and gate any row removal behind nested cross-validation. Tune only on separate dev seeds.

**Part G: The experiment runner**

1. For each dataset, seed and corruption level: split, corrupt the training set, run every system, save one row of metrics per system to a CSV after every run.
2. Write a summary function that builds mean and standard deviation tables from the CSV.
3. Do a quick run first (2 seeds, small samples) before the full run.

**Part H: The notebook and plots**

1. Walk through: setup, load data, EDA plots, one graph run per dataset with SHAP, full benchmark, tables, charts, leak test, findings.
2. Charts to include: class distribution and correlation, SHAP summary, MCC and F1 by noise level, F1 against MCC against balanced accuracy, train against test, Reviewer detection heatmap, leak test.

**Part I: Running it for real**

1. Run Heart with 10 seeds, and Taxi and Amazon with 3 seeds each. The full benchmark is slow because of the Reviewer's repeated cross-validation.
2. To speed it up, set `ab.cv_gain_from_dropping.__defaults__ = ((0,),)` before the run (one repeat instead of three), and run one dataset per cell so each saves on its own.
3. In my run: Taxi (3 seeds) took 8 minutes and Amazon (3 seeds) 7.3 minutes after the speed-up. The one-run-per-dataset walkthrough took about 2.5 minutes.
4. Optionally run AutoGluon on the same splits (60 seconds per fit).

## 10. Limits, lessons, and what to claim

**What I can claim**

- The Reviewer catches data integrity faults: all injected impossible values, and an injected leaked feature in 10 of 10 runs.
- It prevents a fake result: systems without it reported a perfect MCC of 1.0 on the leak test, and the Reviewer reported the honest 0.65.
- On clean data the fixed Reviewer does no harm on Heart and Amazon.

**What I must not claim**

- That LangGraph beats the single agent or AutoGluon on accuracy. It does not, in these experiments.
- That the Reviewer improves accuracy or fixes label noise. It never removed labels in the final run, and it is slightly lower on Heart at 15% and 30% noise (within the spread).
- That the Reviewer catches natural leaks. It missed `payment_type`.
- That the LLM makes the decisions. Rules route the graph. The LLM only writes the plan note and summary.

**Limits**

- The single agent is a stand-in for a one-pass assistant. I did not run ChatGPT.
- Heart used 10 seeds on 303 rows; Taxi and Amazon used 3 seeds on 5,000-row samples. Small gaps are not reliable.
- The injected leak is easy to spot by design.
- The impossible-value bounds are written by hand per dataset.
- AutoGluon numbers in this document come from an earlier, different run unless rerun on the same splits.
- Train scores are measured on corrupted training data.

**Lessons**

1. Audit your own result before you defend it. A good-looking number with an unfair comparison is worth nothing.
2. Always include a do-nothing baseline. It exposed how weak F1 was.
3. Never choose a model on the test set, and never fill missing values or resample before the split.
4. A check that changes the data must prove it helps, with a test that cannot see the validation rows.
5. Report honest null results. "No accuracy gain, but it catches data faults" is a stronger and more believable finding than a fake win.
6. Save results to CSV as you go and build every table and chart from the files.

## 11. File map, commands and the git workflow

**Files in `research/agentic_benchmark/`**

| File | What it does |
| --- | --- |
| `agentic_bench.py` | The whole harness: loaders, split, metrics, baselines, corruption, Reviewer, LangGraph graph, experiment runner |
| `agentic_viz.py` | Plots and summary tables built from the results |
| `full_study.ipynb` | End-to-end notebook: EDA, walkthrough with SHAP, full benchmark, charts, leak test |
| `run_heart.py`, `run_taxi.py`, `run_benchmark.py`, `run_leak.py`, `run_*_ag.py` | Scripts from the first fair benchmark (one per experiment, AutoGluon in the `_ag` ones) |
| `results/` | Raw CSVs from the first fair benchmark |
| `results_full/` | Raw CSVs and plots from the full study |
| `README.md` | Short results summary |

**Run it**

1. `cd ~/multi-agent-ds-benchmark`
2. Optional: `ollama serve` in another terminal, and set `USE_LLM = True` in the notebook config cell.
3. Open `full_study.ipynb`, edit the three data paths in the first code cell, and run all cells.
4. Run it with `QUICK=1` in the environment first for a short test (2 seeds, small samples).
5. Outputs go to `outputs/` (CSV files) and the EDA and SHAP images land next to the notebook.

**Things that went wrong, and the fixes**

- `ModuleNotFoundError: vaderSentiment` in the notebook: the kernel was a different environment. Fix: `%pip install vaderSentiment` in a notebook cell, then restart the kernel.
- A cell running for 100+ minutes: the Reviewer's repeated cross-validation. Fix: one repeat (`__defaults__` line above), fewer seeds, and one dataset per cell with progress printing.
- Ollama switched off by mistake: only matters when `USE_LLM = True`. Restart it with `ollama serve`.
- Typing a placeholder path like `cd <path to ...>` into the terminal: use the real folder; `find ~ -maxdepth 4 -type d -name "umd_classes*"` locates it.

**Git and pull request workflow**

- Repository: my fork `Under-the-stars/umd_classes`, default branch `master`. It is far behind the upstream repo, so I do not use "Sync fork".
- Work branch: `research/fair-benchmark`. Files live under `research/agentic_benchmark/`.
- Update flow: `git checkout research/fair-benchmark`, copy the files in, `git add research/agentic_benchmark`, commit, then `git push origin research/fair-benchmark`.
- Do not add the `helpers_root` submodule change that `git status` shows.
- Pull requests from a fork go to the upstream repo by default, so create them with `gh pr create --repo <owner/repo> --base master --head <branch>`. PR #920 was opened against the professor's repo, so check its target and the task requirements.
- A CI error about `helpers_root/.python-version` comes from the course repo setup, not from these files.
