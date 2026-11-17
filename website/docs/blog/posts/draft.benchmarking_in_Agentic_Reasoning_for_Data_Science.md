---
title: "Our AI Pipeline Won. Then We Checked the Benchmark."
authors:
  - <author_id_1>
  - <author_id_2>
  - <author_id_3>
date: 2026-11-15
description: "A reviewer agent did not make our data science pipeline more accurate, but it caught broken data that other systems missed."
categories:
  - <allowed category>
---

Our multi-agent pipeline beat AutoGluon at predicting heart disease, 88% F1 against 73%. We were ready to write it up as a win.

Then we looked at the test set. It had 61 rows. The two systems had used different splits, and we had picked the winning model by F1 on that same test set. The win was not real, and we had to rerun the whole project to find out what was.

<!-- more -->

## The short answer

A reviewer agent did not make our pipeline more accurate. It made it safer.

On a fair test, our multi-agent system, a single-pass pipeline and AutoGluon landed within each other's margin of error. What the reviewer did do was catch broken data. It found every impossible value we planted, and it flagged a leaked feature in 10 of 10 runs, where the other systems reported a perfect score of 1.0 that meant nothing. With the leak flagged, the honest score was 0.65.

This matters because agent frameworks promise reliability, and most write-ups prove it with one impressive score. A score tells you how a system did on one dataset on one day. Reliability is different. It means the system notices when something is wrong with its inputs and says so before it reports a number you might act on. That second property is what we set out to measure, and it turned out to be harder to measure than accuracy.

## What we tested

The question: does an AI pipeline with a built-in quality checker give more reliable results than a plain one-pass pipeline, especially when the data is messy?

We built the pipeline as a [LangGraph](https://www.langchain.com/langgraph) graph with five steps: plan, explore the data, train models, review, write a report. The review step is the interesting one. It checks the work and can send the graph back to an earlier step instead of moving on.

We compared it with a single-pass version, with [AutoGluon](https://auto.gluon.ai), and with a model that predicts "yes" for everyone. We ran all of them on three datasets:

- **Heart disease:** the [Cleveland data](https://archive.ics.uci.edu/dataset/45/heart+disease), 303 patients. Tiny and easy to overfit.
- **NYC taxi tips:** a sample of [yellow cab trips](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). Whether a ride was tipped.
- **Amazon Beauty reviews:** a sample of [Amazon reviews](https://amazon-reviews-2023.github.io/). Whether a review is positive.

Then we damaged the training data on purpose: random wrong labels (5%, 15% and 30%), impossible values like an age of -9999, and one fake column that leaks the answer. The test data always stayed clean.

## How the pipeline works

Each of the five steps has one job.

- **Planner:** looks at the data and writes down the risks, such as unbalanced classes or missing values. A language model can add a one-line plan, but nothing depends on it.
- **EDA analyst:** fills in missing values using the training rows only, then applies any fixes the reviewer asks for, such as dropping a row or a column.
- **ML architect:** tries three standard models (random forest, logistic regression and gradient boosting), picks the best one by cross-validated MCC on the training data, and scores it on the held-out test set.
- **Quality reviewer:** the checker. It decides whether the result can be trusted and, if not, where to send the graph back to.
- **Report writer:** records the plan, what was fixed, the metrics and the full log.

```mermaid
flowchart LR
  A[Planner] --> B[EDA Analyst]
  B --> C[ML Architect]
  C --> D{Quality Reviewer}
  D -- pass --> E[Report Writer]
  D -- retry_eda --> B
  D -- retry_ml --> C
```

The two arrows that go back are the point. A plain pipeline runs once and reports whatever comes out. This one can stop, fix the data and try again, up to three times.

**What the reviewer checks**

1. **Impossible values.** An age of -9999 or a negative fare. The reviewer has hand-written limits for each dataset, and it sends the graph back to drop those rows.
2. **A leaked feature.** One column that almost gives away the answer on its own. If dropping a single column makes a suspiciously high score collapse, that column is the leak.
3. **Wrong labels.** Rows where a model trained on the rest of the data strongly disagrees with the given label.
4. **A model that just says yes.** If almost every prediction is positive, or recall and specificity are badly lopsided, the reviewer retries with class weights, a tuned threshold or a different model.

If a retry changes nothing, the graph stops and says so instead of looping.

## Why our first result was wrong

Three things went wrong at once:

- **Different splits and models.** Each system got its own train/test split and its own models, so a score gap could come from anywhere.
- **A 61-row test set.** One extra correct prediction moves the score by about 1.6 points.
- **F1 as the judge.** F1 rewards a model that says "positive" for everyone, and we were also choosing models by F1 on the test set.

The F1 problem is the easiest to see. About 80% of NYC taxi rides are tipped, so a model that always answers "tipped" looks great:

| System (NYC taxi) | F1 | MCC |
|---|---|---|
| Always answers "tipped" | 0.889 | 0.000 |
| Real model | 0.894 | 0.202 |

By F1 they are nearly tied. By [MCC](https://en.wikipedia.org/wiki/Phi_coefficient), a score where 0 means guessing and 1 means perfect, the do-nothing model gets exactly 0. So we rebuilt the benchmark with one split for every system, models chosen only on training data, seeded corruption, many random seeds, and MCC as the main score.

## What the fair test shows

On accuracy, nobody clearly wins. Here is heart disease, scored with MCC on the clean test set, averaged over 10 random splits:

| Wrong labels in training data | Single pass | Reviewer (first version) | Reviewer (rebuilt) |
|---|---|---|---|
| 0% | 0.646 | 0.638 | 0.650 |
| 15% | 0.651 | 0.562 | 0.628 |
| 30% | 0.504 | 0.443 | 0.466 |

Our first reviewer made things worse. It deleted rows it thought were mislabeled, and with only about 240 training rows it threw away good data. We rebuilt it so it removes rows only if a leak-free cross-validation test shows that helps. That fixed most of the damage, but it still does not beat the single pass. On Amazon reviews all systems tie. On taxi tips the rebuilt reviewer scores lower on MCC, because it forces a more balanced model.

**Where the reviewer earns its place**

- **Impossible values:** it found all of them, on all three datasets.
- **Leaks:** with a fake leaked column added, the single pass and the plain graph scored a perfect MCC of 1.0. The reviewer flagged the column in 10 of 10 runs and reported the true 0.65.
- **Natural leaks:** it missed a real one. In the taxi data, the payment type almost gives away the tip, since cash tips are not recorded. Keeping that column lifted every system from 0.21 to 0.91, and nothing caught it.

The agent is not smarter. It is more careful. And its routing is rule-based: the language model only writes the plan and the summary.

**A closer look at the label checks**

The reviewer could often spot suspicious labels. On the Amazon reviews, the rows it flagged were mislabeled about 60% of the time, and it found about 54% of the wrong labels we had planted. But removing the flagged rows never improved the final score. About 40% of what it flagged was fine, and the models cope with the rest well enough anyway. Spotting a problem and fixing it are two different skills.

Building the rebuilt check also taught us something about our own testing. Our first version of the "does removing rows help?" test looked great, even on clean data. It was fooled. The detector that picked the rows to drop had already seen the validation rows, so cleaning looked helpful when it was not. The fix was to repeat the whole detection step inside every cross-validation fold, so nothing from the validation rows leaks in. After that, on clean data the reviewer scored exactly the same as having no reviewer, which is the right answer.

## What to take from this

If you test an AI pipeline, a few habits would have saved us a rerun:

1. **Add a do-nothing baseline.** If "always yes" scores nearly as well, your metric is hiding something.
2. **Give every system the same split.** Choose models on training data only, and never on the test set.
3. **Plant faults and see who notices.** A pipeline that reports a perfect score on leaked data is the one to worry about.
4. **Report the null result.** "No accuracy gain, but it catches broken data" is less exciting than a win, and much more useful.

The limits are real. Heart disease used 10 random splits, but taxi and Amazon used only 3, on 5,000-row samples, and the leak we planted is easy to spot.

Next we want to run AutoGluon on exactly the same splits and corruption, test more seeds on the taxi and Amazon data, let the language model choose the retry strategy instead of fixed rules, and look for a check that catches natural leaks like the payment type, not just planted ones.

The code, the results and a step-by-step write-up are in the [project folder](https://github.com/Under-the-stars/umd_classes/tree/research/fair-benchmark/research/agentic_benchmark). If you have run a similar comparison, or can see a flaw in ours, we would like to hear about it in the comments.
