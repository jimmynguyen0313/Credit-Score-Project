# Credit Score Explorer

An interactive machine learning app that predicts a credit score band (**Poor**, **Standard** or **Good**) from a financial profile. It explains which inputs drive the prediction and finds the smallest change that would improve it.

**[▶ Live demo](#deploying)** · Python · scikit-learn · Streamlit

## Features

- **Prediction with confidence**: the predicted band, plus the probability of each band.
- **What's driving it**: how much each input helps or hurts compared with a typical applicant.
- **How to improve it**: the smallest single change, e.g. *outstanding debt $3,400 → $1,200*, that moves the prediction up a band. When no single change is enough, it shows the changes that help most.
- **Sample profiles**: Strong, Typical and Struggling starting points.
- **How the model works**: model comparison, confusion matrix and the reasoning behind the design.

## Results

Evaluated on **customers the model has never seen**. The dataset has 8 monthly rows per customer, so the train/test split is by customer. A random row split puts the same people on both sides and overstates accuracy.

| Model | Accuracy | Macro F1 |
|---|---|---|
| Baseline (always "Standard") | 52.9% | 23.1% |
| Logistic regression | 62.2% | 58.1% |
| Gradient boosting (unconstrained) | 67.4% | 64.7% |
| **Ordinal gradient boosting (monotonic)**, used in the app | **66.2%** | **63.7%** |

Run `python train.py` to reproduce these numbers. They're also saved in [`models/metrics.json`](models/metrics.json).

## How it works

**Ordinal model with monotonic constraints.** Credit bands are ordered, so instead of one 3-class model the app uses two gradient-boosted binary classifiers:

- one predicts P(score ≥ Standard)
- one predicts P(score = Good)

Their outputs are combined into class probabilities. Binary models support monotonic constraints, so the model is guaranteed never to reward more late payments, more debt or a higher interest rate. That costs about one point of accuracy against an unconstrained model. In return, the explanations and suggestions always point in a sensible direction.

**Explanations** are model-agnostic:
- An input's effect is the change in the expected score when that input is reset to the median applicant's value.
- What-if suggestions search each actionable input toward its "better" end for the first value that changes the predicted band.

**Data cleaning** handles the dataset's deliberate mess:
- numbers with stray underscores (`"1234_"`)
- corrupted outliers (e.g. 1,385 credit cards), set to missing rather than dropping the row
- credit history stored as text (`"22 Years and 11 Months"` → 22.92 years)

## Project structure

```
app.py                  Streamlit app
train.py                Train, evaluate and save the model
credit_model/
  features.py           Feature definitions, ranges and data cleaning
  model.py              Ordinal monotonic gradient-boosting model
  explain.py            Contributions and what-if suggestions
models/
  credit_model.joblib   Trained model used by the app
  metrics.json          Evaluation results shown in the app
tests/                  Unit tests (run on synthetic data, no download needed)
```

## Running locally

```bash
pip install -r requirements-dev.txt
streamlit run app.py
```

The trained model is committed, so the app runs without the dataset.

### Retraining

1. Download `train.csv` from the [Kaggle credit score classification dataset](https://www.kaggle.com/datasets/parisrohan/credit-score-classification).
2. Save it as `data/train.csv` (the `data/` folder is gitignored).
3. Run:

```bash
python train.py
```

### Tests

```bash
pytest
```

## Deploying

The app runs on [Streamlit Community Cloud](https://streamlit.io/cloud) for free:

1. Sign in with GitHub and choose **Create app**.
2. Pick this repository, branch `main`, main file `app.py`.
3. Under **Advanced settings**, choose Python 3.13.

## Background

This started as a team project that compared logistic regression and SVMs on five features. The current version rebuilds it:
- reproducible training
- a customer-level evaluation
- a parsing fix for credit history
- a constrained ordinal model
- an explainable interface

*An educational project on a public, partly synthetic dataset. Not financial advice and not a real credit score.*
