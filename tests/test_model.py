"""Fast tests on synthetic data, so they run without the Kaggle dataset."""

import numpy as np
import pandas as pd
import pytest

from credit_model.explain import contributions, what_if
from credit_model.features import CLASSES, FEATURE_NAMES, FEATURES, clean, parse_history_age
from credit_model.model import OrdinalCreditModel


def test_parse_history_age_uses_months_as_twelfths():
    ages = parse_history_age(pd.Series(["22 Years and 11 Months", "22 Years and 2 Months", "NA"]))
    assert ages[0] == pytest.approx(22 + 11 / 12)
    assert ages[0] > ages[1]  # the old "22.11" parsing got this backwards
    assert np.isnan(ages[2])


def test_clean_strips_underscores_and_drops_corrupt_values():
    raw = pd.DataFrame({
        "Customer_ID": ["a", "b"],
        "Credit_Score": ["Good", "Poor"],
        "Delay_from_due_date": [3, 5],
        "Num_of_Delayed_Payment": ["7_", "4"],
        "Payment_of_Min_Amount": ["No", "NM"],
        "Outstanding_Debt": ["809.98_", "1200"],
        "Credit_Utilization_Ratio": [26.8, 31.9],
        "Num_Credit_Inquiries": [4, 2],
        "Interest_Rate": [3, 5000],          # 5000% is corrupt
        "Num_Credit_Card": [4, 1385],        # so is 1,385 cards
        "Num_Bank_Accounts": [3, 2],
        "Credit_History_Age": ["22 Years and 1 Months", None],
        "Annual_Income": ["19114.12_", "34847.84"],
    })
    df = clean(raw)
    assert df.loc[0, "Num_of_Delayed_Payment"] == 7
    assert df.loc[0, "Outstanding_Debt"] == pytest.approx(809.98)
    assert df.loc[0, "Min_Payment_Only"] == 0
    assert np.isnan(df.loc[1, "Min_Payment_Only"])
    assert np.isnan(df.loc[1, "Interest_Rate"])
    assert np.isnan(df.loc[1, "Num_Credit_Card"])


@pytest.fixture(scope="module")
def fitted():
    """A model trained on synthetic data where a hidden risk score drives the label."""
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({f.name: rng.uniform(f.lo, f.hi, n) for f in FEATURES})
    X["Min_Payment_Only"] = rng.integers(0, 2, n).astype(float)
    risk = (X["Delay_from_due_date"] / 60 + X["Outstanding_Debt"] / 5000 + X["Interest_Rate"] / 34
            + X["Min_Payment_Only"] - X["Credit_History_Age"] / 34 + rng.normal(0, 0.2, n))
    y = pd.Series(np.select([risk > 1.6, risk > 0.9], ["Poor", "Standard"], "Good"))
    model = OrdinalCreditModel(random_state=0, max_iter=100).fit(X, y)
    return model, X.median().to_dict()


def test_probabilities_are_valid(fitted):
    model, medians = fitted
    proba = model.predict_proba(pd.DataFrame([medians]))
    assert proba.shape == (1, 3)
    assert proba.sum() == pytest.approx(1.0)
    assert (proba >= 0).all()
    assert list(model.classes_) == CLASSES


@pytest.mark.parametrize("feature", [f for f in FEATURES if f.direction != 0], ids=lambda f: f.name)
def test_monotonic_constraints_hold(fitted, feature):
    model, medians = fitted
    grid = np.linspace(feature.lo, feature.hi, 25)
    batch = pd.DataFrame([dict(medians, **{feature.name: v}) for v in grid])[FEATURE_NAMES]
    score = model.score_value(batch)
    steps = np.diff(score) * feature.direction
    assert (steps >= -1e-9).all(), f"{feature.name} moved the score the wrong way"


def test_contributions_are_zero_for_the_median_applicant(fitted):
    model, medians = fitted
    effects = contributions(model, medians, medians)
    assert effects["effect"].abs().max() == pytest.approx(0, abs=1e-12)


def test_what_if_suggests_improvements_for_a_poor_profile(fitted):
    model, medians = fitted
    poor = dict(medians, Delay_from_due_date=50, Outstanding_Debt=4500, Interest_Rate=30, Min_Payment_Only=1.0)
    assert model.predict(pd.DataFrame([poor]))[0] == "Poor"
    tips = what_if(model, poor)
    assert tips, "expected at least one suggestion"
    for t in tips:
        assert t.feature in {f.name for f in FEATURES if f.actionable}
        changed = model.predict(pd.DataFrame([dict(poor, **{t.feature: _parse(t.target)})]))[0]
        if t.reaches_next_class:
            assert CLASSES.index(changed) > CLASSES.index("Poor")


def _parse(display: str) -> float:
    if display in ("yes", "no"):
        return 1.0 if display == "yes" else 0.0
    return float(display.replace("$", "").replace(",", "").replace("%", "").replace("days", "").replace("yrs", ""))
