"""Loading and cleaning the Kaggle "Credit score classification" dataset.

The raw data is deliberately messy: numbers with stray underscores
("1234_"), corrupted outliers (e.g. 1,000+ credit cards), and credit
history stored as text ("22 Years and 11 Months"). Corrupt values are
turned into NaN rather than dropping whole rows; the models handle
missing values natively.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Ordered worst → best. The models predict on this ordinal scale.
CLASSES = ["Poor", "Standard", "Good"]


@dataclass(frozen=True)
class Feature:
    name: str           # column name in the cleaned frame
    label: str          # human-readable label for the UI
    lo: float           # plausible range; values outside are treated as corrupt
    hi: float
    direction: int      # +1: higher is better for the score, -1: worse, 0: no constraint
    step: float = 1.0
    fmt: str = "{:,.0f}"
    unit: str = ""
    actionable: bool = True  # can a person realistically change it?

    def show(self, value: float) -> str:
        return self.fmt.format(value) + self.unit


FEATURES = [
    Feature("Delay_from_due_date", "Average days late on payments", 0, 60, -1, unit=" days"),
    Feature("Num_of_Delayed_Payment", "Number of late payments", 0, 28, -1),
    Feature("Min_Payment_Only", "Usually pays only the minimum", 0, 1, -1, fmt="{:.0f}"),
    Feature("Outstanding_Debt", "Outstanding debt", 0, 5000, -1, step=50, fmt="${:,.0f}"),
    Feature("Credit_Utilization_Ratio", "Credit utilization", 20, 50, -1, step=0.5, fmt="{:.1f}", unit="%"),
    Feature("Num_Credit_Inquiries", "Recent credit inquiries", 0, 17, -1),
    Feature("Interest_Rate", "Average interest rate", 1, 34, -1, unit="%"),
    Feature("Num_Credit_Card", "Credit cards", 0, 11, -1, actionable=False),
    Feature("Num_Bank_Accounts", "Bank accounts", 0, 11, -1, actionable=False),
    Feature("Credit_History_Age", "Credit history length", 0, 34, +1, step=0.5, fmt="{:.1f}", unit=" yrs", actionable=False),
    Feature("Annual_Income", "Annual income", 7000, 180000, +1, step=1000, fmt="${:,.0f}", actionable=False),
]
FEATURE_NAMES = [f.name for f in FEATURES]
BY_NAME = {f.name: f for f in FEATURES}


def _number(s: pd.Series) -> pd.Series:
    """Parse numbers like '1234_' or ' 56 ' to float; anything else → NaN."""
    return pd.to_numeric(s.astype(str).str.replace("_", "", regex=False).str.strip(), errors="coerce")


def parse_history_age(s: pd.Series) -> pd.Series:
    """'22 Years and 11 Months' → 22.9167 (years as a float)."""
    parts = s.astype(str).str.extract(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?").astype(float)
    return parts[0] + parts[1] / 12


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw Kaggle frame into model features + Customer_ID + Credit_Score."""
    df = pd.DataFrame(index=raw.index)
    for f in FEATURES:
        if f.name == "Credit_History_Age":
            df[f.name] = parse_history_age(raw[f.name])
        elif f.name == "Min_Payment_Only":
            df[f.name] = raw["Payment_of_Min_Amount"].map({"No": 0.0, "Yes": 1.0})
        else:
            df[f.name] = _number(raw[f.name])
        out_of_range = (df[f.name] < f.lo) | (df[f.name] > f.hi)
        df.loc[out_of_range, f.name] = np.nan

    df["Customer_ID"] = raw["Customer_ID"]
    df["Credit_Score"] = raw["Credit_Score"]
    return df[df["Credit_Score"].isin(CLASSES)]


def load(path) -> pd.DataFrame:
    return clean(pd.read_csv(path, low_memory=False))
