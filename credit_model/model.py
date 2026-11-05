"""Ordinal credit score model with monotonic constraints.

Credit scores are ordered (Poor < Standard < Good), so instead of one
3-class model we train two binary gradient-boosted models:

    P(score >= Standard)   and   P(score == Good)

and combine them into class probabilities (Frank & Hall, 2001).

Binary models let us enforce monotonic constraints — e.g. more late
payments can never *improve* the predicted score — which a plain
multiclass model can't guarantee. That makes the explanations and
"what-if" suggestions trustworthy.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import CLASSES, FEATURE_NAMES, FEATURES


class OrdinalCreditModel:
    classes_ = np.array(CLASSES)

    def __init__(self, random_state: int = 0, **hgb_params):
        params = dict(learning_rate=0.08, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0,
                      early_stopping=True, random_state=random_state)
        params.update(hgb_params)
        monotonic = [f.direction for f in FEATURES]
        self.at_least_standard = HistGradientBoostingClassifier(monotonic_cst=monotonic, **params)
        self.good = HistGradientBoostingClassifier(monotonic_cst=monotonic, **params)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X[FEATURE_NAMES]
        rank = y.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
        self.at_least_standard.fit(X, rank >= 1)
        self.good.fit(X, rank >= 2)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[FEATURE_NAMES]
        p_ge_std = self.at_least_standard.predict_proba(X)[:, 1]
        p_good = self.good.predict_proba(X)[:, 1]
        p_good = np.minimum(p_good, p_ge_std)  # keep the two estimates consistent
        proba = np.column_stack([1 - p_ge_std, p_ge_std - p_good, p_good])
        return np.clip(proba, 0, 1)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def score_value(self, X: pd.DataFrame) -> np.ndarray:
        """Expected rank on a 0 (Poor) – 2 (Good) scale; a smooth 'how good' number."""
        return self.predict_proba(X) @ np.arange(len(CLASSES))
