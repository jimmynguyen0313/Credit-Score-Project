"""Explanations and what-if suggestions for a single applicant.

Both work by querying the model, so they apply to any model with a
`score_value` method:

* contributions(): how much each input moves the score compared with a
  typical applicant (that input set to the training median).
* what_if(): the smallest single change to an actionable input that would
  move the prediction up a class. If no single change gets there, it
  returns the changes that raise the chance of the next class the most.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import CLASSES, FEATURES


@dataclass
class Suggestion:
    feature: str
    label: str
    current: str
    target: str
    new_class: str
    new_class_proba: float
    effort: float  # fraction of the input's range that has to change; used for sorting
    reaches_next_class: bool = True
    target_class: str = ""
    target_proba_before: float = 0.0
    target_proba_after: float = 0.0


def _frame(values: dict) -> pd.DataFrame:
    return pd.DataFrame([{f.name: values[f.name] for f in FEATURES}])


def contributions(model, values: dict, medians: dict) -> pd.DataFrame:
    """Per-input effect on the expected score (0 = Poor … 2 = Good) vs. the median applicant."""
    base = model.score_value(_frame(values))[0]
    rows = []
    for f in FEATURES:
        swapped = dict(values, **{f.name: medians[f.name]})
        delta = base - model.score_value(_frame(swapped))[0]
        rows.append({"feature": f.name, "label": f.label, "effect": float(delta)})
    out = pd.DataFrame(rows)
    return out.reindex(out["effect"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def what_if(model, values: dict, max_suggestions: int = 3, steps: int = 60) -> list[Suggestion]:
    """Smallest single-input changes that would lift the predicted class by one level."""
    current_class = model.predict(_frame(values))[0]
    rank = CLASSES.index(current_class)
    if rank == len(CLASSES) - 1:
        return []
    target_class = CLASSES[rank + 1]

    before = model.predict_proba(_frame(values))[0][rank + 1]
    flips, partial = [], []
    for f in FEATURES:
        if not f.actionable or f.direction == 0:
            continue
        start = values[f.name]
        end = f.lo if f.direction < 0 else f.hi
        if start == end:
            continue

        # Walk from the current value toward the "better" end of the range in one batch.
        grid = np.linspace(start, end, steps + 1)[1:]
        if f.step >= 1:
            grid = np.unique(np.round(grid / f.step) * f.step)
            grid = grid[grid != start]
            grid = grid[np.argsort(np.abs(grid - start))]
        if len(grid) == 0:
            continue
        batch = pd.concat([_frame(dict(values, **{f.name: v})) for v in grid], ignore_index=True)
        proba = model.predict_proba(batch)
        reached = np.flatnonzero(proba.argmax(axis=1) >= rank + 1)

        if len(reached):
            i, bucket = reached[0], flips
        else:
            i, bucket = int(proba[:, rank + 1].argmax()), partial
            if proba[i, rank + 1] - before < 0.02:
                continue
        bucket.append(Suggestion(
            feature=f.name,
            label=f.label,
            current=_display(f, start),
            target=_display(f, grid[i]),
            new_class=CLASSES[int(proba[i].argmax())],
            new_class_proba=float(proba[i].max()),
            effort=abs(grid[i] - start) / (f.hi - f.lo),
            reaches_next_class=bool(len(reached)),
            target_class=target_class,
            target_proba_before=float(before),
            target_proba_after=float(proba[i, rank + 1]),
        ))

    if flips:
        flips.sort(key=lambda s: s.effort)
        return flips[:max_suggestions]
    partial.sort(key=lambda s: s.target_proba_after, reverse=True)
    return partial[:max_suggestions]


def _display(f, value: float) -> str:
    if f.name == "Min_Payment_Only":
        return "yes" if value >= 0.5 else "no"
    return f.show(value)
