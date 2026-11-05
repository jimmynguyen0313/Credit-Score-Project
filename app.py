"""Credit Score Explorer — Streamlit app.

    streamlit run app.py
"""

import json
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from credit_model.explain import contributions, what_if
from credit_model.features import BY_NAME, CLASSES

st.set_page_config(page_title="Credit Score Explorer", page_icon="◐", layout="wide")

COLORS = {"Poor": "#f2777a", "Standard": "#f5c26b", "Good": "#5fd3a2"}
ACCENT, MUTED, BORDER, SURFACE = "#7c9cff", "#a3a8b4", "#2a2e36", "#181b21"

st.html(f"""
<style>
  .block-container {{ max-width: 1120px; padding-top: 2.5rem; padding-bottom: 4rem; }}
  [data-testid="stHeader"] {{ background: transparent; }}
  h1, h2, h3 {{ letter-spacing: -0.02em; }}
  .eyebrow {{ font-family: ui-monospace, Menlo, monospace; font-size: .8rem; color: {ACCENT}; margin-bottom: .25rem; }}
  .lede {{ color: {MUTED}; font-size: 1.05rem; max-width: 60ch; margin-top: -.5rem; }}
  .section {{ font-size: .75rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
              color: {MUTED}; margin: 1.25rem 0 .25rem; }}
  .section:first-child {{ margin-top: 0; }}
  .result-label {{ font-size: .8rem; color: {MUTED}; text-transform: uppercase; letter-spacing: .08em; }}
  .result-class {{ font-size: 3rem; font-weight: 700; line-height: 1.1; letter-spacing: -0.03em; margin: .1rem 0 .2rem; }}
  .result-sub {{ color: {MUTED}; font-size: .95rem; }}
  .bar {{ display: flex; height: 10px; border-radius: 999px; overflow: hidden; margin: 1.1rem 0 .5rem; background: {BORDER}; }}
  .bar span {{ display: block; height: 100%; }}
  .legend {{ display: flex; gap: 1.25rem; font-size: .85rem; color: {MUTED}; }}
  .legend i {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: .4rem; }}
  .tip {{ border: 1px solid {BORDER}; border-radius: 12px; padding: .8rem 1rem; margin-bottom: .6rem; background: {SURFACE}; }}
  .tip b {{ font-weight: 600; }}
  .tip .arrow {{ color: {ACCENT}; font-weight: 600; }}
  .tip small {{ display: block; color: {MUTED}; margin-top: .15rem; }}
  .note {{ color: {MUTED}; font-size: .85rem; }}
  [data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
</style>
""")


@st.cache_resource
def load_model():
    return joblib.load("models/credit_model.joblib")


@st.cache_data
def load_metrics():
    return json.loads(Path("models/metrics.json").read_text())


bundle = load_model()
model, medians = bundle["model"], bundle["medians"]

# ── Header ───────────────────────────────────────────────────────────────
st.html('<div class="eyebrow">Machine learning demo</div>')
st.title("Credit Score Explorer")
st.html('<p class="lede">Adjust a financial profile to see the predicted credit score band, '
        'what is driving it, and the smallest change that would improve it.</p>')

explore, about = st.tabs(["Explore", "How the model works"])

# ── Inputs ───────────────────────────────────────────────────────────────

PRESETS = {
    "Strong": dict(Delay_from_due_date=3, Num_of_Delayed_Payment=2, Min_Payment_Only=0.0, Outstanding_Debt=400,
                   Credit_Utilization_Ratio=27.0, Num_Credit_Inquiries=2, Interest_Rate=6, Num_Credit_Card=4,
                   Num_Bank_Accounts=3, Credit_History_Age=24.0, Annual_Income=72000),
    "Typical": None,  # filled from training medians
    "Struggling": dict(Delay_from_due_date=32, Num_of_Delayed_Payment=19, Min_Payment_Only=1.0, Outstanding_Debt=3400,
                       Credit_Utilization_Ratio=36.0, Num_Credit_Inquiries=10, Interest_Rate=24, Num_Credit_Card=7,
                       Num_Bank_Accounts=7, Credit_History_Age=9.0, Annual_Income=28000),
}


def snap(name: str, value: float) -> float:
    f = BY_NAME[name]
    v = round(value / f.step) * f.step
    return int(v) if f.step >= 1 else float(v)


def apply_preset():
    choice = st.session_state.get("preset")
    if not choice:
        return
    profile = PRESETS[choice] or medians
    for name, value in profile.items():
        st.session_state[name] = bool(value >= 0.5) if name == "Min_Payment_Only" else snap(name, value)


if "Delay_from_due_date" not in st.session_state:
    st.session_state["preset"] = "Struggling"
    apply_preset()


SLIDER_FORMAT = {"Outstanding_Debt": "$%d", "Annual_Income": "$%d", "Credit_Utilization_Ratio": "%.1f%%",
                 "Interest_Rate": "%d%%", "Credit_History_Age": "%.1f yrs", "Delay_from_due_date": "%d days"}


def slider(name: str, label: str | None = None):
    f = BY_NAME[name]
    lo, hi, step = (int(f.lo), int(f.hi), int(f.step)) if f.step >= 1 else (float(f.lo), float(f.hi), float(f.step))
    return st.slider(label or f.label, lo, hi, step=step, key=name, format=SLIDER_FORMAT.get(name))


with explore:
    left, gap, right = st.columns([1, 0.08, 1.25])

    with left:
        st.segmented_control("Start from a sample profile", list(PRESETS), key="preset",
                             on_change=apply_preset, selection_mode="single")
        with st.container(border=True):
            st.html('<div class="section">Payment history</div>')
            delay = slider("Delay_from_due_date")
            late = slider("Num_of_Delayed_Payment")
            min_only = st.toggle("Usually pays only the minimum", key="Min_Payment_Only")

            st.html('<div class="section">Debt &amp; credit</div>')
            debt = slider("Outstanding_Debt", "Outstanding debt")
            util = slider("Credit_Utilization_Ratio", "Credit utilization")
            inquiries = slider("Num_Credit_Inquiries")
            rate = slider("Interest_Rate", "Average interest rate")
            c1, c2 = st.columns(2)
            with c1:
                cards = slider("Num_Credit_Card")
            with c2:
                banks = slider("Num_Bank_Accounts")

            st.html('<div class="section">Background</div>')
            history = slider("Credit_History_Age", "Credit history length")
            income = slider("Annual_Income", "Annual income")

    values = {
        "Delay_from_due_date": delay, "Num_of_Delayed_Payment": late, "Min_Payment_Only": float(min_only),
        "Outstanding_Debt": debt, "Credit_Utilization_Ratio": util, "Num_Credit_Inquiries": inquiries,
        "Interest_Rate": rate, "Num_Credit_Card": cards, "Num_Bank_Accounts": banks,
        "Credit_History_Age": history, "Annual_Income": income,
    }
    proba = model.predict_proba(pd.DataFrame([values]))[0]
    predicted = CLASSES[int(proba.argmax())]

    # ── Result ───────────────────────────────────────────────────────────
    with right:
        bar = "".join(f'<span style="width:{p * 100:.1f}%;background:{COLORS[c]}"></span>'
                      for c, p in zip(CLASSES, proba))
        legend = "".join(f'<span><i style="background:{COLORS[c]}"></i>{c} {p:.0%}</span>'
                         for c, p in zip(CLASSES, proba))
        st.html(f"""
          <div class="result-label">Predicted band</div>
          <div class="result-class" style="color:{COLORS[predicted]}">{predicted}</div>
          <div class="result-sub">{proba.max():.0%} confidence</div>
          <div class="bar">{bar}</div>
          <div class="legend">{legend}</div>
        """)

        st.subheader("What's driving it", anchor=False)
        st.caption("Each input's effect compared with a typical applicant. Right helps, left hurts.")
        effects = contributions(model, values, medians)
        effects = effects[effects["effect"].abs() > 0.005].head(6)
        if effects.empty:
            st.html('<p class="note">This profile is close to the typical applicant, '
                    'so no single input stands out.</p>')
        else:
            effects["direction"] = effects["effect"].map(lambda v: "Helps" if v > 0 else "Hurts")
            chart = (
                alt.Chart(effects)
                .mark_bar(cornerRadius=4, height=16)
                .encode(
                    x=alt.X("effect:Q", title=None, axis=alt.Axis(labels=False, ticks=False, grid=False, domain=False)),
                    y=alt.Y("label:N", sort=None, title=None,
                            axis=alt.Axis(labelLimit=260, labelColor=MUTED, labelFontSize=13, ticks=False, domain=False)),
                    color=alt.Color("direction:N", legend=None,
                                    scale=alt.Scale(domain=["Helps", "Hurts"], range=[COLORS["Good"], COLORS["Poor"]])),
                    tooltip=[alt.Tooltip("label:N", title="Input"), alt.Tooltip("effect:Q", title="Effect", format="+.2f")],
                )
                .properties(height=40 * len(effects))
                .configure_view(stroke=None)
                .configure(background="transparent")
            )
            st.altair_chart(chart, use_container_width=True)

        st.subheader("How to improve it", anchor=False)
        tips = what_if(model, values)
        if predicted == CLASSES[-1]:
            st.html('<p class="note">Already in the top band. Keep payments on time and debt low.</p>')
        elif not tips:
            st.html('<p class="note">No single change makes a meaningful difference here. '
                    'Try combining improvements on the left.</p>')
        else:
            for t in tips:
                if t.reaches_next_class:
                    outcome = f"moves you to <b>{t.new_class}</b> ({t.new_class_proba:.0%} confidence)"
                else:
                    outcome = (f"raises your chance of <b>{t.target_class}</b> "
                               f"from {t.target_proba_before:.0%} to {t.target_proba_after:.0%}")
                st.html(f'<div class="tip"><b>{t.label}</b>: {t.current} '
                        f'<span class="arrow">→</span> {t.target}<small>{outcome}</small></div>')
            if not tips[0].reaches_next_class:
                st.caption(f"No single change reaches {tips[0].target_class} on its own; these help the most.")

# ── About the model ──────────────────────────────────────────────────────
with about:
    m = load_metrics()
    st.subheader("Results on customers the model has never seen", anchor=False)
    st.write(
        f"Trained on {m['rows']:,} monthly records from {m['customers']:,} customers in the "
        "[Kaggle credit score classification dataset](https://www.kaggle.com/datasets/parisrohan/credit-score-classification). "
        "The data has eight rows per customer, so the test set holds out whole customers. "
        "A random row split would let the model see the same people during training and overstate accuracy."
    )

    results = pd.DataFrame(m["results"]).rename(columns={"model": "Model", "accuracy": "Accuracy", "macro_f1": "Macro F1"})
    st.dataframe(
        results, hide_index=True, use_container_width=True,
        column_config={
            "Accuracy": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
            "Macro F1": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
        },
    )

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.subheader("Why an ordinal, monotonic model?", anchor=False)
        st.write(
            "Credit bands are ordered (Poor < Standard < Good), so the model is two gradient-boosted "
            "classifiers, one for *at least Standard* and one for *Good*, combined into "
            "probabilities. Each is constrained so that, for example, more late payments or more debt "
            "can never improve the prediction."
        )
        st.write(
            "That costs about one point of accuracy against an unconstrained model. In return the "
            "explanations and suggestions always point in a direction that makes sense."
        )
    with c2:
        st.subheader("Confusion matrix", anchor=False)
        labels, matrix = m["confusion_matrix"]["labels"], m["confusion_matrix"]["matrix"]
        cm = pd.DataFrame([
            {"Actual": a, "Predicted": p, "Count": matrix[i][j], "Share": matrix[i][j] / max(sum(matrix[i]), 1)}
            for i, a in enumerate(labels) for j, p in enumerate(labels)
        ])
        base = alt.Chart(cm).encode(
            x=alt.X("Predicted:N", sort=labels, axis=alt.Axis(labelAngle=0, labelColor=MUTED, titleColor=MUTED)),
            y=alt.Y("Actual:N", sort=labels, axis=alt.Axis(labelColor=MUTED, titleColor=MUTED)),
        )
        heat = base.mark_rect(cornerRadius=4).encode(
            color=alt.Color("Share:Q", legend=None, scale=alt.Scale(range=[SURFACE, ACCENT])))
        text = base.mark_text(fontSize=13).encode(
            text=alt.Text("Share:Q", format=".0%"),
            color=alt.condition("datum.Share > 0.5", alt.value("#0f1115"), alt.value("#e8e9ec")))
        st.altair_chart((heat + text).properties(height=240).configure_view(stroke=None)
                        .configure(background="transparent"), use_container_width=True)
        st.caption("Rows sum to 100%: how each actual band was classified.")

    st.html('<p class="note">An educational project on a public, partly synthetic dataset. '
            'Not financial advice and not a real credit score.</p>')
