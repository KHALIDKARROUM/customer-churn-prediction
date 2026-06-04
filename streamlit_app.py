from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

import dashboard_core as core


st.set_page_config(
    page_title="Retention Operations Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
        --page: #f6f8fb;
        --surface: #ffffff;
        --surface-soft: #f2f5f9;
        --ink: #18212f;
        --muted: #64748b;
        --line: #d9e2ec;
        --line-strong: #c7d3e1;
        --blue: #2563eb;
        --teal: #0f9f8f;
        --coral: #e85d75;
        --amber: #d99a18;
        --green: #18a058;
        --shadow: 0 14px 34px rgba(24, 33, 47, 0.08);
    }

    .stApp {
        background: var(--page);
        color: var(--ink);
    }

    .main .block-container {
        max-width: 1480px;
        padding: 1.2rem 1.4rem 2.6rem;
    }

    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--line);
    }

    div[data-testid="stSidebarHeader"] {
        display: none;
    }

    .side-brand {
        align-items: center;
        display: flex;
        gap: 0.75rem;
        padding: 1.25rem 0 1.5rem;
    }

    .brand-mark {
        align-items: center;
        background: linear-gradient(135deg, var(--blue), var(--teal));
        border-radius: 8px;
        color: #fff;
        display: inline-flex;
        font-weight: 900;
        height: 40px;
        justify-content: center;
        width: 40px;
    }

    .side-brand strong,
    .side-brand span {
        display: block;
    }

    .side-brand span,
    .eyebrow,
    .kpi-card span,
    .feature-row span,
    .model-metric span {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 800;
        text-transform: uppercase;
    }

    .side-brand .brand-mark {
        color: #ffffff;
        display: inline-flex;
        font-size: 1rem;
    }

    div[role="radiogroup"] label {
        border-radius: 6px;
        min-height: 2.35rem;
        padding: 0.34rem 0.55rem;
    }

    .topbar {
        align-items: center;
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow);
        display: flex;
        gap: 1rem;
        justify-content: space-between;
        margin-bottom: 1.2rem;
        min-height: 4rem;
        padding: 0.8rem 1rem;
    }

    .topbar h1 {
        color: var(--ink);
        font-size: 1.18rem;
        line-height: 1.1;
        margin: 0;
    }

    .search-shell {
        background: var(--surface-soft);
        border: 1px solid var(--line);
        border-radius: 8px;
        color: #8898ad;
        flex: 1;
        font-size: 0.9rem;
        max-width: 420px;
        min-height: 2.45rem;
        padding: 0.65rem 0.85rem;
    }

    .model-badge {
        display: grid;
        gap: 0.1rem;
        text-align: right;
    }

    .model-badge span {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 800;
        text-transform: uppercase;
    }

    .model-badge strong {
        color: var(--teal);
        font-size: 0.92rem;
    }

    .page-heading {
        align-items: end;
        display: flex;
        gap: 1rem;
        justify-content: space-between;
        margin-bottom: 1rem;
    }

    .page-heading h2 {
        color: var(--ink);
        font-size: clamp(2rem, 4vw, 3rem);
        line-height: 1.05;
        margin: 0.45rem 0 0;
        overflow-wrap: anywhere;
    }

    .page-heading p {
        color: var(--muted);
        line-height: 1.5;
        margin: 0.65rem 0 0;
        max-width: 760px;
    }

    .kpi-card,
    .html-panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow);
    }

    .kpi-card {
        border-top: 4px solid var(--blue);
        min-height: 7.8rem;
        padding: 1.05rem;
    }

    .kpi-card.coral {
        border-top-color: var(--coral);
    }

    .kpi-card.green {
        border-top-color: var(--green);
    }

    .kpi-card.amber {
        border-top-color: var(--amber);
    }

    .kpi-card strong {
        color: var(--ink);
        display: block;
        font-size: 2.15rem;
        line-height: 1;
        margin: 1.05rem 0 0.45rem;
    }

    .kpi-card small {
        color: var(--muted);
        display: block;
        font-size: 0.86rem;
    }

    .panel-heading {
        align-items: center;
        border-bottom: 1px solid var(--line);
        display: flex;
        justify-content: space-between;
        min-height: 3.7rem;
        padding: 0.95rem 1.05rem;
    }

    .panel-heading h3 {
        color: var(--ink);
        font-size: 1rem;
        margin: 0.25rem 0 0;
    }

    .segmented {
        background: var(--surface-soft);
        border: 1px solid var(--line);
        border-radius: 7px;
        display: flex;
        padding: 3px;
    }

    .segmented span {
        border-radius: 5px;
        color: var(--muted);
        font-size: 0.8rem;
        font-weight: 800;
        padding: 0.35rem 0.75rem;
    }

    .segmented span:first-child {
        background: #fff;
        box-shadow: 0 2px 8px rgba(24, 33, 47, 0.08);
        color: var(--blue);
    }

    .decision {
        display: grid;
        gap: 0;
    }

    .risk-result {
        border-bottom: 1px solid var(--line);
        padding: 1.2rem 1.05rem;
    }

    .risk-result span {
        color: var(--coral);
        font-size: 0.86rem;
        font-weight: 900;
        text-transform: uppercase;
    }

    .risk-result.moderate span {
        color: var(--amber);
    }

    .risk-result.low span {
        color: var(--green);
    }

    .risk-result strong {
        color: var(--ink);
        display: block;
        font-size: 3rem;
        line-height: 1;
        margin: 0.7rem 0;
    }

    .risk-result p,
    .predictor-row p {
        color: var(--muted);
        margin: 0;
    }

    .action-list {
        color: #34445c;
        display: grid;
        gap: 0.75rem;
        line-height: 1.45;
        margin: 0;
        padding: 1.05rem 1.05rem 1.05rem 2.15rem;
    }

    .feature-list,
    .predictor-list,
    .model-grid {
        display: grid;
        gap: 0.9rem;
        padding: 1.05rem;
    }

    .feature-row {
        display: grid;
        gap: 0.5rem;
    }

    .feature-meta {
        align-items: center;
        display: flex;
        justify-content: space-between;
    }

    .feature-meta strong {
        color: var(--blue);
        font-size: 0.86rem;
    }

    .meter {
        background: #e8eef6;
        border-radius: 999px;
        height: 8px;
        overflow: hidden;
    }

    .meter span {
        background: linear-gradient(90deg, var(--blue), var(--teal));
        border-radius: inherit;
        display: block;
        height: 100%;
    }

    .predictor-row {
        align-items: center;
        border-bottom: 1px solid var(--line);
        display: grid;
        gap: 0.75rem;
        grid-template-columns: 30px minmax(0, 1fr);
        padding-bottom: 0.8rem;
    }

    .predictor-row:last-child {
        border-bottom: 0;
        padding-bottom: 0;
    }

    .rank {
        align-items: center;
        background: rgba(37, 99, 235, 0.10);
        border-radius: 6px;
        color: var(--blue);
        display: inline-flex;
        font-size: 0.82rem;
        font-weight: 900;
        height: 30px;
        justify-content: center;
        width: 30px;
    }

    .predictor-row strong {
        color: var(--ink);
        display: block;
        font-size: 0.9rem;
        margin-bottom: 0.2rem;
    }

    .model-grid {
        gap: 0;
    }

    .model-metric {
        align-items: center;
        border-bottom: 1px solid var(--line);
        display: flex;
        justify-content: space-between;
        min-height: 3.35rem;
    }

    .model-metric:last-child {
        border-bottom: 0;
    }

    .model-metric strong {
        color: var(--ink);
    }

    div[data-testid="stPlotlyChart"],
    div[data-testid="stDataFrame"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow);
        overflow: hidden;
    }

    div[data-testid="stExpander"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: none;
    }

    div[data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: var(--shadow);
        padding: 1rem;
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.9rem 1rem;
    }

    div[data-testid="stFormSubmitButton"] button {
        background: var(--blue);
        border: 0;
        color: #ffffff;
        font-weight: 900;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input {
        background: #ffffff;
        border-color: var(--line-strong);
        color: var(--ink);
    }

    @media (max-width: 760px) {
        .topbar,
        .page-heading {
            align-items: stretch;
            flex-direction: column;
        }

        .model-badge {
            text-align: left;
        }

        .page-heading h2 {
            font-size: 1.7rem;
            line-height: 1.14;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_data() -> pd.DataFrame:
    return core.load_customer_data()


def _kpi_card(item: dict[str, str], accent: str | None = None) -> str:
    accent_class = escape(accent or item.get("accent", ""))
    return f"""
    <div class="kpi-card {accent_class}">
        <span>{escape(item["label"])}</span>
        <strong>{escape(item["value"])}</strong>
        <small>{escape(item["detail"])}</small>
    </div>
    """


def _decision_panel(prediction: dict[str, object]) -> str:
    actions = "".join(
        f"<li>{escape(action)}</li>" for action in prediction["actions"]
    )
    return f"""
    <div class="html-panel decision">
        <div class="panel-heading">
            <div>
                <span class="eyebrow">Current profile</span>
                <h3>Prediction summary</h3>
            </div>
        </div>
        <div class="risk-result {escape(prediction["risk_class"])}">
            <span>{escape(prediction["risk_level"])} risk</span>
            <strong>{escape(prediction["probability_label"])}</strong>
            <p>{escape(prediction["prediction"])}</p>
        </div>
        <ul class="action-list">{actions}</ul>
    </div>
    """


def _feature_panel(summary: dict[str, object]) -> str:
    rows = []
    for driver in summary["top_drivers"]:
        rows.append(
            f"""
            <div class="feature-row">
                <div class="feature-meta">
                    <span>{escape(driver["DisplayName"])}</span>
                    <strong>{escape(driver["ImportanceLabel"])}</strong>
                </div>
                <div class="meter"><span style="width: {escape(driver["BarWidth"])}"></span></div>
            </div>
            """
        )
    return f"""
    <div class="html-panel">
        <div class="panel-heading">
            <div>
                <span class="eyebrow">Model signals</span>
                <h3>Feature importance</h3>
            </div>
        </div>
        <div class="feature-list">{''.join(rows)}</div>
    </div>
    """


def _predictor_panel(summary: dict[str, object]) -> str:
    rows = []
    for predictor in summary["top_predictors"]:
        rows.append(
            f"""
            <div class="predictor-row">
                <span class="rank">{predictor["rank"]}</span>
                <div>
                    <strong>{escape(predictor["title"])}</strong>
                    <p>{escape(predictor["body"])}</p>
                </div>
            </div>
            """
        )
    return f"""
    <div class="html-panel">
        <div class="panel-heading">
            <div>
                <span class="eyebrow">Segments</span>
                <h3>Top churn predictors</h3>
            </div>
        </div>
        <div class="predictor-list">{''.join(rows)}</div>
    </div>
    """


def _model_panel(summary: dict[str, object]) -> str:
    rows = []
    for metric in summary["model_metrics"]:
        rows.append(
            f"""
            <div class="model-metric">
                <span>{escape(metric["label"])}</span>
                <strong>{escape(metric["value"])}</strong>
            </div>
            """
        )
    return f"""
    <div class="html-panel">
        <div class="panel-heading">
            <div>
                <span class="eyebrow">Validation</span>
                <h3>Model health</h3>
            </div>
        </div>
        <div class="model-grid">{''.join(rows)}</div>
    </div>
    """


data = _load_data()
summary = core.get_dashboard_summary()
figures = core.build_plotly_figures()
default_profile = core.default_customer_profile()
default_prediction = core.predict_churn(default_profile)

with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">
            <span class="brand-mark">R</span>
            <div>
                <strong>Retention Hub</strong>
                <span>Telco churn analytics</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.radio(
        "Navigation",
        ["Overview", "Risk analysis", "Model signals", "Customer score"],
        label_visibility="collapsed",
    )
    st.caption(f"Model: {core.MODEL_NAME}")

st.markdown(
    f"""
    <div class="topbar">
        <h1>Retention Operations Dashboard</h1>
        <div class="search-shell">Search customer or segment</div>
        <div class="model-badge">
            <span>Validation</span>
            <strong>{escape(summary["model_metrics"][0]["value"])} accuracy</strong>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-heading">
        <div>
            <span class="eyebrow">Retention operations</span>
            <h2>Customer churn dashboard</h2>
            <p>Track churn risk and model signals.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_items = list(summary["mission_kpis"]) + [summary["kpis"][2]]
kpi_accents = [item.get("accent", "") for item in summary["mission_kpis"]] + ["amber"]
kpi_cols = st.columns(4)
for col, item, accent in zip(kpi_cols, kpi_items, kpi_accents):
    col.markdown(_kpi_card(item, accent), unsafe_allow_html=True)

st.markdown("")
left, right = st.columns([0.72, 0.28], gap="medium")
with left:
    st.markdown(
        """
        <div class="html-panel">
            <div class="panel-heading">
                <div>
                    <span class="eyebrow">Population risk</span>
                    <h3>Churn risk distribution</h3>
                </div>
                <div class="segmented"><span>Score</span><span>Probability</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        figures["risk_curve"],
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )
with right:
    st.markdown(_decision_panel(default_prediction), unsafe_allow_html=True)

st.markdown("")
signal_a, signal_b, signal_c = st.columns([0.36, 0.36, 0.28], gap="medium")
with signal_a:
    st.markdown(_feature_panel(summary), unsafe_allow_html=True)
with signal_b:
    st.markdown(_predictor_panel(summary), unsafe_allow_html=True)
with signal_c:
    st.markdown(_model_panel(summary), unsafe_allow_html=True)

chart_a, chart_b = st.columns(2, gap="medium")
with chart_a:
    st.plotly_chart(
        figures["by_contract"],
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )
with chart_b:
    st.plotly_chart(
        figures["by_tenure"],
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

chart_c, chart_d = st.columns(2, gap="medium")
with chart_c:
    st.plotly_chart(
        figures["distribution"],
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )
with chart_d:
    st.plotly_chart(
        figures["comparison"],
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

st.markdown("### Customer score")
profile = {}
with st.form("customer_score"):
    form_fields = core.field_definitions(default_profile)
    form_cols = st.columns(3)

    for index, field in enumerate(form_fields):
        with form_cols[index % len(form_cols)]:
            if field["type"] == "select":
                options = [option["value"] for option in field["options"]]
                selected_index = (
                    options.index(str(field["value"]))
                    if str(field["value"]) in options
                    else 0
                )
                profile[field["name"]] = st.selectbox(
                    field["label"],
                    options,
                    index=selected_index,
                )
            elif field["type"] == "checkbox":
                profile[field["name"]] = 1 if st.checkbox(
                    field["label"],
                    value=field["checked"],
                ) else 0
            else:
                profile[field["name"]] = st.number_input(
                    field["label"],
                    min_value=field["min"],
                    max_value=field["max"],
                    value=field["value"],
                    step=field["step"],
                )

    st.form_submit_button("Score customer")

prediction = core.predict_churn(profile)
score_cols = st.columns([0.22, 0.22, 0.56])
score_cols[0].metric("Churn probability", prediction["probability_label"])
score_cols[1].metric("Risk level", prediction["risk_level"])
score_cols[2].write("Retention actions")
for action in prediction["actions"]:
    score_cols[2].write(f"- {action}")

with st.expander("Customer records", expanded=False):
    st.dataframe(
        data[
            [
                "customerID",
                "Contract",
                "InternetService",
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "Churn",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
