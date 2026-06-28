from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from model_training import ARTIFACT_VERSION, train_model_artifact


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "artifacts" / "churn_model.joblib"

RANDOM_STATE = 42

FEATURE_FIELDS = [
    "tenure",
    "Contract",
    "TotalCharges",
    "InternetService",
    "MonthlyCharges",
    "PaymentMethod",
]

NUMERIC_FIELDS = ["tenure", "MonthlyCharges", "TotalCharges"]

FIELD_LABELS = {
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "tenure": "Tenure",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
}

NUMERIC_LIMITS = {
    "tenure": {"min": 0, "max": 72, "step": 1},
    "MonthlyCharges": {"min": 18.0, "max": 120.0, "step": 0.05},
    "TotalCharges": {"min": 0.0, "max": 9000.0, "step": 0.05},
}

MISSION_COLORS = {
    "page": "#f6f8fb",
    "panel": "#ffffff",
    "panel_alt": "#f2f5f9",
    "line": "#d9e2ec",
    "grid": "#ecf1f6",
    "text": "#18212f",
    "muted": "#64748b",
    "cyan": "#2563eb",
    "teal": "#0f9f8f",
    "coral": "#e85d75",
    "amber": "#d99a18",
    "green": "#18a058",
}


class CustomerProfileValidationError(ValueError):
    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__("; ".join(f"{field}: {message}" for field, message in errors.items()))


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _money(value: float) -> str:
    return f"${value:,.0f}"


def _format_number(value: float | int) -> str:
    return f"{value:,.0f}"


def _risk_thresholds(artifacts: dict[str, Any]) -> tuple[float, float]:
    moderate = float(np.clip(artifacts.get("decision_threshold", 0.5), 0.1, 0.8))
    high = float(min(0.95, max(0.65, moderate + 0.2)))
    return moderate, high


def _feature_display_name(name: str) -> str:
    return FIELD_LABELS.get(name, name)


def _segment_rate(data: pd.DataFrame, mask: pd.Series) -> str:
    segment = data.loc[mask, "Churn"]
    if segment.empty:
        return "No records"
    return f"{_percent((segment == 'Yes').mean())} churn rate"


def _darken_figure(
    figure: go.Figure,
    title: str,
    height: int,
    margin: dict[str, int] | None = None,
) -> go.Figure:
    layout_options: dict[str, Any] = dict(
        paper_bgcolor=MISSION_COLORS["panel"],
        plot_bgcolor=MISSION_COLORS["panel"],
        font=dict(
            color=MISSION_COLORS["muted"],
            family='Inter, "Segoe UI", ui-sans-serif, system-ui, sans-serif',
        ),
        height=height,
        margin=margin or dict(l=44, r=24, t=56, b=44),
        hoverlabel=dict(
            bgcolor=MISSION_COLORS["panel_alt"],
            bordercolor=MISSION_COLORS["line"],
            font=dict(color=MISSION_COLORS["text"]),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
        ),
    )
    if title:
        layout_options["title"] = dict(
            text=title,
            x=0.02,
            y=0.96,
            xanchor="left",
            font=dict(size=13, color=MISSION_COLORS["text"]),
        )
    figure.update_layout(**layout_options)
    figure.update_xaxes(
        gridcolor=MISSION_COLORS["grid"],
        linecolor=MISSION_COLORS["line"],
        zerolinecolor=MISSION_COLORS["line"],
        tickfont=dict(color=MISSION_COLORS["muted"], size=10),
        title_font=dict(color=MISSION_COLORS["muted"]),
    )
    figure.update_yaxes(
        gridcolor=MISSION_COLORS["grid"],
        linecolor=MISSION_COLORS["line"],
        zerolinecolor=MISSION_COLORS["line"],
        tickfont=dict(color=MISSION_COLORS["muted"], size=10),
        title_font=dict(color=MISSION_COLORS["muted"]),
    )
    return figure


@lru_cache(maxsize=1)
def _clean_data_cached() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = data["TotalCharges"].replace(" ", np.nan)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna().reset_index(drop=True)
    return data


def load_customer_data() -> pd.DataFrame:
    return _clean_data_cached().copy()


@lru_cache(maxsize=1)
def _data_hash() -> str:
    return hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def get_model_artifacts() -> dict[str, Any]:
    expected_hash = _data_hash()
    if MODEL_PATH.exists():
        try:
            artifact = joblib.load(MODEL_PATH)
            if (
                artifact.get("artifact_version") == ARTIFACT_VERSION
                and artifact.get("data_hash") == expected_hash
                and artifact.get("feature_fields") == FEATURE_FIELDS
            ):
                return artifact
        except (OSError, ValueError, TypeError, KeyError):
            pass

    # Keep local development usable before train_model.py has been run. The
    # generated artifact should be used in deployment to avoid startup work.
    return train_model_artifact(
        load_customer_data(),
        FEATURE_FIELDS,
        NUMERIC_FIELDS,
        random_state=RANDOM_STATE,
        data_hash=expected_hash,
    )


@lru_cache(maxsize=1)
def _customer_scores_cached() -> pd.DataFrame:
    data = _clean_data_cached().copy()
    artifacts = get_model_artifacts()
    oof_probabilities = artifacts.get("oof_probabilities")
    if oof_probabilities is not None and len(oof_probabilities) == len(data):
        data["RiskScore"] = np.asarray(oof_probabilities)
    else:
        data["RiskScore"] = artifacts["pipeline"].predict_proba(
            data[FEATURE_FIELDS]
        )[:, 1]
    moderate_threshold, high_threshold = _risk_thresholds(artifacts)
    data["RiskBand"] = pd.cut(
        data["RiskScore"],
        bins=[0, moderate_threshold, high_threshold, 1],
        labels=["Low", "Moderate", "High"],
        include_lowest=True,
    )
    return data


def score_customer_population() -> pd.DataFrame:
    return _customer_scores_cached().copy()


def get_model_comparison() -> pd.DataFrame:
    return get_model_artifacts()["model_comparison"].copy()


def get_model_name() -> str:
    return str(get_model_artifacts()["model_name"])


def churn_rate_table(group_col: str) -> pd.DataFrame:
    data = load_customer_data()
    grouped = (
        data.groupby(group_col)["Churn"]
        .agg(Customers="size", Churned=lambda values: (values == "Yes").sum())
        .reset_index()
    )
    grouped["ChurnRate"] = grouped["Churned"] / grouped["Customers"]
    return grouped.sort_values("ChurnRate", ascending=False)


def get_dashboard_summary() -> dict[str, Any]:
    data = load_customer_data()
    scored = score_customer_population()
    artifacts = get_model_artifacts()
    churn_rate = (data["Churn"] == "Yes").mean()
    churned = data[data["Churn"] == "Yes"]
    retained = data[data["Churn"] == "No"]
    _, high_risk_threshold = _risk_thresholds(artifacts)
    at_risk = scored[scored["RiskScore"] >= high_risk_threshold]
    accuracy = artifacts["metrics"]["Accuracy"]
    model_name = str(artifacts["model_name"])
    majority_baseline = data["Churn"].value_counts(normalize=True).max()

    kpis = [
        {
            "label": "Customers analyzed",
            "value": _format_number(len(data)),
            "detail": "Clean Telco records",
        },
        {
            "label": "Observed churn",
            "value": _percent(churn_rate),
            "detail": f"{_format_number(len(churned))} churned customers",
        },
        {
            "label": "Historical monthly revenue lost",
            "value": _money(churned["MonthlyCharges"].sum()),
            "detail": "Monthly charges among observed churners",
        },
        {
            "label": "Best model F1",
            "value": _percent(artifacts["metrics"]["F1-score"]),
            "detail": model_name,
        },
    ]

    mission_kpis = [
        {
            "label": "Overall churn rate",
            "value": _percent(churn_rate),
            "detail": "Observed in the cleaned dataset",
            "accent": "coral",
        },
        {
            "label": "Customers at risk",
            "value": _format_number(len(at_risk)),
            "detail": f"{_percent(len(at_risk) / len(scored))} scored high risk",
            "accent": "cyan",
        },
        {
            "label": "Model accuracy",
            "value": _percent(accuracy),
            "detail": f"{_percent(majority_baseline)} majority baseline",
            "accent": "green",
        },
    ]

    insights = [
        {
            "title": "Early tenure is the sharpest signal",
            "body": (
                "Churned customers have a median tenure of "
                f"{churned['tenure'].median():.0f} months versus "
                f"{retained['tenure'].median():.0f} months for retained customers."
            ),
        },
        {
            "title": "Contract type matters",
            "body": (
                "Month-to-month customers show the highest observed churn rate, "
                "which matches the model's top churn-driver ranking."
            ),
        },
        {
            "title": "High monthly bills need attention",
            "body": (
                "Churned customers average "
                f"{_money(churned['MonthlyCharges'].mean())} per month versus "
                f"{_money(retained['MonthlyCharges'].mean())} for retained customers."
            ),
        },
    ]

    model_metrics = [
        {
            "label": name,
            "value": f"{value:.3f}" if name == "Brier score" else _percent(value),
        }
        for name, value in artifacts["metrics"].items()
    ]

    top_drivers_table = artifacts["grouped_importance"].head(6).copy()
    top_importance = top_drivers_table["Importance"].max()
    top_drivers_table["DisplayName"] = top_drivers_table["OriginalFeature"].map(
        _feature_display_name
    )
    top_drivers_table["NormalizedImportance"] = (
        top_drivers_table["Importance"] / top_importance
    )
    top_drivers_table["ImportanceLabel"] = top_drivers_table[
        "NormalizedImportance"
    ].map(lambda v: f"{v:.2f}")
    top_drivers_table["BarWidth"] = top_drivers_table["NormalizedImportance"].map(
        lambda v: f"{max(10, v * 100):.0f}%"
    )
    top_drivers = top_drivers_table.to_dict("records")

    top_predictors = [
        {
            "rank": 1,
            "title": "Month-to-month Contract",
            "body": _segment_rate(data, data["Contract"] == "Month-to-month"),
        },
        {
            "rank": 2,
            "title": "Tenure <= 6 Months",
            "body": _segment_rate(data, data["tenure"] <= 6),
        },
        {
            "rank": 3,
            "title": "Fiber Optic Service",
            "body": _segment_rate(data, data["InternetService"] == "Fiber optic"),
        },
        {
            "rank": 4,
            "title": "Monthly Charges >= $80",
            "body": _segment_rate(data, data["MonthlyCharges"] >= 80),
        },
        {
            "rank": 5,
            "title": "Electronic Check Payment",
            "body": _segment_rate(data, data["PaymentMethod"] == "Electronic check"),
        },
    ]

    return {
        "kpis": kpis,
        "mission_kpis": mission_kpis,
        "insights": insights,
        "model_metrics": model_metrics,
        "top_drivers": top_drivers,
        "top_predictors": top_predictors,
        "confusion_matrix": artifacts["confusion_matrix"].tolist(),
        "model_name": model_name,
        "decision_threshold": artifacts["decision_threshold"],
    }


def build_plotly_figures() -> dict[str, go.Figure]:
    data = load_customer_data()
    scored = score_customer_population()
    artifacts = get_model_artifacts()
    model_name = str(artifacts["model_name"])

    ranked_scores = scored.sort_values("RiskScore").reset_index(drop=True)
    ranked_scores["Bucket"] = pd.qcut(
        ranked_scores.index + 1,
        q=min(80, len(ranked_scores)),
        labels=False,
    )
    risk_curve_data = (
        ranked_scores.groupby("Bucket", as_index=False)
        .agg(RiskScore=("RiskScore", "mean"), Customers=("customerID", "size"))
        .reset_index(drop=True)
    )
    risk_curve_data["Position"] = np.linspace(0, 100, len(risk_curve_data))
    marker_indexes = [
        (risk_curve_data["Position"] - target).abs().idxmin()
        for target in [72, 88]
    ]
    marker_points = risk_curve_data.loc[marker_indexes].drop_duplicates("Bucket")

    risk_curve = go.Figure()
    risk_curve.add_vrect(
        x0=72,
        x1=100,
        fillcolor=MISSION_COLORS["coral"],
        opacity=0.14,
        line_width=0,
    )
    risk_curve.add_trace(
        go.Scatter(
            x=risk_curve_data["Position"],
            y=risk_curve_data["RiskScore"],
            mode="lines",
            line=dict(
                color=MISSION_COLORS["cyan"],
                width=5,
                shape="spline",
                smoothing=1.25,
            ),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.08)",
            hovertemplate="Population rank %{x:.0f}<br>Risk probability %{y:.1%}<extra></extra>",
            name="Score",
        )
    )
    risk_curve.add_trace(
        go.Scatter(
            x=marker_points["Position"],
            y=marker_points["RiskScore"],
            mode="markers",
            marker=dict(
                color=MISSION_COLORS["coral"],
                line=dict(color="#ffffff", width=2),
                size=14,
            ),
            hovertemplate="Risk probability %{y:.1%}<extra></extra>",
            name="High-risk markers",
            showlegend=False,
        )
    )
    risk_curve.add_annotation(
        x=85,
        y=max(0.52, float(marker_points["RiskScore"].max()) + 0.06),
        text="High risk zone",
        showarrow=False,
        font=dict(color=MISSION_COLORS["coral"], size=11),
        bgcolor="rgba(232, 93, 117, 0.10)",
        bordercolor="rgba(232, 93, 117, 0.25)",
        borderwidth=1,
    )
    risk_curve.update_xaxes(
        range=[0, 100],
        tickmode="array",
        tickvals=[0, 50, 100],
        ticktext=["Low risk", "Avg risk", "High risk"],
        title="",
    )
    risk_curve.update_yaxes(range=[0, 1], tickformat=".0%", title="")
    _darken_figure(
        risk_curve,
        "",
        470,
        margin=dict(l=48, r=24, t=32, b=48),
    )

    churn_counts = (
        data["Churn"].value_counts().reindex(["No", "Yes"]).reset_index()
    )
    churn_counts.columns = ["Churn", "Customers"]
    distribution = px.pie(
        churn_counts,
        values="Customers",
        names="Churn",
        hole=0.58,
        color="Churn",
        color_discrete_map={"No": "#2a9d8f", "Yes": "#e76f51"},
    )
    distribution.update_traces(textinfo="percent+label", marker_line_width=0)
    _darken_figure(
        distribution,
        "Churn distribution",
        height=330,
        margin=dict(l=20, r=20, t=54, b=20),
    )

    contract = churn_rate_table("Contract")
    contract["RateLabel"] = contract["ChurnRate"].map(lambda value: f"{value:.1%}")
    by_contract = px.bar(
        contract,
        x="Contract",
        y="ChurnRate",
        text="RateLabel",
        color="Contract",
        color_discrete_sequence=["#e76f51", "#e9c46a", "#2a9d8f"],
    )
    by_contract.update_yaxes(tickformat=".0%", title="Churn rate")
    by_contract.update_xaxes(title="")
    by_contract.update_traces(textposition="outside", cliponaxis=False)
    _darken_figure(
        by_contract,
        "Churn rate by contract",
        height=330,
        margin=dict(l=36, r=20, t=54, b=46),
    )
    by_contract.update_layout(showlegend=False)

    tenure_data = data.copy()
    tenure_data["TenureBand"] = pd.cut(
        tenure_data["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
    )
    tenure_rates = (
        tenure_data.groupby("TenureBand", observed=True)["Churn"]
        .agg(Customers="size", Churned=lambda values: (values == "Yes").sum())
        .reset_index()
    )
    tenure_rates["ChurnRate"] = tenure_rates["Churned"] / tenure_rates["Customers"]
    tenure_rates["RateLabel"] = tenure_rates["ChurnRate"].map(lambda value: f"{value:.1%}")
    by_tenure = px.line(
        tenure_rates,
        x="TenureBand",
        y="ChurnRate",
        markers=True,
        text="RateLabel",
    )
    by_tenure.update_traces(
        line=dict(color=MISSION_COLORS["cyan"], width=3),
        marker=dict(size=9, color=MISSION_COLORS["coral"]),
    )
    by_tenure.update_yaxes(tickformat=".0%", title="Churn rate")
    by_tenure.update_xaxes(title="Tenure band")
    _darken_figure(
        by_tenure,
        "Churn falls as tenure grows",
        height=330,
        margin=dict(l=36, r=20, t=54, b=46),
    )

    top_features = artifacts["grouped_importance"].head(10).copy()
    top_features["DisplayName"] = top_features["OriginalFeature"].map(
        _feature_display_name
    )
    top_features = top_features.sort_values("Importance")
    drivers = px.bar(
        top_features,
        x="Importance",
        y="DisplayName",
        orientation="h",
        color="Importance",
        color_continuous_scale=[
            MISSION_COLORS["cyan"],
            MISSION_COLORS["teal"],
            MISSION_COLORS["coral"],
        ],
    )
    drivers.update_xaxes(title="Feature importance")
    drivers.update_yaxes(title="")
    _darken_figure(
        drivers,
        f"Top churn drivers - {model_name}",
        height=390,
        margin=dict(l=40, r=20, t=54, b=42),
    )
    drivers.update_layout(coloraxis_showscale=False)

    comparison = get_model_comparison()
    comparison_fig = px.bar(
        comparison.sort_values("F1-score", ascending=True),
        x="F1-score",
        y="Model",
        orientation="h",
        color="Recall",
        color_continuous_scale=[MISSION_COLORS["amber"], MISSION_COLORS["teal"]],
        text=comparison.sort_values("F1-score", ascending=True)["F1-score"].map(
            lambda value: f"{value:.1%}"
        ),
    )
    comparison_fig.update_xaxes(tickformat=".0%", title="F1-score")
    comparison_fig.update_yaxes(title="")
    comparison_fig.update_traces(textposition="outside", cliponaxis=False)
    _darken_figure(
        comparison_fig,
        "Training CV model comparison",
        height=330,
        margin=dict(l=40, r=28, t=54, b=42),
    )
    comparison_fig.update_layout(coloraxis_showscale=False)

    corr_source = scored.assign(
        ChurnBinary=(scored["Churn"] == "Yes").astype(int)
    )[
        [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "SeniorCitizen",
            "RiskScore",
            "ChurnBinary",
        ]
    ]
    corr = corr_source.corr()
    corr_labels = [
        "Tenure",
        "Monthly",
        "Total",
        "Senior",
        "Risk",
        "Churn",
    ]
    correlation = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr_labels,
            y=corr_labels,
            zmin=-1,
            zmax=1,
            colorscale=[
                [0, MISSION_COLORS["coral"]],
                [0.5, "#ffffff"],
                [1, MISSION_COLORS["teal"]],
            ],
            colorbar=dict(
                thickness=10,
                len=0.75,
                tickfont=dict(color=MISSION_COLORS["muted"], size=10),
            ),
            hovertemplate="%{y} vs %{x}<br>Correlation %{z:.2f}<extra></extra>",
        )
    )
    correlation.update_yaxes(autorange="reversed")
    _darken_figure(
        correlation,
        "",
        320,
        margin=dict(l=62, r=20, t=24, b=54),
    )

    return {
        "risk_curve": risk_curve,
        "distribution": distribution,
        "by_contract": by_contract,
        "by_tenure": by_tenure,
        "drivers": drivers,
        "comparison": comparison_fig,
        "correlation": correlation,
    }


def default_customer_profile() -> dict[str, Any]:
    data = load_customer_data()
    candidates = data[data["Churn"] == "Yes"].sort_values(
        ["tenure", "MonthlyCharges"],
        ascending=[True, False],
    )
    row = candidates.iloc[0] if not candidates.empty else data.iloc[0]
    return coerce_customer_profile({field: row[field] for field in FEATURE_FIELDS})


def coerce_customer_profile(profile: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "tenure": 1,
        "InternetService": "Fiber optic",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0,
        "TotalCharges": 80.0,
    }
    merged = {**defaults, **profile}
    coerced: dict[str, Any] = {}
    errors: dict[str, str] = {}
    data = _clean_data_cached()

    for field in FEATURE_FIELDS:
        value = merged[field]
        if field in NUMERIC_FIELDS:
            try:
                number = float(value)
            except (TypeError, ValueError):
                errors[field] = "must be a number"
                continue
            if not np.isfinite(number):
                errors[field] = "must be finite"
                continue
            limits = NUMERIC_LIMITS[field]
            if not limits["min"] <= number <= limits["max"]:
                errors[field] = f"must be between {limits['min']} and {limits['max']}"
                continue
            if field == "tenure":
                if not number.is_integer():
                    errors[field] = "must be a whole number of months"
                    continue
                coerced[field] = int(number)
            else:
                coerced[field] = number
            continue

        text_value = str(value).strip()
        allowed = set(data[field].dropna().astype(str).unique())
        if text_value not in allowed:
            errors[field] = "is not a recognized option"
            continue
        coerced[field] = text_value

    if errors:
        raise CustomerProfileValidationError(errors)

    return coerced


def field_definitions(profile: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    data = load_customer_data()
    profile = coerce_customer_profile(profile or default_customer_profile())
    fields = []

    for field in FEATURE_FIELDS:
        if field in NUMERIC_LIMITS:
            limits = NUMERIC_LIMITS[field]
            fields.append(
                {
                    "name": field,
                    "label": FIELD_LABELS[field],
                    "type": "number",
                    "value": profile[field],
                    **limits,
                }
            )
            continue

        options = sorted(data[field].dropna().astype(str).unique().tolist())
        fields.append(
            {
                "name": field,
                "label": FIELD_LABELS[field],
                "type": "select",
                "value": profile[field],
                "options": [
                    {
                        "value": option,
                        "selected": option == str(profile[field]),
                    }
                    for option in options
                ],
            }
        )

    return fields


def predict_churn(profile: dict[str, Any]) -> dict[str, Any]:
    profile = coerce_customer_profile(profile)
    artifacts = get_model_artifacts()
    row = pd.DataFrame([profile], columns=FEATURE_FIELDS)
    probability = artifacts["pipeline"].predict_proba(row)[0, 1]
    decision_threshold = float(artifacts.get("decision_threshold", 0.5))
    predicted = probability >= decision_threshold
    moderate_threshold, high_threshold = _risk_thresholds(artifacts)
    risk_level = (
        "High"
        if probability >= high_threshold
        else "Moderate"
        if probability >= moderate_threshold
        else "Low"
    )

    return {
        "profile": profile,
        "probability": probability,
        "probability_label": _percent(probability),
        "prediction": "Churn likely" if predicted else "Likely retained",
        "risk_level": risk_level,
        "risk_class": risk_level.lower(),
        "decision_threshold": decision_threshold,
        "actions": retention_actions(profile, probability),
    }


def retention_actions(profile: dict[str, Any], probability: float) -> list[str]:
    actions = []

    if profile["Contract"] == "Month-to-month":
        actions.append("Offer an annual-contract incentive or loyalty credit.")
    if profile["tenure"] <= 12:
        actions.append("Prioritize an early-life retention check-in.")
    if profile["PaymentMethod"] == "Electronic check":
        actions.append("Move payment to automatic billing with a small discount.")
    if profile["InternetService"] == "Fiber optic":
        actions.append("Review fiber pricing, service quality, and support tickets.")
    if profile["MonthlyCharges"] >= 80:
        actions.append("Audit plan fit and remove unused premium services.")

    if not actions:
        actions.append("Monitor satisfaction and keep renewal outreach light.")
    if probability >= 0.65:
        actions.insert(0, "Queue this customer for retention outreach this week.")

    return actions[:4]


def profile_from_mapping(values: dict[str, Any]) -> dict[str, Any]:
    profile: dict[str, Any] = {}
    defaults = default_customer_profile()

    for field in FEATURE_FIELDS:
        profile[field] = values.get(field, defaults[field])

    return coerce_customer_profile(profile)


def sample_records(limit: int = 25) -> list[dict[str, Any]]:
    columns = [
        "customerID",
        "Contract",
        "InternetService",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]
    return load_customer_data()[columns].head(limit).to_dict("records")
