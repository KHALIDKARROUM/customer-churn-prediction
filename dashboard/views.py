from __future__ import annotations

from functools import lru_cache

from django.http import HttpRequest, HttpResponseNotAllowed
from django.shortcuts import render

import dashboard_core as core


@lru_cache(maxsize=1)
def _chart_html(data_hash: str, training_run_id: str) -> dict[str, str]:
    """Cache rendered chart fragments for the deployed data and model."""
    figures = core.get_dashboard_figures()
    return {
        name: figure.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"displayModeBar": False, "responsive": True},
        )
        for name, figure in figures.items()
    }


def _page_context(
    profile: dict | None = None,
    form_errors: dict[str, str] | None = None,
    prediction: dict | None = None,
) -> dict:
    artifacts = core.get_model_artifacts()
    if profile is None:
        profile = core.default_customer_profile()
    return {
        "summary": core.get_dashboard_summary(),
        "charts": _chart_html(artifacts["data_hash"], artifacts["training_run_id"]),
        "form_fields": core.field_definitions(profile, form_errors),
        "form_errors": form_errors or {},
        "prediction": prediction,
        "model_name": core.get_model_name(),
        "sample_records": core.sample_records(),
    }


def _score_submission(request: HttpRequest) -> tuple[dict, dict[str, str], dict | None, int]:
    submitted_profile = core.raw_profile_from_mapping(request.POST)
    try:
        profile = core.profile_from_mapping(request.POST)
    except core.CustomerProfileValidationError as exc:
        return submitted_profile, exc.errors, None, 400
    return profile, {}, core.predict_churn(profile), 200


def _model_unavailable(request: HttpRequest, exc: core.ModelArtifactUnavailableError):
    return render(
        request,
        "dashboard/model_unavailable.html",
        {"model_error": str(exc)},
        status=503,
    )


def index(request: HttpRequest):
    try:
        if request.method == "POST":
            profile, form_errors, prediction, status = _score_submission(request)
        else:
            profile = core.default_customer_profile()
            form_errors = {}
            prediction = None
            status = 200
        return render(
            request,
            "dashboard/index.html",
            _page_context(profile, form_errors, prediction),
            status=status,
        )
    except core.ModelArtifactUnavailableError as exc:
        return _model_unavailable(request, exc)


def score_customer(request: HttpRequest):
    """Return only the score panel for JavaScript submissions."""
    if request.method != "POST":
        return HttpResponseNotAllowed(["POST"])
    try:
        profile, form_errors, prediction, status = _score_submission(request)
        context = {
            "form_fields": core.field_definitions(profile, form_errors),
            "form_errors": form_errors,
            "prediction": prediction,
        }
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return render(request, "dashboard/_score_panel.html", context, status=status)
        return render(request, "dashboard/index.html", _page_context(profile, form_errors, prediction), status=status)
    except core.ModelArtifactUnavailableError as exc:
        return _model_unavailable(request, exc)
