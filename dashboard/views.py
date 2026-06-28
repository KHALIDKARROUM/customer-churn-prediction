from __future__ import annotations

from django.shortcuts import render

import dashboard_core as core


def index(request):
    form_error = None
    response_status = 200
    if request.method == "POST":
        try:
            profile = core.profile_from_mapping(request.POST)
        except core.CustomerProfileValidationError as exc:
            profile = core.default_customer_profile()
            form_error = str(exc)
            response_status = 400
    else:
        profile = core.default_customer_profile()

    prediction = core.predict_churn(profile)
    figures = core.build_plotly_figures()
    charts = {}
    for index, (name, figure) in enumerate(figures.items()):
        charts[name] = figure.to_html(
            full_html=False,
            include_plotlyjs=True if index == 0 else False,
            config={"displayModeBar": False, "responsive": True},
        )

    context = {
        "summary": core.get_dashboard_summary(),
        "charts": charts,
        "form_fields": core.field_definitions(profile),
        "form_error": form_error,
        "prediction": prediction,
        "model_name": core.get_model_name(),
        "model_comparison": core.get_model_comparison().to_dict("records"),
        "sample_records": core.sample_records(),
    }
    return render(request, "dashboard/index.html", context, status=response_status)
