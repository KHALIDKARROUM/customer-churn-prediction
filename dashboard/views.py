from __future__ import annotations

from django.shortcuts import render

import dashboard_core as core


def index(request):
    if request.method == "POST":
        profile = core.profile_from_mapping(request.POST)
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
        "prediction": prediction,
        "model_name": core.MODEL_NAME,
        "model_comparison": core.get_model_comparison().to_dict("records"),
        "sample_records": core.sample_records(),
    }
    return render(request, "dashboard/index.html", context)
