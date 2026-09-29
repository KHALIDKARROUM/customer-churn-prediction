from __future__ import annotations

from html import escape

from django.test import SimpleTestCase
from streamlit.testing.v1 import AppTest

import dashboard_core as core


class StreamlitPredictionTests(SimpleTestCase):
    def assert_matching_results(self, app: AppTest, profile: dict) -> None:
        self.assertEqual(len(app.exception), 0)
        expected = core.predict_churn(profile)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["Churn probability"], expected["probability_label"])
        self.assertEqual(metrics["Risk level"], expected["risk_level"])
        panels = [item.value for item in app.markdown if "Prediction summary" in item.value]
        self.assertEqual(len(panels), 1)
        self.assertIn(f'<strong>{expected["probability_label"]}</strong>', panels[0])
        self.assertIn(f'{expected["risk_level"]} risk', panels[0])
        for action in expected["actions"]:
            self.assertIn(escape(action), panels[0])

    def test_results_follow_submitted_customer_and_survive_other_interactions(self):
        app = AppTest.from_file(str(core.BASE_DIR / "streamlit_app.py"), default_timeout=120).run()
        self.assert_matching_results(app, core.default_customer_profile())

        profile = {
            "tenure": 72,
            "Contract": "Two year",
            "TotalCharges": 1440.0,
            "InternetService": "No",
            "MonthlyCharges": 20.0,
            "PaymentMethod": "Mailed check",
        }
        values_by_label = {core.FIELD_LABELS[field]: value for field, value in profile.items()}
        for widget in app.number_input:
            widget.set_value(values_by_label[widget.label])
        for widget in app.selectbox:
            widget.select(values_by_label[widget.label])
        next(button for button in app.button if button.label == "Score customer").click().run()
        self.assert_matching_results(app, profile)

        app.text_input[0].set_value("Two year").run()
        self.assert_matching_results(app, profile)

        profile["Contract"] = "Month-to-month"
        next(widget for widget in app.selectbox if widget.label == "Contract").select("Month-to-month")
        next(button for button in app.button if button.label == "Score customer").click().run()
        self.assert_matching_results(app, profile)
