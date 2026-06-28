from __future__ import annotations

from django.test import Client, SimpleTestCase

import dashboard_core as core


class ChurnCoreTests(SimpleTestCase):
    def test_data_cleaning_removes_blank_total_charges(self):
        data = core.load_customer_data()
        self.assertEqual(len(data), 7032)
        self.assertFalse(data["TotalCharges"].isna().any())

    def test_categorical_inputs_change_predictions(self):
        base = core.default_customer_profile()
        month_to_month = {**base, "Contract": "Month-to-month"}
        two_year = {**base, "Contract": "Two year"}

        first = core.predict_churn(month_to_month)["probability"]
        second = core.predict_churn(two_year)["probability"]

        self.assertNotAlmostEqual(first, second, places=6)

    def test_profile_validation_rejects_invalid_values(self):
        profile = core.default_customer_profile()
        profile["tenure"] = 100
        profile["Contract"] = "Forever"

        with self.assertRaises(core.CustomerProfileValidationError) as context:
            core.coerce_customer_profile(profile)

        self.assertIn("tenure", context.exception.errors)
        self.assertIn("Contract", context.exception.errors)

    def test_population_scores_are_out_of_fold(self):
        artifact = core.get_model_artifacts()
        scores = core.score_customer_population()

        self.assertEqual(len(artifact["oof_probabilities"]), len(scores))
        self.assertTrue(scores["RiskScore"].between(0, 1).all())

    def test_model_comparison_is_generated(self):
        comparison = core.get_model_comparison()
        self.assertGreaterEqual(len(comparison), 3)
        self.assertIn(core.get_model_name(), comparison["Model"].tolist())


class DashboardViewTests(SimpleTestCase):
    def setUp(self):
        self.client = Client(HTTP_HOST="localhost")

    def test_dashboard_get(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Customer churn dashboard")

    def test_dashboard_valid_post(self):
        response = self.client.post(
            "/",
            {
                "tenure": "12",
                "Contract": "Two year",
                "TotalCharges": "1200",
                "InternetService": "DSL",
                "MonthlyCharges": "50",
                "PaymentMethod": "Mailed check",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Prediction summary")

    def test_dashboard_invalid_post(self):
        response = self.client.post(
            "/",
            {
                "tenure": "not-a-number",
                "Contract": "Two year",
                "TotalCharges": "1200",
                "InternetService": "DSL",
                "MonthlyCharges": "50",
                "PaymentMethod": "Mailed check",
            },
        )
        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "must be a number", status_code=400)
