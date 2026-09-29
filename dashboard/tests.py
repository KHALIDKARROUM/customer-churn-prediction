from __future__ import annotations

import hashlib
import json
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
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

    def test_risk_chart_and_count_use_non_churners_and_configured_threshold(self):
        artifact = {**core.get_model_artifacts(), "decision_threshold": 0.6}
        scored = core.load_customer_data().head(5)
        scored["Churn"] = ["No", "No", "Yes", "No", "No"]
        scored["RiskScore"] = [0.1, 0.79, 0.99, 0.8, 0.9]

        with (
            patch.object(core, "get_model_artifacts", return_value=artifact),
            patch.object(core, "score_customer_population", return_value=scored),
        ):
            summary = core.get_dashboard_summary()
            figure = core.build_plotly_figures()["risk_curve"]

        kpi = next(item for item in summary["mission_kpis"] if "at risk" in item["label"])
        self.assertEqual(kpi["value"], "2")
        self.assertEqual(kpi["detail"], "50.0% of recorded non-churners")
        self.assertIn("4 customers recorded as not churned", summary["population_risk_note"])
        self.assertIn("80.0%", summary["population_risk_note"])
        self.assertEqual(list(figure.data[0].y), [0.1, 0.79, 0.8, 0.9, 0.9])
        zone = next(shape for shape in figure.layout.shapes if shape.type == "rect")
        self.assertEqual((zone.x0, zone.x1), (50, 100))
        threshold_line = next(shape for shape in figure.layout.shapes if shape.type == "line")
        self.assertEqual(threshold_line.y0, 0.8)

    def test_risk_bands_include_exact_thresholds_and_probability_endpoints(self):
        data = core.load_customer_data().head(4)
        artifact = {"decision_threshold": 0.4, "oof_probabilities": [0, 0.4, 0.65, 1]}
        with (
            patch.object(core, "_clean_data_cached", return_value=data),
            patch.object(core, "get_model_artifacts", return_value=artifact),
        ):
            scores = core._customer_scores_cached.__wrapped__()
        self.assertEqual(scores["RiskBand"].tolist(), ["Low", "Moderate", "High", "High"])

    def test_risk_curve_handles_no_high_scores_all_high_scores_and_empty_population(self):
        for scores, expected_start in [([], None), ([0.1, 0.64], None), ([0.65, 1], 0)]:
            with self.subTest(scores=scores):
                figure = core._build_risk_curve(pd.Series(scores, dtype=float), 0.65)
                zones = [shape for shape in figure.layout.shapes if shape.type == "rect"]
                if expected_start is None:
                    self.assertEqual(zones, [])
                else:
                    self.assertEqual((zones[0].x0, zones[0].x1), (expected_start, 100))
                if not scores:
                    self.assertTrue(any(
                        "No non-churned customers" in annotation.text
                        for annotation in figure.layout.annotations
                    ))

    def test_no_non_churners_produces_zero_risk_count(self):
        scored = core.load_customer_data().head(2)
        scored["Churn"] = "Yes"
        scored["RiskScore"] = 0.99
        with patch.object(core, "score_customer_population", return_value=scored):
            summary = core.get_dashboard_summary()
        kpi = next(item for item in summary["mission_kpis"] if "at risk" in item["label"])
        self.assertEqual(kpi["value"], "0")
        self.assertEqual(kpi["detail"], "0.0% of recorded non-churners")


class ModelArtifactTests(SimpleTestCase):
    def tearDown(self):
        core.get_model_artifacts.cache_clear()
        core._customer_scores_cached.cache_clear()

    def test_missing_artifact_has_an_actionable_error(self):
        with TemporaryDirectory() as directory:
            missing_model = core.Path(directory) / "churn_model.joblib"
            missing_metadata = core.Path(directory) / "model_metadata.json"
            with (
                patch.object(core, "MODEL_PATH", missing_model),
                patch.object(core, "METADATA_PATH", missing_metadata),
            ):
                core.get_model_artifacts.cache_clear()
                with self.assertRaisesRegex(core.ModelArtifactUnavailableError, "python train_model.py"):
                    core.get_model_artifacts()

    def test_artifact_hash_and_runtime_mismatch_are_rejected_before_scoring(self):
        artifact = core.get_model_artifacts()
        with TemporaryDirectory() as directory:
            model_path = core.Path(directory) / "churn_model.joblib"
            metadata_path = core.Path(directory) / "model_metadata.json"

            core.joblib.dump(artifact, model_path)
            metadata = {
                "artifact_version": artifact["artifact_version"],
                "artifact_sha256": "not-the-file-hash",
                "runtime_environment": artifact["runtime_environment"],
                "training_code_hash": artifact["training_code_hash"],
                "training_run_id": artifact["training_run_id"],
                "data_hash": artifact["data_hash"],
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with (
                patch.object(core, "MODEL_PATH", model_path),
                patch.object(core, "METADATA_PATH", metadata_path),
            ):
                core.get_model_artifacts.cache_clear()
                with self.assertRaisesRegex(core.ModelArtifactUnavailableError, "does not match"):
                    core.get_model_artifacts()

            incompatible = {**artifact, "runtime_environment": {**artifact["runtime_environment"], "numpy": "0.0"}}
            core.joblib.dump(incompatible, model_path)
            metadata.update(
                artifact_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                runtime_environment=incompatible["runtime_environment"],
            )
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with (
                patch.object(core, "MODEL_PATH", model_path),
                patch.object(core, "METADATA_PATH", metadata_path),
            ):
                core.get_model_artifacts.cache_clear()
                with self.assertRaisesRegex(core.ModelArtifactUnavailableError, "runtime package versions"):
                    core.get_model_artifacts()


class DashboardViewTests(SimpleTestCase):
    def setUp(self):
        self.client = Client(HTTP_HOST="localhost")

    def test_dashboard_get(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Customer churn dashboard")
        self.assertContains(response, "plotly/plotly.min.js")
        self.assertLess(len(response.content), 250_000)

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
        self.assertContains(response, "Suggested follow-up")

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
        self.assertContains(response, 'value="not-a-number"', status_code=400)
        self.assertContains(response, "Two year", status_code=400)
        self.assertContains(response, "No score was created", status_code=400)
        self.assertNotContains(response, "Suggested follow-up", status_code=400)

    def test_dashboard_missing_values_are_required_and_not_scored(self):
        response = self.client.post("/", {"tenure": "12"})
        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "is required", status_code=400)
        self.assertContains(response, 'value="12"', status_code=400)
        self.assertNotContains(response, "Suggested follow-up", status_code=400)

    def test_score_endpoint_replaces_only_the_score_panel(self):
        response = self.client.post(
            "/score/",
            {
                "tenure": "12",
                "Contract": "Two year",
                "TotalCharges": "1200",
                "InternetService": "DSL",
                "MonthlyCharges": "50",
                "PaymentMethod": "Mailed check",
            },
            HTTP_X_REQUESTED_WITH="XMLHttpRequest",
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'id="score"')
        self.assertContains(response, "Suggested follow-up")
        self.assertNotContains(response, "Risk among non-churned customers")
        self.assertNotContains(response, "plotly-graph-div")

    def test_chart_html_is_cached_across_full_page_requests(self):
        with patch.object(core, "get_dashboard_figures", wraps=core.get_dashboard_figures) as figures:
            from dashboard import views

            views._chart_html.cache_clear()
            self.client.get("/")
            self.client.get("/")
        self.assertEqual(figures.call_count, 1)
