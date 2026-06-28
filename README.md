# Telco Churn Dashboard

A reproducible customer-churn project with one shared analytics/model layer and two user interfaces:

- Django dashboard: `python manage.py runserver`
- Streamlit dashboard: `streamlit run streamlit_app.py`

The project cleans the local Telco dataset, selects a classifier using training-only cross-validation, calibrates its scores, evaluates it once on a held-out test set, and saves the fitted preprocessing/model artifact used by both dashboards.

## Quick start

Python 3.13 is used in CI.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py
```

Run either interface:

```powershell
python manage.py runserver
streamlit run streamlit_app.py
```

Open `http://127.0.0.1:8000` for Django or the URL printed by Streamlit, usually `http://localhost:8501`.

## Test and verify

```powershell
python manage.py check
python manage.py test
python -m compileall -q .
```

The tests cover cleaning, categorical scoring, validation, out-of-fold population scores, generated model comparisons, and Django GET/POST behavior. GitHub Actions runs training and the same checks on every push and pull request.

## Modeling workflow

`train_model.py` is the source of truth for the deployable model:

1. Convert blank `TotalCharges` values to missing values and remove the 11 incomplete rows.
2. Hold out a stratified 20% test set.
3. Compare logistic regression, random forest, and gradient boosting with five-fold cross-validation on the training set, ranked by mean F1.
4. Calibrate the selected model with sigmoid calibration.
5. Choose the classification threshold from training-only out-of-fold predictions.
6. Evaluate once on the untouched holdout using accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, and a confusion matrix.
7. Refit on all clean records and save `artifacts/churn_model.joblib` plus readable metadata.

Preprocessing is part of the fitted pipeline. This makes categorical handling identical during training and single-customer inference. Dashboard population-risk charts use out-of-fold scores rather than in-sample predictions.

## Project structure

```text
dashboard_core.py       Shared data, scoring, chart, and form logic
model_training.py       Preprocessing, selection, calibration, and evaluation
train_model.py          Reproducible artifact CLI
streamlit_app.py        Streamlit interface
dashboard/              Django application, templates, static files, and tests
churn_dashboard/        Django project configuration
churn.ipynb             Exploratory analysis notebook
MODEL_CARD.md            Intended use, metrics, and limitations
```

## Production configuration

Development defaults are local-only. Set these variables for a deployed Django instance:

```powershell
$env:DJANGO_DEBUG="false"
$env:DJANGO_SECRET_KEY="replace-with-a-long-random-secret"
$env:DJANGO_ALLOWED_HOSTS="churn.example.com"
python manage.py collectstatic --noinput
gunicorn churn_dashboard.wsgi:application
```

Train the model during the build/release step. The binary joblib artifact is intentionally ignored by Git because it is generated and tied to dependency versions.

## Data and limitations

The repository includes `Telco-Customer-Churn.csv` with 7,043 rows. Its authoritative source URL and redistribution license are not recorded in the original project, so verify and document them before public or commercial redistribution.

This is a cross-sectional demonstration dataset with no explicit prediction horizon. Scores show patterns in this dataset; they are not causal claims or evidence that a retention action will work. Review drift, calibration, subgroup performance, privacy, and intervention outcomes before operational use. See [MODEL_CARD.md](MODEL_CARD.md).
