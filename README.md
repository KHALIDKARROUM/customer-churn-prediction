# Telco Churn Dashboard

A reproducible customer-churn project with one shared analytics/model layer and two user interfaces:

- Django dashboard: `python manage.py runserver`
- Streamlit dashboard: `streamlit run streamlit_app.py`

The project cleans the local Telco dataset, selects a classifier using training-only cross-validation, calibrates its scores, evaluates it once on a held-out test set, and saves the fitted preprocessing/model artifact used by both dashboards.

## Quick start

Python 3.13 is used in CI.

`requirements.txt` is a hash-checked lock file. `requirements.in` lists the direct dependencies. Regenerate the lock in a Python 3.13 environment after intentionally changing a direct dependency:

```powershell
uv pip compile requirements.in --universal --python-version 3.13 --generate-hashes -o requirements.txt
```

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --require-hashes -r requirements.txt
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

The tests cover cleaning, categorical scoring, validation, population-score checks, generated model comparisons, and Django GET/POST behavior. Regression tests also check matching Streamlit predictions after submissions and filtering, field-level score validation, model-artifact integrity and runtime compatibility, risk thresholds and eligible populations, and production CSS delivery with HTTPS redirects. GitHub Actions installs the hash-checked lock, runs a dependency check, trains the model, runs the tests, checks production configuration, and collects static files on every push and pull request.

## Modeling workflow

`train_model.py` is the source of truth for the deployable model:

1. Convert blank `TotalCharges` values to missing values and remove the 11 incomplete rows.
2. Hold out a stratified 20% test set.
3. Compare logistic regression, random forest, and gradient boosting with five-fold cross-validation on the training set, ranked by mean F1.
4. Calibrate the selected model with sigmoid calibration.
5. Choose the classification threshold from training-only out-of-fold predictions.
6. Evaluate once on the untouched holdout using accuracy, precision, recall, F1, ROC-AUC, PR-AUC, Brier score, and a confusion matrix.
7. Refit on all clean records and save `artifacts/churn_model.joblib` plus readable metadata.

Preprocessing is part of the fitted pipeline. This makes categorical handling identical during training and single-customer inference. The artifact and metadata record a dataset hash, training-code hash, git revision, package versions, a unique training-run ID, and an artifact checksum. The dashboard refuses to score if any integrity, input, or runtime compatibility check fails. Dashboard population-risk charts use out-of-fold scores rather than in-sample predictions.

The risk curve and "Non-churned customers at risk" count include only records whose observed `Churn` value is `No`. The shaded share includes scores at or above the displayed high-risk threshold, using the same threshold as individual predictions. Other charts still describe the full historical sample. These records are a demonstration cohort, not a verified current outreach list; the dataset has no future prediction window.

## Project structure

```text
dashboard_core.py       Shared data, scoring, chart, and form logic
model_training.py       Preprocessing, selection, calibration, and evaluation
train_model.py          Reproducible artifact CLI
requirements.in         Direct Python dependency constraints
requirements.txt        Hash-checked dependency lock
streamlit_app.py        Streamlit interface
dashboard/              Django application, templates, static files, and tests
churn_dashboard/        Django project configuration
churn.ipynb             Exploratory analysis notebook
MODEL_CARD.md            Intended use, metrics, and limitations
```

## Production configuration

Development defaults are local-only. The example below targets **Linux or WSL**, using Gunicorn behind an HTTPS reverse proxy. WhiteNoise serves collected static files through the application, including compressed CSS and filenames with content hashes for caching.

Set these variables in both the release environment and the running service. Generate a long random secret once and keep it in your hosting environment's secret storage:

```bash
export DJANGO_DEBUG=false
export DJANGO_SECRET_KEY='<your-persistent-random-secret>'
export DJANGO_ALLOWED_HOSTS=churn.example.com
export DJANGO_TRUST_PROXY_HEADERS=true

python -m pip install --require-hashes -r requirements.txt
python train_model.py
python manage.py check --deploy --fail-level WARNING
python manage.py collectstatic --noinput
gunicorn churn_dashboard.wsgi:application --bind 127.0.0.1:8000 --workers 2 --access-logfile - --error-logfile -
```

Use [deploy/nginx.conf.example](deploy/nginx.conf.example) for the reverse proxy, replacing the domain and certificate paths with your own. Install a valid TLS certificate before enabling that configuration, validate it with `nginx -t`, and reload Nginx. Run Gunicorn under your hosting platform's process supervisor. Production settings enforce HTTPS and one year of HSTS, including subdomains.

`DJANGO_TRUST_PROXY_HEADERS` defaults to `false`. Enable it only when the trusted proxy **overwrites** `X-Forwarded-Proto` and the application port is accessible only to that proxy. The example binds Gunicorn to loopback and replaces that header with Nginx's request scheme, so Django recognizes HTTPS and avoids redirect loops. If your hosting platform supplies the proxy, configure the equivalent behavior there. See [Django's proxy setting documentation](https://docs.djangoproject.com/en/6.0/ref/settings/#secure-proxy-ssl-header).

Run `collectstatic` with production settings on each release and deploy its `staticfiles/` output alongside the application. WhiteNoise serves those files without a separate `/static/` rule in Nginx. See the [WhiteNoise deployment guide](https://whitenoise.readthedocs.io/en/stable/django.html).

Train the model during the build/release step and deploy `artifacts/churn_model.joblib` with its generated `artifacts/model_metadata.json`. The binary artifact is intentionally ignored by Git because it is generated and tied to dependency versions. The application returns a clear 503 response instead of retraining during a request when either file is missing, altered, or incompatible.

Verify the deployment before directing users to it:

```bash
python manage.py test dashboard.test_deployment --verbosity 2
curl --fail --head https://churn.example.com/
curl --fail --head https://churn.example.com/static/dashboard/styles.css
```

The automated smoke test loads the real production settings in isolated processes, collects static files into temporary directories, verifies compressed and cached CSS delivery, and checks HTTPS handling with proxy trust both enabled and disabled. The `curl` checks verify the deployed site's public HTTPS route and stylesheet.

## Data and limitations

The repository includes `Telco-Customer-Churn.csv` with 7,043 rows. Its authoritative source URL and redistribution license are not recorded in the original project, so verify and document them before public or commercial redistribution.

This is a cross-sectional demonstration dataset with no explicit prediction horizon. Scores show patterns in this dataset; they are not causal claims or evidence that a retention action will work. Review drift, calibration, subgroup performance, privacy, and intervention outcomes before operational use. See [MODEL_CARD.md](MODEL_CARD.md).
