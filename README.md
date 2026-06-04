# Telco Churn Dashboard

This project now includes two dashboard entry points backed by the same churn analytics code:

- Django dashboard: `python manage.py runserver`
- Streamlit dashboard: `streamlit run streamlit_app.py`

The shared module `dashboard_core.py` loads `Telco-Customer-Churn.csv`, cleans blank `TotalCharges`, trains the notebook-selected Random Forest model, builds dashboard charts, and scores customer churn risk using the six strongest predictors: tenure, contract, total charges, internet service, monthly charges, and payment method.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Run Django

```powershell
python manage.py runserver
```

Open `http://127.0.0.1:8000`.

## Run Streamlit

```powershell
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.
