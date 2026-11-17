"""REST API скоринговой модели.

Запуск: uvicorn src.main:app --reload
Swagger: http://127.0.0.1:8000/docs

POST /predict — принимает заявку, возвращает вероятность дефолта и решение.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.features import add_ratio_features, clean_application

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
FEATURES_PATH = os.getenv("FEATURES_PATH", "models/feature_list.json")
METRICS_PATH = "models/metrics.json"

app = FastAPI(title="Credit Scoring API", version="1.0")

# Значения по умолчанию
THRESHOLD = 0.5

# Загрузка моделей и фичей, если они существуют
try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, encoding="utf-8") as f:
        FEATURES: list[str] = json.load(f)
    
    # Загрузка порога если метрики существуют
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = json.load(f)
            if metrics.get("best_model") == "catboost":
                THRESHOLD = metrics.get("catboost", {}).get("best_threshold_f1", 0.5)
            else:
                THRESHOLD = metrics.get("logreg_woe", {}).get("best_threshold_f1", 0.5)
except Exception:
    # Если модели не загружены, имитация для тестов
    print("WARNING: Model or features not loaded. API may not work correctly.")
    model = None
    FEATURES = []

# Значения по умолчанию для полей, которые клиент не передал:
# модель увидит NaN и попадёт в WOE-бин пропусков — так же, как при обучении.
FIELD_MAP = {
    "name_contract_type": "NAME_CONTRACT_TYPE",
    "code_gender": "CODE_GENDER",
    "flag_own_car": "FLAG_OWN_CAR",
    "flag_own_realty": "FLAG_OWN_REALTY",
    "amt_income_total": "AMT_INCOME_TOTAL",
    "amt_credit": "AMT_CREDIT",
    "amt_annuity": "AMT_ANNUITY",
    "days_birth": "DAYS_BIRTH",
    "days_employed": "DAYS_EMPLOYED",
    "cnt_fam_members": "CNT_FAM_MEMBERS",
    "name_income_type": "NAME_INCOME_TYPE",
    "name_education_type": "NAME_EDUCATION_TYPE",
    "name_family_status": "NAME_FAMILY_STATUS",
    "name_housing_type": "NAME_HOUSING_TYPE",
    "occupation_type": "OCCUPATION_TYPE",
    "organization_type": "ORGANIZATION_TYPE",
    "bureau_loan_count": "BUREAU_LOAN_COUNT",
    "bureau_active_count": "BUREAU_ACTIVE_COUNT",
    "bureau_days_credit_mean": "BUREAU_DAYS_CREDIT_MEAN",
    "bureau_overdue_max": "BUREAU_OVERDUE_MAX",
    "bureau_overdue_loans": "BUREAU_OVERDUE_LOANS",
    "bureau_debt_to_credit": "BUREAU_DEBT_TO_CREDIT",
}


class LoanApplication(BaseModel):
    name_contract_type: str = Field(default="Cash loans")
    code_gender: str = Field(default="F")
    flag_own_car: str = Field(default="N")
    flag_own_realty: str = Field(default="Y")
    amt_income_total: float = Field(gt=0, description="Доход, руб.")
    amt_credit: float = Field(gt=0, description="Сумма кредита, руб.")
    amt_annuity: Optional[float] = None
    days_birth: int = Field(le=-1, description="Возраст в днях, отрицательное число")
    days_employed: Optional[int] = Field(default=None, le=-1)
    cnt_fam_members: float = Field(default=1.0, gt=0)
    name_income_type: str = Field(default="Working")
    name_education_type: str = Field(default="Secondary / secondary special")
    name_family_status: str = Field(default="Married")
    name_housing_type: str = Field(default="House / apartment")
    occupation_type: Optional[str] = None
    organization_type: Optional[str] = None
    bureau_loan_count: Optional[float] = None
    bureau_active_count: Optional[float] = None
    bureau_days_credit_mean: Optional[float] = None
    bureau_overdue_max: Optional[float] = None
    bureau_overdue_loans: Optional[float] = None
    bureau_debt_to_credit: Optional[float] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "n_features": len(FEATURES) if FEATURES else 0, "threshold": THRESHOLD}


@app.post("/predict")
def predict(application: LoanApplication) -> dict:
    if model is None:
        return {"error": "Model not loaded"}
    
    row = {FIELD_MAP[k]: v for k, v in application.model_dump().items()}
    df = pd.DataFrame([row])
    df = add_ratio_features(clean_application(df))
    df = df.reindex(columns=FEATURES)  # нехватка полей -> NaN -> WOE-бин пропусков

    proba = float(model.predict_proba(df)[:, 1][0])
    return {
        "default_probability": round(proba, 4),
        "threshold": THRESHOLD,
        "decision": "DECLINE" if proba >= THRESHOLD else "APPROVE",
    }