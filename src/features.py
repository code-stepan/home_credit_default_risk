"""Загрузка Home Credit, очистка аномалий, генерация признаков."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = str(PROJECT_ROOT / "data")
ANOMALY_DAYS_EMPLOYED = 365243  # служебная "заглушка" в датасете

CAT_FEATURES = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE", "OCCUPATION_TYPE", "ORGANIZATION_TYPE",
    "NAME_TYPE_SUITE", "WEEKDAY_APPR_PROCESS_START",
    "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
]


def build_dataset(data_dir: str = DATA_DIR) -> pd.DataFrame:
    app = pd.read_csv(f"{data_dir}/application_train.csv")
    bureau = pd.read_csv(f"{data_dir}/bureau.csv")

    df = clean_application(app)
    df = add_ratio_features(df)
    bureau_features = build_bureau_features(bureau)
    df = df.merge(bureau_features, on="SK_ID_CURR", how="left")
    return df


def clean_application(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"] == ANOMALY_DAYS_EMPLOYED).astype(int)
    df.loc[df["DAYS_EMPLOYED"] == ANOMALY_DAYS_EMPLOYED, "DAYS_EMPLOYED"] = np.nan
    # CODE_GENDER == 'XNA' (4 строки) -> самый частый класс
    df.loc[df["CODE_GENDER"] == "XNA", "CODE_GENDER"] = "F"
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CREDIT_TO_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_TO_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TO_ANNUITY"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["EMPLOYED_TO_AGE"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["INCOME_PER_FAM_MEMBER"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    return df


def build_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    """Агрегаты по внешним кредитам заёмщика из таблицы bureau."""
    agg = bureau.groupby("SK_ID_CURR").agg(
        BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("CREDIT_ACTIVE", lambda s: int((s == "Active").sum())),
        BUREAU_DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        BUREAU_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_OVERDUE_LOANS=("CREDIT_DAY_OVERDUE", lambda s: int((s > 0).sum())),
        BUREAU_DEBT_SUM=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUREAU_CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
    )
    agg["BUREAU_DEBT_TO_CREDIT"] = agg["BUREAU_DEBT_SUM"] / agg["BUREAU_CREDIT_SUM"].replace(0, np.nan)
    return agg.reset_index()


def select_features(df: pd.DataFrame, target_col: str = "TARGET",
                    id_cols: tuple[str, ...] = ("SK_ID_CURR",),
                    max_missing_share: float = 0.6) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Отбор признаков: убираем ID, таргет и колонки с большим числом пропусков."""
    y = df[target_col].copy()
    missing = df.isna().mean()
    drop = list(id_cols) + [target_col] + missing[missing > max_missing_share].index.tolist()
    X = df.drop(columns=[c for c in drop if c in df.columns])
    cat_cols = [c for c in CAT_FEATURES if c in X.columns]
    return X, y, cat_cols


def prepare_catboost_matrix(X: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """CatBoost требует строки в категориальных признаках без NaN."""
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].fillna("__NA__").astype(str)
    return X