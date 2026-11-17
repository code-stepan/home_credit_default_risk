from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import evaluate as ev
from src.features import build_dataset, prepare_catboost_matrix, select_features
from src.woe import WOETransformer

SEED = 42
N_SPLITS = 5

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"


def make_logreg_pipeline() -> Pipeline:
    return Pipeline([
        ("woe", WOETransformer(n_bins=10)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=SEED))
    ])


def make_catboost(cat_cols: list[str]) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5,
        auto_class_weights="Balanced",
        cat_features=cat_cols,
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )


def cross_validate(model_fn, X, y, cat_cols=None):
    """Ручная CV: нужен контроль fit-параметров CatBoost (cat_features)."""
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED)
    scores = []
    for tr_idx, va_idx in skf.split(X, y):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx].reset_index(drop=True), y.iloc[va_idx].reset_index(drop=True)
        model = model_fn()
        if cat_cols:
            model.fit(Xtr, ytr)
        else:
            model.fit(Xtr, ytr)
        proba = model.predict_proba(Xva)[:, 1]
        scores.append(roc_auc_score(yva, proba))
    return scores


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print("1/5 Загрузка и подготовка данных...")
    df = build_dataset()
    X, y, cat_cols = select_features(df)
    y = y.reset_index(drop=True)
    print(f"   Признаков: {X.shape[1]}, наблюдений: {len(X)}")

    X_train, X_ho, y_train, y_ho = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    y_train, y_ho = y_train.reset_index(drop=True), y_ho.reset_index(drop=True)

    print("2/5 Кросс-валидация логистической регрессии (WOE)...")
    lr_scores = cross_validate(make_logreg_pipeline, X_train, y_train)
    print(f"   ROC-AUC CV: {np.mean(lr_scores):.4f} ± {np.std(lr_scores):.4f}")

    print("3/5 Кросс-валидация CatBoost...")
    X_train_cb = prepare_catboost_matrix(X_train, cat_cols)
    X_ho_cb = prepare_catboost_matrix(X_ho, cat_cols)
    cb_scores = cross_validate(
        lambda: make_catboost(cat_cols), X_train_cb, y_train, cat_cols
    )
    print(f"   ROC-AUC CV: {np.mean(cb_scores):.4f} ± {np.std(cb_scores):.4f}")

    print("4/5 Финальное обучение и оценка на holdout...")
    lr = make_logreg_pipeline().fit(X_train, y_train)
    cb = make_catboost(cat_cols).fit(X_train_cb, y_train)

    lr_proba = lr.predict_proba(X_ho)[:, 1]
    cb_proba = cb.predict_proba(X_ho_cb)[:, 1]
    lr_train_proba = lr.predict_proba(X_train)[:, 1]
    cb_train_proba = cb.predict_proba(X_train_cb)[:, 1]

    metrics = {
        "logreg_woe": {
            "cv_roc_auc_mean": round(float(np.mean(lr_scores)), 4),
            **ev.full_report(y_ho, lr_proba),
            "psi_holdout_vs_train": round(ev.psi(lr_train_proba, lr_proba), 4),
        },
        "catboost": {
            "cv_roc_auc_mean": round(float(np.mean(cb_scores)), 4),
            **ev.full_report(y_ho, cb_proba),
            "psi_holdout_vs_train": round(ev.psi(cb_train_proba, cb_proba), 4),
        },
    }
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # Выбираем лучшую модель по ROC-AUC на holdout
    if metrics["catboost"]["roc_auc"] >= metrics["logreg_woe"]["roc_auc"]:
        best_name, best_model, best_proba = "catboost", cb, cb_proba
    else:
        best_name, best_model, best_proba = "logreg_woe", lr, lr_proba
    print(f"   Лучшая модель: {best_name}")

    ev.save_plots(y_ho, best_proba, prefix=str(REPORTS_DIR / best_name))
    metrics["best_model"] = best_name
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    joblib.dump(best_model, MODELS_DIR / "model.joblib")
    with open(MODELS_DIR / "feature_list.json", "w", encoding="utf-8") as f:
        json.dump(list(X.columns), f, ensure_ascii=False)

    print("5/5 SHAP-интерпретация...")
    if best_name == "catboost":
        explainer = shap.TreeExplainer(cb)
        sample = X_ho_cb.sample(min(2000, len(X_ho_cb)), random_state=SEED)
        shap_values = explainer.shap_values(sample)
        plt.figure()
        shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "shap_summary.png", dpi=150)
        plt.close()
    else:
        # Для логрега — важность через |коэффициент| * std признака
        woe_X = lr.named_steps["woe"].transform(X_ho)
        scaled = lr.named_steps["scaler"].transform(woe_X)
        coefs = np.abs(lr.named_steps["clf"].coef_[0])
        imp = pd.Series(coefs, index=X_ho.columns).sort_values().tail(15)
        imp.plot.barh(figsize=(8, 6), color="#3b82f6")
        plt.title("Топ-15 признаков (|β| в стандартизованном пространстве)")
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / "logreg_importance.png", dpi=150)
        plt.close()

    print("Готово. Артефакты: models/, reports/")


if __name__ == "__main__":
    main()