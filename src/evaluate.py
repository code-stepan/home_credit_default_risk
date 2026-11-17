from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def gini(y_true, y_score) -> float:
    """Gini = 2 * AUC - 1"""
    return 2 * roc_auc_score(y_true, y_score) - 1


def ks(y_true, y_score) -> float:
    """Статистика Колмогорова-Смирнова: максимальный разрыв между
    кумулятивными распределениями скоров 'хороших' и 'плохих'."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index по скорам.

    expected — скоры на обучении (эталонное распределение),
    actual   — скоры на новых данных (holdout / прод).
    PSI < 0.1 — стабильна, 0.1-0.25 — умеренный сдвиг, > 0.25 — модель дрейфует.
    """
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    # pd.cut -> Categorical.value_counts(normalize) не работает в старом pandas; оборачиваем в Series
    cats_e = pd.cut(expected, edges, include_lowest=True)
    cats_a = pd.cut(actual, edges, include_lowest=True)
    e = pd.Series(cats_e).value_counts(normalize=True, sort=False)
    a = pd.Series(cats_a).value_counts(normalize=True, sort=False)
    e, a = e + 1e-6, a + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def best_threshold_f1(y_true, y_score) -> tuple[float, float]:
    """Порог, максимизирующий F1. В проде порог выбирают по матрице
    стоимости ошибок банка (цена пропущенного дефолта vs упущенная сделка)."""
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    i = int(np.nanargmax(f1[:-1]))
    return float(thr[i]), float(f1[i])


def full_report(y_true, y_score) -> dict:
    thr, f1 = best_threshold_f1(y_true, y_score)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "gini": round(gini(y_true, y_score), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "ks": round(ks(y_true, y_score), 4),
        "best_threshold_f1": round(thr, 4),
        "f1_at_threshold": round(f1, 4),
    }


def save_plots(y_true, y_score, prefix: str = "reports/model") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))  # noqa: RUF059

    fpr, tpr, _ = roc_curve(y_true, y_score)
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_score):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set(title="ROC-кривая", xlabel="FPR", ylabel="TPR")
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    axes[1].plot(rec, prec, label=f"PR-AUC = {average_precision_score(y_true, y_score):.3f}")
    axes[1].set(title="Precision-Recall", xlabel="Recall", ylabel="Precision")
    axes[1].legend()

    axes[2].hist(y_score[y_true == 0], bins=50, alpha=0.6, label="Не дефолт", density=True)
    axes[2].hist(y_score[y_true == 1], bins=50, alpha=0.6, label="Дефолт", density=True)
    axes[2].set(title="Распределение скоров", xlabel="P(дефолт)")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{prefix}_curves.png", dpi=150)
    plt.close()