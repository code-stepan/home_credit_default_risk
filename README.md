# Кредитный скоринг: предсказание дефолта заёмщика

Пет-проект бинарной классификации на данных [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk)
(307 000 заявок, 100+ признаков). Построен полный цикл: от EDA до REST API.

## Задача
Предсказать вероятность дефолта по кредитной заявке и оценить модель
банковскими метриками (Gini, KS, PSI) с объяснимостью через SHAP.

## Результаты
| Модель | ROC-AUC (CV) | ROC-AUC (holdout) | Gini | KS | PSI |
|---|---|---|---|---|---|
| LogReg + WOE | <подставь> | <...> | <...> | <...> | <...> |
| CatBoost | <подставь> | <...> | <...> | <...> | <...> |

## Что внутри
- EDA: дефолтность по возрасту/доходу/образованию, анализ пропусков (`notebooks/01_eda.ipynb`)
- Feature engineering: ратио-признаки (DTI), агрегаты кредитной истории из bureau,
  обработка аномалии DAYS_EMPLOYED=365243 (`src/features.py`)
- WOE/IV-биннинг с подсчётом Information Value (`src/woe.py`)
- Сравнение: интерпретируемая логрегрессия на WOE vs CatBoost (`src/train.py`)
- Валидация: стратифицированная 5-fold CV, holdout, PSI-мониторинг дрейфа (`src/evaluate.py`)
- Объяснимость: SHAP summary и разбор отдельных заявок (`src/train.py`, `src/main.py`)
- Продакшн: FastAPI + Docker, признаки собираются SQL-запросом (`sql/load_features.sql`)

## Стек
Python, Pandas, NumPy, scikit-learn, CatBoost, SHAP, FastAPI, Docker, PostgreSQL (SQL)

## Запуск
см. раздел Quickstart выше / команды в репозитории