-- Витрина признаков для скоринга: заявка + агрегаты по внешним кредитам.
-- Демонстрирует тот же feature engineering, что и src/features.py, но на SQL.

WITH bureau_agg AS (
    SELECT
        sk_id_curr,
        COUNT(*)                                              AS bureau_loan_count,
        COUNT(*) FILTER (WHERE credit_active = 'Active')      AS bureau_active_count,
        AVG(days_credit)                                      AS bureau_days_credit_mean,
        MAX(credit_day_overdue)                               AS bureau_overdue_max,
        COUNT(*) FILTER (WHERE credit_day_overdue > 0)        AS bureau_overdue_loans,
        SUM(amt_credit_sum_debt)                              AS bureau_debt_sum,
        SUM(amt_credit_sum)                                   AS bureau_credit_sum
    FROM bureau
    GROUP BY sk_id_curr
)
SELECT
    a.sk_id_curr,
    a.target,
    -- Ратио-признаки
    a.amt_credit / NULLIF(a.amt_income_total, 0)          AS credit_to_income,
    a.amt_annuity / NULLIF(a.amt_income_total, 0)         AS annuity_to_income,
    a.amt_credit / NULLIF(a.amt_annuity, 0)               AS credit_to_annuity,
    -a.days_employed / 365.25                             AS employed_years,
    -a.days_birth / 365.25                                AS age_years,
    a.amt_income_total / NULLIF(a.cnt_fam_members, 0)     AS income_per_fam_member,
    -- Базовые поля заявки
    a.name_contract_type,
    a.code_gender,
    a.name_education_type,
    a.name_income_type,
    a.occupation_type,
    a.name_type_suite,
    a.name_family_status,
    a.name_housing_type,
    a.occupation_type,
    a.organization_type,
    a.name_income_type,
    a.weekday_appr_process_start,
    a.housetype_mode,
    a.wallsmaterial_mode,
    a.emergencystate_mode,
    -- Агрегаты бюро
    b.bureau_loan_count,
    b.bureau_active_count,
    b.bureau_days_credit_mean,
    b.bureau_overdue_max,
    b.bureau_overdue_loans,
    b.bureau_debt_sum / NULLIF(b.bureau_credit_sum, 0)    AS bureau_debt_to_credit
FROM application a
LEFT JOIN bureau_agg b USING (sk_id_curr)
WHERE a.code_gender <> 'XNA'