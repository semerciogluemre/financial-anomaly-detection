-- =============================================================================
-- 06_email_domain_risk.sql
-- Historical fraud rate per email domain (payer and recipient).
-- High-risk threshold: fraud_rate > 0.10 (10%).
-- For domains seen fewer than 10 times we fall back to the global fraud rate
-- to avoid noisy estimates from rare domains.
-- =============================================================================

WITH global_stats AS (
    SELECT AVG(COALESCE(isFraud, 0)) AS global_fraud_rate
    FROM fraud_combined
),

p_domain_stats AS (
    SELECT
        P_emaildomain,
        COUNT(*)                        AS p_domain_txn_count,
        SUM(COALESCE(isFraud, 0))       AS p_domain_fraud_count,
        AVG(COALESCE(isFraud, 0))       AS p_domain_fraud_rate
    FROM fraud_combined
    WHERE P_emaildomain IS NOT NULL
    GROUP BY P_emaildomain
),

r_domain_stats AS (
    SELECT
        R_emaildomain,
        COUNT(*)                        AS r_domain_txn_count,
        SUM(COALESCE(isFraud, 0))       AS r_domain_fraud_count,
        AVG(COALESCE(isFraud, 0))       AS r_domain_fraud_rate
    FROM fraud_combined
    WHERE R_emaildomain IS NOT NULL
    GROUP BY R_emaildomain
)

SELECT
    fc.TransactionID,
    fc.card1,
    fc.isFraud,
    fc.P_emaildomain,
    fc.R_emaildomain,

    -- ── Payer domain features ─────────────────────────────────────────────
    pds.p_domain_txn_count,
    pds.p_domain_fraud_rate,

    -- Smoothed rate: use domain rate if ≥ 10 obs, else global fallback
    CASE
        WHEN pds.p_domain_txn_count >= 10 THEN pds.p_domain_fraud_rate
        ELSE gs.global_fraud_rate
    END                                     AS p_domain_risk_score,

    CASE
        WHEN COALESCE(pds.p_domain_fraud_rate, 0) > 0.1 THEN 1 ELSE 0
    END                                     AS p_domain_high_risk_flag,

    CASE WHEN pds.p_domain_txn_count IS NULL THEN 1 ELSE 0 END
                                            AS p_domain_unknown_flag,

    -- ── Recipient domain features ─────────────────────────────────────────
    rds.r_domain_txn_count,
    rds.r_domain_fraud_rate,

    CASE
        WHEN rds.r_domain_txn_count >= 10 THEN rds.r_domain_fraud_rate
        ELSE gs.global_fraud_rate
    END                                     AS r_domain_risk_score,

    CASE
        WHEN COALESCE(rds.r_domain_fraud_rate, 0) > 0.1 THEN 1 ELSE 0
    END                                     AS r_domain_high_risk_flag,

    CASE WHEN rds.r_domain_txn_count IS NULL THEN 1 ELSE 0 END
                                            AS r_domain_unknown_flag,

    -- ── Combined domain risk ──────────────────────────────────────────────
    -- Max of payer and recipient risk scores
    MAX(
        COALESCE(
            CASE WHEN pds.p_domain_txn_count >= 10
                 THEN pds.p_domain_fraud_rate ELSE gs.global_fraud_rate END,
            gs.global_fraud_rate
        ),
        COALESCE(
            CASE WHEN rds.r_domain_txn_count >= 10
                 THEN rds.r_domain_fraud_rate ELSE gs.global_fraud_rate END,
            gs.global_fraud_rate
        )
    )                                       AS combined_domain_risk_score

FROM fraud_combined fc
CROSS JOIN global_stats gs
LEFT JOIN p_domain_stats pds ON fc.P_emaildomain = pds.P_emaildomain
LEFT JOIN r_domain_stats rds ON fc.R_emaildomain = rds.R_emaildomain
ORDER BY fc.TransactionID;
