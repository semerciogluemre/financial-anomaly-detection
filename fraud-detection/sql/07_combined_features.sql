-- =============================================================================
-- 07_combined_features.sql
-- Master feature matrix — joins all feature blocks into one row per transaction.
-- This is the canonical query called by the Python FeatureEngineer pipeline.
--
-- NOTE: Velocity features (txn_count_1h / 24h) are computed in Python via
-- pandas rolling windows (see FeatureEngineer.build_feature_matrix) because
-- a self-join on 590k rows is prohibitively slow in SQLite.
--
-- Output columns (grouped by source):
--   [base]       TransactionID, isFraud, TransactionAmt, ProductCD, card1-6
--   [amount]     amt_zscore, amt_anomaly_flag, amt_to_mean_ratio,
--                card_amt_mean, card_amt_stddev
--   [time]       hour_of_day, day_of_week, is_off_hours, is_weekend,
--                hour_fraud_rate
--   [device]     device_risk_score, high_risk_device_flag, unknown_device_flag
--   [address]    addr_mismatch_flag, card_mismatch_rate, volatile_addr_flag
--   [email]      p_domain_risk_score, r_domain_risk_score,
--                combined_domain_risk_score, p/r_domain_high_risk_flag
-- =============================================================================

WITH

-- ── 1. Amount stats ──────────────────────────────────────────────────────────
card_stats AS (
    SELECT
        card1,
        AVG(TransactionAmt)                                     AS amt_mean,
        SQRT(MAX(
            AVG(TransactionAmt * TransactionAmt)
            - AVG(TransactionAmt) * AVG(TransactionAmt), 0))    AS amt_stddev
    FROM fraud_combined
    WHERE TransactionAmt IS NOT NULL
    GROUP BY card1
),

-- ── 2. Time features ─────────────────────────────────────────────────────────
time_base AS (
    SELECT
        TransactionID,
        CAST(((TransactionDT + 1512009600) % 86400) / 3600 AS INTEGER)  AS hour_of_day,
        CAST(strftime('%w', datetime(TransactionDT + 1512009600, 'unixepoch')) AS INTEGER)
                                                                         AS day_of_week
    FROM fraud_combined
    WHERE TransactionDT IS NOT NULL
),

hour_agg AS (
    SELECT
        hour_of_day,
        AVG(COALESCE(fc.isFraud, 0)) AS hour_fraud_rate
    FROM time_base tb
    JOIN fraud_combined fc ON tb.TransactionID = fc.TransactionID
    GROUP BY hour_of_day
),

-- ── 3. Device risk ───────────────────────────────────────────────────────────
device_combo AS (
    SELECT
        DeviceType, DeviceInfo,
        COUNT(*)                    AS combo_count,
        AVG(COALESCE(isFraud, 0))   AS combo_rate
    FROM fraud_combined
    WHERE DeviceType IS NOT NULL AND DeviceInfo IS NOT NULL
    GROUP BY DeviceType, DeviceInfo
),

device_type AS (
    SELECT
        DeviceType,
        COUNT(*)                    AS type_count,
        AVG(COALESCE(isFraud, 0))   AS type_rate
    FROM fraud_combined
    WHERE DeviceType IS NOT NULL
    GROUP BY DeviceType
),

-- ── 4. Address mismatch ──────────────────────────────────────────────────────
addr_stats AS (
    SELECT
        card1,
        AVG(CASE WHEN addr1 IS NOT NULL AND addr2 IS NOT NULL
                  AND addr1 != addr2 THEN 1.0 ELSE 0.0 END) AS card_mismatch_rate,
        COUNT(DISTINCT addr1)                                AS distinct_addr1_count
    FROM fraud_combined
    GROUP BY card1
),

-- ── 5. Email domain risk ─────────────────────────────────────────────────────
global_rate AS (
    SELECT AVG(COALESCE(isFraud, 0)) AS g FROM fraud_combined
),

p_domain AS (
    SELECT P_emaildomain,
           COUNT(*)                  AS p_cnt,
           AVG(COALESCE(isFraud,0))  AS p_rate
    FROM fraud_combined WHERE P_emaildomain IS NOT NULL
    GROUP BY P_emaildomain
),

r_domain AS (
    SELECT R_emaildomain,
           COUNT(*)                  AS r_cnt,
           AVG(COALESCE(isFraud,0))  AS r_rate
    FROM fraud_combined WHERE R_emaildomain IS NOT NULL
    GROUP BY R_emaildomain
)

-- ── Final join ───────────────────────────────────────────────────────────────
SELECT
    -- Base
    fc.TransactionID,
    fc.isFraud,
    fc.TransactionAmt,
    fc.ProductCD,
    fc.card1,
    fc.card2,
    fc.card3,
    fc.card4,
    fc.card5,
    fc.card6,
    fc.TransactionDT,

    -- Amount
    CASE WHEN cs.amt_stddev > 0
         THEN (fc.TransactionAmt - cs.amt_mean) / cs.amt_stddev
         ELSE 0.0 END                                               AS amt_zscore,
    CASE WHEN cs.amt_stddev > 0
          AND ABS((fc.TransactionAmt - cs.amt_mean) / cs.amt_stddev) > 3
         THEN 1 ELSE 0 END                                          AS amt_anomaly_flag,
    CASE WHEN cs.amt_mean > 0
         THEN fc.TransactionAmt / cs.amt_mean
         ELSE NULL END                                              AS amt_to_mean_ratio,
    cs.amt_mean                                                     AS card_amt_mean,
    cs.amt_stddev                                                   AS card_amt_stddev,

    -- Time
    tb.hour_of_day,
    tb.day_of_week,
    CASE WHEN tb.hour_of_day BETWEEN 0 AND 4 THEN 1 ELSE 0 END     AS is_off_hours,
    CASE WHEN tb.day_of_week IN (0, 6)        THEN 1 ELSE 0 END     AS is_weekend,
    ha.hour_fraud_rate,

    -- Device
    CASE
        WHEN dc.combo_count >= 5 THEN dc.combo_rate
        WHEN dt.type_count  >= 1 THEN dt.type_rate
        ELSE NULL
    END                                                             AS device_risk_score,
    CASE WHEN COALESCE(dc.combo_rate, dt.type_rate, 0) > 0.1
         THEN 1 ELSE 0 END                                          AS high_risk_device_flag,
    CASE WHEN dc.combo_count IS NULL THEN 1 ELSE 0 END              AS unknown_device_flag,

    -- Address
    CASE
        WHEN fc.addr1 IS NULL OR fc.addr2 IS NULL THEN NULL
        WHEN fc.addr1 != fc.addr2                 THEN 1
        ELSE 0
    END                                                             AS addr_mismatch_flag,
    ads.card_mismatch_rate,
    CASE WHEN ads.distinct_addr1_count > 3 THEN 1 ELSE 0 END        AS volatile_addr_flag,

    -- Email domains
    CASE WHEN pd.p_cnt >= 10 THEN pd.p_rate ELSE gr.g END           AS p_domain_risk_score,
    CASE WHEN rd.r_cnt >= 10 THEN rd.r_rate ELSE gr.g END           AS r_domain_risk_score,
    MAX(
        COALESCE(CASE WHEN pd.p_cnt >= 10 THEN pd.p_rate ELSE gr.g END, gr.g),
        COALESCE(CASE WHEN rd.r_cnt >= 10 THEN rd.r_rate ELSE gr.g END, gr.g)
    )                                                               AS combined_domain_risk_score,
    CASE WHEN COALESCE(pd.p_rate, 0) > 0.1 THEN 1 ELSE 0 END       AS p_domain_high_risk_flag,
    CASE WHEN COALESCE(rd.r_rate, 0) > 0.1 THEN 1 ELSE 0 END       AS r_domain_high_risk_flag

FROM fraud_combined fc
CROSS JOIN global_rate gr
LEFT JOIN card_stats      cs  ON fc.card1          = cs.card1
LEFT JOIN time_base       tb  ON fc.TransactionID  = tb.TransactionID
LEFT JOIN hour_agg        ha  ON tb.hour_of_day    = ha.hour_of_day
LEFT JOIN device_combo    dc  ON fc.DeviceType     = dc.DeviceType
                             AND fc.DeviceInfo     = dc.DeviceInfo
LEFT JOIN device_type     dt  ON fc.DeviceType     = dt.DeviceType
LEFT JOIN addr_stats      ads ON fc.card1          = ads.card1
LEFT JOIN p_domain        pd  ON fc.P_emaildomain  = pd.P_emaildomain
LEFT JOIN r_domain        rd  ON fc.R_emaildomain  = rd.R_emaildomain
ORDER BY fc.TransactionID;
