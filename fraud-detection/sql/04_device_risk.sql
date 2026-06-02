-- =============================================================================
-- 04_device_risk.sql
-- Device-level fraud risk scores.
-- Computes historical fraud rate per (DeviceType, DeviceInfo) pair and
-- per DeviceType alone (coarser fallback). Joins back to each transaction.
--
-- device_combo_fraud_rate : fraud rate for exact DeviceType+DeviceInfo combo
-- device_type_fraud_rate  : fraud rate for DeviceType only (coarser)
-- device_risk_score       : combo rate when available, else type rate
-- =============================================================================

WITH device_combo_stats AS (
    SELECT
        DeviceType,
        DeviceInfo,
        COUNT(*)                        AS combo_txn_count,
        SUM(COALESCE(isFraud, 0))       AS combo_fraud_count,
        AVG(COALESCE(isFraud, 0))       AS combo_fraud_rate
    FROM fraud_combined
    WHERE DeviceType IS NOT NULL
      AND DeviceInfo IS NOT NULL
    GROUP BY DeviceType, DeviceInfo
),

device_type_stats AS (
    SELECT
        DeviceType,
        COUNT(*)                        AS type_txn_count,
        SUM(COALESCE(isFraud, 0))       AS type_fraud_count,
        AVG(COALESCE(isFraud, 0))       AS type_fraud_rate
    FROM fraud_combined
    WHERE DeviceType IS NOT NULL
    GROUP BY DeviceType
)

SELECT
    fc.TransactionID,
    fc.card1,
    fc.isFraud,
    fc.DeviceType,
    fc.DeviceInfo,

    -- Combo-level stats
    dcs.combo_txn_count,
    dcs.combo_fraud_rate,

    -- Type-level stats (fallback)
    dts.type_txn_count,
    dts.type_fraud_rate,

    -- Best available risk score: use combo if seen ≥ 5 times, else type
    CASE
        WHEN dcs.combo_txn_count >= 5 THEN dcs.combo_fraud_rate
        WHEN dts.type_txn_count  >= 1 THEN dts.type_fraud_rate
        ELSE NULL
    END                                 AS device_risk_score,

    -- High-risk device flag (risk score > 0.1)
    CASE
        WHEN COALESCE(dcs.combo_fraud_rate, dts.type_fraud_rate, 0) > 0.1
        THEN 1 ELSE 0
    END                                 AS high_risk_device_flag,

    -- Unknown device flag (not seen in training)
    CASE WHEN dcs.combo_txn_count IS NULL THEN 1 ELSE 0 END
                                        AS unknown_device_flag

FROM fraud_combined fc
LEFT JOIN device_combo_stats dcs
       ON  fc.DeviceType = dcs.DeviceType
       AND fc.DeviceInfo = dcs.DeviceInfo
LEFT JOIN device_type_stats dts
       ON  fc.DeviceType = dts.DeviceType
ORDER BY fc.TransactionID;
