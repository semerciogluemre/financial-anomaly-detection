-- =============================================================================
-- 02_amount_stats.sql
-- Per-card amount statistics and z-score for the current transaction.
-- SQLite lacks native STDDEV, so we use the computational formula:
--   stddev = SQRT(E[X²] - (E[X])²)
-- A z-score > 3 or < -3 flags a statistically anomalous amount.
-- =============================================================================

WITH card_stats AS (
    SELECT
        card1,
        COUNT(TransactionID)                                            AS txn_count,
        AVG(TransactionAmt)                                             AS amt_mean,
        MIN(TransactionAmt)                                             AS amt_min,
        MAX(TransactionAmt)                                             AS amt_max,
        -- Population stddev via computational formula
        SQRT(
            MAX(
                AVG(TransactionAmt * TransactionAmt)
                - AVG(TransactionAmt) * AVG(TransactionAmt),
                0.0   -- guard against floating-point negatives near zero
            )
        )                                                               AS amt_stddev
    FROM fraud_combined
    WHERE TransactionAmt IS NOT NULL
    GROUP BY card1
)

SELECT
    fc.TransactionID,
    fc.card1,
    fc.TransactionAmt,
    fc.isFraud,

    cs.txn_count        AS card_txn_count,
    cs.amt_mean         AS card_amt_mean,
    cs.amt_stddev       AS card_amt_stddev,
    cs.amt_min          AS card_amt_min,
    cs.amt_max          AS card_amt_max,

    -- Z-score: how many std-devs this amount is from the card mean
    CASE
        WHEN cs.amt_stddev > 0
        THEN (fc.TransactionAmt - cs.amt_mean) / cs.amt_stddev
        ELSE 0.0
    END                 AS amt_zscore,

    -- Flag: |z| > 3 is statistically anomalous
    CASE
        WHEN cs.amt_stddev > 0
         AND ABS((fc.TransactionAmt - cs.amt_mean) / cs.amt_stddev) > 3
        THEN 1 ELSE 0
    END                 AS amt_anomaly_flag,

    -- Ratio of this transaction to the card's historical mean
    CASE
        WHEN cs.amt_mean > 0
        THEN fc.TransactionAmt / cs.amt_mean
        ELSE NULL
    END                 AS amt_to_mean_ratio

FROM fraud_combined fc
JOIN card_stats cs ON fc.card1 = cs.card1
ORDER BY fc.TransactionID;
