-- =============================================================================
-- 03_time_features.sql
-- Temporal features derived from TransactionDT.
-- TransactionDT is a delta in seconds from 2017-11-30 (competition reference).
-- Reference epoch (Unix): 1512009600  → 2017-11-30 00:00:00 UTC
--
-- hour_of_day  : 0–23
-- day_of_week  : 0 = Sunday … 6 = Saturday  (strftime %w convention)
-- is_off_hours : 1 when hour is 0–4 (midnight to 4:59 am)
-- =============================================================================

WITH time_base AS (
    SELECT
        TransactionID,
        card1,
        TransactionDT,
        TransactionAmt,
        isFraud,

        -- Absolute Unix timestamp
        TransactionDT + 1512009600                          AS unix_ts,

        -- Hour of day (0-23)
        CAST(
            ((TransactionDT + 1512009600) % 86400) / 3600
        AS INTEGER)                                         AS hour_of_day,

        -- Day of week via SQLite strftime (0=Sunday)
        CAST(
            strftime(
                '%w',
                datetime(TransactionDT + 1512009600, 'unixepoch')
            )
        AS INTEGER)                                         AS day_of_week

    FROM fraud_combined
    WHERE TransactionDT IS NOT NULL
),

time_agg AS (
    -- Fraud rate by hour bucket (for context / downstream use)
    SELECT
        hour_of_day,
        COUNT(*)                        AS hour_txn_count,
        AVG(COALESCE(isFraud, 0))       AS hour_fraud_rate
    FROM time_base
    GROUP BY hour_of_day
)

SELECT
    tb.TransactionID,
    tb.card1,
    tb.TransactionDT,
    tb.TransactionAmt,
    tb.isFraud,
    tb.unix_ts,
    tb.hour_of_day,
    tb.day_of_week,

    -- Off-hours flag: midnight through 4:59 am
    CASE WHEN tb.hour_of_day BETWEEN 0 AND 4 THEN 1 ELSE 0 END
                                        AS is_off_hours,

    -- Weekend flag (Saturday=6, Sunday=0)
    CASE WHEN tb.day_of_week IN (0, 6)  THEN 1 ELSE 0 END
                                        AS is_weekend,

    -- Hour-level fraud rate (historical base rate for that hour)
    ta.hour_fraud_rate,
    ta.hour_txn_count

FROM time_base tb
JOIN time_agg  ta ON tb.hour_of_day = ta.hour_of_day
ORDER BY tb.TransactionID;
