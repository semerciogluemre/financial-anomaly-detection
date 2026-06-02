-- =============================================================================
-- 01_transaction_velocity.sql
-- Transaction velocity features per card (card1).
-- TransactionDT is a timedelta in seconds from a reference datetime.
-- We compute rolling counts over the last 1h and 24h by self-joining on
-- the same card1 where the prior DT falls within the window.
-- =============================================================================

WITH velocity AS (
    SELECT
        t.TransactionID,
        t.card1,
        t.TransactionDT,
        t.isFraud,

        -- Count of transactions by the same card in the past 1 hour (3600 s)
        COUNT(prev.TransactionID) AS txn_count_1h,

        -- Count of transactions by the same card in the past 24 hours (86400 s)
        SUM(
            CASE
                WHEN prev.TransactionDT >= t.TransactionDT - 86400
                THEN 1 ELSE 0
            END
        ) AS txn_count_24h

    FROM fraud_combined t
    LEFT JOIN fraud_combined prev
        ON  prev.card1        = t.card1
        AND prev.TransactionID != t.TransactionID
        AND prev.TransactionDT >= t.TransactionDT - 3600
        AND prev.TransactionDT <  t.TransactionDT

    GROUP BY
        t.TransactionID,
        t.card1,
        t.TransactionDT,
        t.isFraud
),

velocity_stats AS (
    SELECT
        card1,
        AVG(txn_count_1h)               AS avg_1h_velocity,
        AVG(txn_count_1h) + 2 * (
            -- SQLite has no STDDEV; approximate with manual formula
            SQRT(
                AVG(txn_count_1h * txn_count_1h)
                - AVG(txn_count_1h) * AVG(txn_count_1h)
            )
        )                               AS high_velocity_threshold_1h,
        AVG(txn_count_24h)              AS avg_24h_velocity,
        AVG(txn_count_24h) + 2 * (
            SQRT(
                AVG(txn_count_24h * txn_count_24h)
                - AVG(txn_count_24h) * AVG(txn_count_24h)
            )
        )                               AS high_velocity_threshold_24h
    FROM velocity
    GROUP BY card1
)

SELECT
    v.TransactionID,
    v.card1,
    v.TransactionDT,
    v.isFraud,
    v.txn_count_1h,
    v.txn_count_24h,
    vs.avg_1h_velocity,
    vs.avg_24h_velocity,

    -- Binary flags: 1 when current transaction exceeds mean + 2σ
    CASE
        WHEN v.txn_count_1h > vs.high_velocity_threshold_1h  THEN 1 ELSE 0
    END AS high_velocity_flag_1h,

    CASE
        WHEN v.txn_count_24h > vs.high_velocity_threshold_24h THEN 1 ELSE 0
    END AS high_velocity_flag_24h

FROM velocity v
JOIN velocity_stats vs ON v.card1 = vs.card1
ORDER BY v.TransactionID;
