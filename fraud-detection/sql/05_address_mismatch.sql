-- =============================================================================
-- 05_address_mismatch.sql
-- Billing vs shipping address mismatch features.
-- addr1 = billing zip code proxy, addr2 = country code proxy.
-- Mismatch across transactions on the same card can signal stolen card use.
-- =============================================================================

WITH card_address_stats AS (
    SELECT
        card1,
        COUNT(*)                                                AS card_txn_count,

        -- How many transactions on this card had a mismatch
        SUM(
            CASE
                WHEN addr1 IS NOT NULL
                 AND addr2 IS NOT NULL
                 AND addr1 != addr2
                THEN 1 ELSE 0
            END
        )                                                       AS card_mismatch_count,

        -- Distinct addr1 values seen on this card (addr diversity)
        COUNT(DISTINCT addr1)                                   AS distinct_addr1_count,

        -- Distinct addr2 values seen on this card
        COUNT(DISTINCT addr2)                                   AS distinct_addr2_count,

        -- Mismatch rate for this card
        AVG(
            CASE
                WHEN addr1 IS NOT NULL
                 AND addr2 IS NOT NULL
                 AND addr1 != addr2
                THEN 1.0 ELSE 0.0
            END
        )                                                       AS card_mismatch_rate
    FROM fraud_combined
    GROUP BY card1
)

SELECT
    fc.TransactionID,
    fc.card1,
    fc.addr1,
    fc.addr2,
    fc.isFraud,

    -- Transaction-level mismatch flag
    CASE
        WHEN fc.addr1 IS NULL OR fc.addr2 IS NULL THEN NULL   -- unknown
        WHEN fc.addr1 != fc.addr2                 THEN 1
        ELSE 0
    END                                                         AS addr_mismatch_flag,

    -- Card-level aggregates
    cas.card_txn_count,
    cas.card_mismatch_count,
    cas.card_mismatch_rate,
    cas.distinct_addr1_count,
    cas.distinct_addr2_count,

    -- High-mismatch card flag: card has mismatched more than 50% of the time
    CASE
        WHEN cas.card_mismatch_rate > 0.5 THEN 1 ELSE 0
    END                                                         AS high_mismatch_card_flag,

    -- Address volatility: card has used many distinct billing addresses
    CASE
        WHEN cas.distinct_addr1_count > 3 THEN 1 ELSE 0
    END                                                         AS volatile_addr_flag

FROM fraud_combined fc
LEFT JOIN card_address_stats cas ON fc.card1 = cas.card1
ORDER BY fc.TransactionID;
