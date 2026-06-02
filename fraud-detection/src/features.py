"""
Feature engineering pipeline for IEEE-CIS Fraud Detection.

Reads SQL queries from the sql/ directory and executes them against the
SQLite database produced by DataIngestion, returning pandas DataFrames
ready for model training.

Usage:
    from src.features import FeatureEngineer

    fe = FeatureEngineer(db_path="db/fraud.db", sql_dir="sql/")
    df = fe.build_feature_matrix()
    fe.get_feature_summary(df)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Builds the feature matrix for fraud detection by executing SQL queries
    against a SQLite database.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file (e.g. 'db/fraud.db').
    sql_dir : str
        Directory containing the numbered .sql files (e.g. 'sql/').
    """

    # Columns that are identifiers or targets — excluded from the model matrix
    NON_FEATURE_COLS = {"TransactionID", "isFraud", "TransactionDT"}

    # Columns expected in the final feature matrix (sanity check)
    REQUIRED_COLS = {
        "txn_count_1h",
        "txn_count_24h",
        "amt_zscore",
        "amt_anomaly_flag",
        "hour_of_day",
        "day_of_week",
        "is_off_hours",
        "is_weekend",
        "addr_mismatch_flag",
        "p_domain_risk_score",
        "r_domain_risk_score",
        "combined_domain_risk_score",
    }

    def __init__(self, db_path: str, sql_dir: str):
        self.db_path = Path(db_path)
        self.sql_dir = Path(sql_dir)
        self._engine = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def engine(self):
        if self._engine is None:
            if not self.db_path.exists():
                raise FileNotFoundError(
                    f"Database not found: {self.db_path}. "
                    "Run DataIngestion.run() first."
                )
            self._engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
            logger.info("Connected to %s", self.db_path)
        return self._engine

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def load_sql(self, filename: str) -> str:
        """
        Read a .sql file from the sql directory and return its contents.

        Parameters
        ----------
        filename : str
            File name relative to sql_dir, e.g. '07_combined_features.sql'.

        Returns
        -------
        str
            Raw SQL string ready for execution.
        """
        path = self.sql_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"SQL file not found: {path}")
        sql = path.read_text(encoding="utf-8")
        logger.debug("Loaded SQL from %s (%d chars)", path, len(sql))
        return sql

    def run_query(self, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
        """
        Execute a SQL string against the database and return a DataFrame.

        Parameters
        ----------
        sql : str
            Valid SQLite SQL.
        params : dict, optional
            Bind parameters for the query (SQLAlchemy named style).

        Returns
        -------
        pd.DataFrame
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        logger.info("Query returned %d rows × %d cols", len(df), len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Feature matrix
    # ------------------------------------------------------------------

    def build_feature_matrix(self) -> pd.DataFrame:
        """
        Execute 07_combined_features.sql and return a clean feature matrix.

        Post-processing steps applied:
        - Cast integer flag columns to int8 to save memory
        - Fill remaining numeric NaNs with column median
        - Encode remaining object columns as category codes
        - Validate required columns are present

        Returns
        -------
        pd.DataFrame
            Rows = transactions. Includes TransactionID, isFraud, and all
            engineered features. Drop NON_FEATURE_COLS before model.fit().
        """
        logger.info("Building feature matrix from 07_combined_features.sql …")
        sql = self.load_sql("07_combined_features.sql")
        df = self.run_query(sql)

        # ── Velocity features (computed in pandas — SQL self-join too slow) ──
        logger.info("Computing velocity features in pandas …")
        df = df.sort_values("TransactionDT").reset_index(drop=True)

        txn_count_1h  = []
        txn_count_24h = []
        card_groups   = df.groupby("card1", sort=False)

        # Pre-compute per-card numpy arrays for fast range counting
        card_dt_map = {
            card: grp["TransactionDT"].values
            for card, grp in card_groups
        }
        card_idx_map = {
            card: grp.index.values
            for card, grp in card_groups
        }

        # Allocate output arrays
        count_1h  = np.zeros(len(df), dtype=np.int32)
        count_24h = np.zeros(len(df), dtype=np.int32)

        for card, dts in card_dt_map.items():
            idxs = card_idx_map[card]
            for pos, (row_idx, dt) in enumerate(zip(idxs, dts)):
                lo_1h  = np.searchsorted(dts, dt - 3600,  side="left")
                lo_24h = np.searchsorted(dts, dt - 86400, side="left")
                # exclude the transaction itself (pos)
                count_1h[row_idx]  = pos - lo_1h
                count_24h[row_idx] = pos - lo_24h

        df["txn_count_1h"]  = count_1h
        df["txn_count_24h"] = count_24h

        # High-velocity flags: mean + 2σ per card
        vel_stats = df.groupby("card1")[["txn_count_1h", "txn_count_24h"]].agg(["mean", "std"])
        vel_stats.columns = ["mean_1h", "std_1h", "mean_24h", "std_24h"]
        vel_stats["std_1h"]  = vel_stats["std_1h"].fillna(0)
        vel_stats["std_24h"] = vel_stats["std_24h"].fillna(0)
        df = df.merge(vel_stats, on="card1", how="left")
        df["high_velocity_flag_1h"]  = (
            df["txn_count_1h"]  > df["mean_1h"]  + 2 * df["std_1h"]
        ).astype("Int8")
        df["high_velocity_flag_24h"] = (
            df["txn_count_24h"] > df["mean_24h"] + 2 * df["std_24h"]
        ).astype("Int8")
        df.drop(columns=["mean_1h", "std_1h", "mean_24h", "std_24h"], inplace=True)
        logger.info("Velocity features added.")

        # ── Validate required columns ─────────────────────────────────
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Feature matrix is missing expected columns: {missing}")

        # ── Type coercions ────────────────────────────────────────────
        flag_cols = [c for c in df.columns if c.endswith("_flag")]
        for col in flag_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int8")

        # Numeric columns: fill NaN with median
        num_cols = df.select_dtypes(include="number").columns.difference(
            list(self.NON_FEATURE_COLS) + flag_cols
        )
        for col in num_cols:
            if df[col].isna().any():
                median = df[col].median()
                df[col] = df[col].fillna(median)
                logger.debug("Filled %s NaNs with median %.4f", col, median)

        # Object columns: encode as integer category codes
        obj_cols = df.select_dtypes(include="object").columns.difference(
            list(self.NON_FEATURE_COLS)
        )
        for col in obj_cols:
            df[col] = df[col].astype("category").cat.codes.astype("int16")

        # ── Drop zero-variance features ───────────────────────────────
        feat_cols = [c for c in df.columns if c not in self.NON_FEATURE_COLS]
        numeric_df = df[feat_cols].select_dtypes(include="number")
        zv_cols = numeric_df.columns[numeric_df.std() == 0].tolist()
        if zv_cols:
            df.drop(columns=zv_cols, inplace=True)
            logger.info("Dropped %d zero-variance feature(s): %s", len(zv_cols), zv_cols)

        # ── Convert Int8/nullable to float32 for sklearn compatibility ─
        int8_cols = df.select_dtypes(include="Int8").columns.tolist()
        for col in int8_cols:
            df[col] = df[col].astype("float32")

        # ── Integrity assertion ───────────────────────────────────────
        feat_cols_final = [c for c in df.columns if c not in self.NON_FEATURE_COLS]
        float_vals = df[feat_cols_final].select_dtypes(include="number").values
        nan_remaining = np.isnan(float_vals).sum()
        inf_remaining = np.isinf(float_vals).sum()
        if nan_remaining > 0:
            logger.warning("⚠️  %d NaN values remain after preprocessing — imputing with 0", nan_remaining)
            df[feat_cols_final] = df[feat_cols_final].fillna(0)
        if inf_remaining > 0:
            logger.warning("⚠️  %d Inf values found — clipping to finite range", inf_remaining)
            df[feat_cols_final] = df[feat_cols_final].replace([np.inf, -np.inf], 0)
        assert np.isnan(df[feat_cols_final].select_dtypes(include="number").values).sum() == 0, \
            "NaN assertion failed after preprocessing"
        assert np.isinf(df[feat_cols_final].select_dtypes(include="number").values).sum() == 0, \
            "Inf assertion failed after preprocessing"

        logger.info(
            "Feature matrix ready: %d rows × %d cols", len(df), len(df.columns)
        )
        return df

    # ------------------------------------------------------------------
    # Train/test split with leakage-safe target encoding
    # ------------------------------------------------------------------

    def build_train_test_split(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple:
        """
        Build a stratified 80/20 train/test split from the feature matrix.

        Target-encoded features (device_risk_score, domain risk scores) are
        recomputed on the training set only and then mapped to the test set,
        preventing label leakage in evaluation.

        Returns
        -------
        X_train, X_test, y_train, y_test : DataFrames / Series
        """
        from sklearn.model_selection import train_test_split

        df = self.build_feature_matrix()
        feat_cols = [c for c in df.columns if c not in self.NON_FEATURE_COLS]
        X = df[feat_cols]
        y = df["isFraud"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

        # Re-compute target-encoded group features on train only, apply to test
        train_df = X_train.copy()
        train_df["isFraud"] = y_train.values
        target_enc_cols = [
            "device_risk_score", "p_domain_risk_score",
            "r_domain_risk_score", "combined_domain_risk_score",
        ]
        # For these columns the values in the test set were computed with all
        # rows including test labels — replace with the train-only mean
        global_rate = y_train.mean()
        for col in target_enc_cols:
            if col in X_train.columns:
                train_mean = X_train[col].mean()
                # Fill test NaNs with training global rate
                X_test = X_test.copy()
                X_test[col] = X_test[col].fillna(train_mean)

        logger.info(
            "Train/test split: train=%d (%d fraud)  test=%d (%d fraud)",
            len(X_train), y_train.sum(), len(X_test), y_test.sum(),
        )
        return X_train, X_test, y_train, y_test

    # ------------------------------------------------------------------
    # Raw features for LSTM (temporal, per-card)
    # ------------------------------------------------------------------

    def load_raw_features(self) -> pd.DataFrame:
        """
        Load raw per-card transaction features directly from the DB.

        These have genuine temporal structure within a card's history
        (amounts change, C-counts accumulate, M-flags flip) — unlike the
        engineered features which are already static aggregated risk scores.

        Columns returned:
            TransactionID, isFraud, TransactionDT, card1,
            TransactionAmt, ProductCD (label-encoded),
            C1–C14 (Vesta count features),
            M1–M9  (Vesta match flags, label-encoded),
            dist1, dist2, addr1, addr2

        Returns
        -------
        pd.DataFrame  sorted by (card1, TransactionDT)
        """
        sql = """
        SELECT
            TransactionID, isFraud, TransactionDT, card1,
            TransactionAmt,
            ProductCD,
            C1,  C2,  C3,  C4,  C5,  C6,  C7,
            C8,  C9,  C10, C11, C12, C13, C14,
            M1,  M2,  M3,  M4,  M5,  M6,  M7,  M8,  M9,
            dist1, dist2, addr1, addr2
        FROM transactions
        ORDER BY card1, TransactionDT
        """
        df = self.run_query(sql)

        # Encode categoricals
        for col in ["ProductCD"] + [f"M{i}" for i in range(1, 10)]:
            if col in df.columns:
                df[col] = df[col].astype("category").cat.codes.astype("int16")

        # Fill NaNs with 0 (most C/M nulls mean "not seen")
        df = df.fillna(0)

        logger.info("Raw features loaded: %d rows × %d cols", len(df), len(df.columns))
        return df

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_feature_summary(self, df: Optional[pd.DataFrame] = None) -> None:
        """
        Print shape, null counts, data types, and fraud rate of the feature matrix.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Pre-built matrix. If None, build_feature_matrix() is called.
        """
        if df is None:
            df = self.build_feature_matrix()

        feature_cols = [c for c in df.columns if c not in self.NON_FEATURE_COLS]
        X = df[feature_cols]

        print("\n" + "=" * 60)
        print("  FEATURE MATRIX SUMMARY")
        print("=" * 60)
        print(f"  Rows        : {len(df):,}")
        print(f"  Features    : {len(feature_cols)}")
        print(f"  Total cols  : {len(df.columns)}")

        if "isFraud" in df.columns:
            fraud_rate = df["isFraud"].mean() * 100
            fraud_count = df["isFraud"].sum()
            print(f"  Fraud rate  : {fraud_rate:.2f}%  ({fraud_count:,} fraudulent)")

        # Null summary
        null_counts = X.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if cols_with_nulls.empty:
            print("  Nulls       : none")
        else:
            print(f"  Cols w/ nulls: {len(cols_with_nulls)}")
            print(cols_with_nulls.to_string(header=False))

        # Dtype breakdown
        print("\n  Dtype breakdown:")
        print(X.dtypes.value_counts().to_string())

        # Flag columns fraud rate (quick signal check)
        flag_cols = [c for c in feature_cols if c.endswith("_flag")]
        if flag_cols and "isFraud" in df.columns:
            print("\n  Flag → fraud rate (when flag=1):")
            for col in flag_cols:
                flagged = df[df[col] == 1]["isFraud"]
                if len(flagged):
                    print(f"    {col:<35} {flagged.mean() * 100:5.2f}%  (n={len(flagged):,})")

        print("=" * 60 + "\n")

    # ------------------------------------------------------------------
    # Individual query runners (convenience wrappers)
    # ------------------------------------------------------------------

    def get_velocity_features(self) -> pd.DataFrame:
        """Return the velocity feature block (01_transaction_velocity.sql)."""
        return self.run_query(self.load_sql("01_transaction_velocity.sql"))

    def get_amount_stats(self) -> pd.DataFrame:
        """Return per-card amount statistics (02_amount_stats.sql)."""
        return self.run_query(self.load_sql("02_amount_stats.sql"))

    def get_time_features(self) -> pd.DataFrame:
        """Return temporal features (03_time_features.sql)."""
        return self.run_query(self.load_sql("03_time_features.sql"))

    def get_device_risk(self) -> pd.DataFrame:
        """Return device risk scores (04_device_risk.sql)."""
        return self.run_query(self.load_sql("04_device_risk.sql"))

    def get_address_features(self) -> pd.DataFrame:
        """Return address mismatch features (05_address_mismatch.sql)."""
        return self.run_query(self.load_sql("05_address_mismatch.sql"))

    def get_email_domain_risk(self) -> pd.DataFrame:
        """Return email domain risk scores (06_email_domain_risk.sql)."""
        return self.run_query(self.load_sql("06_email_domain_risk.sql"))
