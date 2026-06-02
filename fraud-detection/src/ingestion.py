"""
IEEE-CIS Fraud Detection dataset ingestion into SQLite.

Usage:
    python src/ingestion.py --data-dir data/ --db-path db/fraud.db

The IEEE-CIS dataset must be downloaded manually from Kaggle:
    https://www.kaggle.com/competitions/ieee-fraud-detection/data

Place the following files in the data directory before running:
    - train_transaction.csv
    - train_identity.csv
    - test_transaction.csv
    - test_identity.csv
"""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    inspect,
    text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class DataIngestion:
    """Loads IEEE-CIS Fraud Detection CSVs into a SQLite database."""

    TRANSACTION_COLS = {
        "TransactionID": Integer,
        "isFraud": Integer,
        "TransactionDT": Integer,
        "TransactionAmt": Float,
        "ProductCD": String(8),
        "card1": Integer,
        "card2": Float,
        "card3": Float,
        "card4": String(32),
        "card5": Float,
        "card6": String(32),
        "addr1": Float,
        "addr2": Float,
        "dist1": Float,
        "dist2": Float,
        "P_emaildomain": String(64),
        "R_emaildomain": String(64),
    }

    IDENTITY_COLS = {
        "TransactionID": Integer,
        "DeviceType": String(16),
        "DeviceInfo": String(128),
        "id_01": Float,
        "id_02": Float,
        "id_03": Float,
        "id_04": Float,
        "id_05": Float,
        "id_06": Float,
        "id_07": Float,
        "id_08": Float,
        "id_09": Float,
        "id_10": Float,
        "id_11": Float,
        "id_12": String(8),
        "id_13": Float,
        "id_14": Float,
        "id_15": String(8),
        "id_16": String(8),
        "id_17": Float,
        "id_18": Float,
        "id_19": Float,
        "id_20": Float,
        "id_21": Float,
        "id_22": Float,
        "id_23": String(8),
        "id_24": Float,
        "id_25": Float,
        "id_26": Float,
        "id_27": String(8),
        "id_28": String(8),
        "id_29": String(8),
        "id_30": String(64),
        "id_31": String(64),
        "id_32": Float,
        "id_33": String(16),
        "id_34": String(16),
        "id_35": String(8),
        "id_36": String(8),
        "id_37": String(8),
        "id_38": String(8),
    }

    def __init__(self, data_dir: str, db_path: str, chunk_size: int = 50_000):
        self.data_dir = Path(data_dir)
        self.db_path = Path(db_path)
        self.chunk_size = chunk_size
        self.engine = None
        self.metadata = MetaData()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        logger.info("Connected to database at %s", self.db_path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _build_v_columns(self, n: int) -> dict:
        """Return V1–Vn float columns (transaction V-features)."""
        return {f"V{i}": Float for i in range(1, n + 1)}

    def _build_c_m_columns(self) -> dict:
        cols = {}
        for i in range(1, 15):
            cols[f"C{i}"] = Float
        for i in range(1, 10):
            cols[f"M{i}"] = String(8)
        return cols

    def create_schema(self) -> None:
        transaction_col_defs = {**self.TRANSACTION_COLS,
                                **self._build_c_m_columns(),
                                **self._build_v_columns(339)}

        def _resolve(col_type):
            """Accept both type classes (Integer) and type instances (String(8))."""
            import inspect
            return col_type() if inspect.isclass(col_type) else col_type

        transaction_columns = [
            Column(name, _resolve(col_type), primary_key=(name == "TransactionID"))
            for name, col_type in transaction_col_defs.items()
        ]

        identity_columns = [
            Column(name, _resolve(col_type), primary_key=(name == "TransactionID"))
            for name, col_type in self.IDENTITY_COLS.items()
        ]

        Table("transactions", self.metadata, *transaction_columns, extend_existing=True)
        Table("identity", self.metadata, *identity_columns, extend_existing=True)

        self.metadata.create_all(self.engine)
        logger.info("Schema created: tables 'transactions' and 'identity'")

        # Create joined view for downstream feature engineering
        with self.engine.connect() as conn:
            conn.execute(text("DROP VIEW IF EXISTS fraud_combined"))
            conn.execute(text(
                "CREATE VIEW fraud_combined AS "
                "SELECT t.*, i.DeviceType, i.DeviceInfo, "
                "i.id_01, i.id_02, i.id_03, i.id_04, i.id_05, "
                "i.id_06, i.id_07, i.id_08, i.id_09, i.id_10, "
                "i.id_11, i.id_12, i.id_13, i.id_14, i.id_15, "
                "i.id_16, i.id_17, i.id_18, i.id_19, i.id_20 "
                "FROM transactions t "
                "LEFT JOIN identity i ON t.TransactionID = i.TransactionID"
            ))
            conn.commit()
        logger.info("View 'fraud_combined' created")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_csv_to_table(self, csv_path: Path, table_name: str) -> int:
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected data file not found: {csv_path}")

        inspector = inspect(self.engine)
        db_columns = {c["name"] for c in inspector.get_columns(table_name)}

        total_rows = 0
        for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=self.chunk_size, low_memory=False)):
            # Keep only columns that exist in the schema
            chunk = chunk[[c for c in chunk.columns if c in db_columns]]
            chunk.to_sql(table_name, con=self.engine, if_exists="append", index=False)
            total_rows += len(chunk)
            logger.info("  chunk %d loaded (%d rows so far)", i + 1, total_rows)

        return total_rows

    def load_transactions(self, split: str = "train") -> None:
        path = self.data_dir / f"{split}_transaction.csv"
        logger.info("Loading transactions from %s", path)
        n = self._load_csv_to_table(path, "transactions")
        logger.info("Loaded %d transaction rows", n)

    def load_identity(self, split: str = "train") -> None:
        path = self.data_dir / f"{split}_identity.csv"
        logger.info("Loading identity from %s", path)
        n = self._load_csv_to_table(path, "identity")
        logger.info("Loaded %d identity rows", n)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        with self.engine.connect() as conn:
            txn_count = conn.execute(text("SELECT COUNT(*) FROM transactions")).scalar()
            idn_count = conn.execute(text("SELECT COUNT(*) FROM identity")).scalar()
            fraud_rate = conn.execute(
                text("SELECT ROUND(AVG(isFraud) * 100, 2) FROM transactions WHERE isFraud IS NOT NULL")
            ).scalar()

        logger.info("Validation — transactions: %d | identity: %d | fraud rate: %s%%",
                    txn_count, idn_count, fraud_rate)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run(self, split: str = "train") -> None:
        self.connect()
        self.create_schema()
        self.load_transactions(split)
        self.load_identity(split)
        self.validate()
        logger.info("Ingestion complete.")


def main():
    parser = argparse.ArgumentParser(description="Ingest IEEE-CIS Fraud Detection data into SQLite")
    parser.add_argument("--data-dir", default="data/", help="Directory containing CSV files")
    parser.add_argument("--db-path", default="db/fraud.db", help="SQLite database path")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()

    ingestion = DataIngestion(
        data_dir=args.data_dir,
        db_path=args.db_path,
        chunk_size=args.chunk_size,
    )
    ingestion.run(split=args.split)


if __name__ == "__main__":
    main()
