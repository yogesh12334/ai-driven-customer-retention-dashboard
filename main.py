import sqlite3
import pandas as pd
import logging
import os
import json
import time
import sys
from datetime import datetime
from dataclasses import dataclass, field
from functools import wraps
from typing import Optional, Dict, List
from pathlib import Path

# ============================================================
#  CONFIG
# ============================================================
@dataclass
class PipelineConfig:
    db_path: str = "database.db"
    csv_path: str = "data/customers.csv"
    log_dir: str = "logs"
    report_dir: str = "reports"
    max_retries: int = 3
    retry_delay: float = 1.5
    churn_alert_threshold: float = 30.0
    required_columns: List[str] = field(default_factory=lambda: [
        "CustomerID", "Churn", "MonthlyCharges", "Tenure", "ContractType"
    ])

# ============================================================
#  UTILITIES
# ============================================================
def pipeline_stage(stage_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("DataPipeline")
            logger.info(f">>> STARTING: {stage_name}")
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.info(f"<<< FINISHED: {stage_name} ({elapsed:.2f}s)")
                return result
            except Exception as e:
                logger.error(f"!!! FAILED: {stage_name} | Error: {str(e)}")
                raise
        return wrapper
    return decorator

# ============================================================
#  CHURN INTELLIGENCE PIPELINE
# ============================================================
class ChurnPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # WINDOWS FIX: Force UTF-8 and handle terminal limits
        if sys.platform == "win32":
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
            
        self.logger = self._setup_logging()
        self.conn: Optional[sqlite3.Connection] = None
        self.results: Dict[str, pd.DataFrame] = {}
        
        # PANDAS FIX: Sab numbers readable honge, scientific notation (e+09) nahi dikhega
        pd.options.display.float_format = "{:,.2f}".format

    def _setup_logging(self) -> logging.Logger:
        Path(self.config.log_dir).mkdir(exist_ok=True)
        log_file = Path(self.config.log_dir) / f"churn_{datetime.now().strftime('%Y%m%d')}.log"
        
        logger = logging.getLogger("DataPipeline")
        logger.setLevel(logging.DEBUG)
        
        fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S")
        
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(logging.INFO)
        
        if not logger.handlers:
            logger.addHandler(fh)
            logger.addHandler(ch)
        return logger

    @pipeline_stage("Database Connection")
    def connect_db(self):
        self.conn = sqlite3.connect(self.config.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")

    @pipeline_stage("Extraction")
    def extract(self) -> pd.DataFrame:
        if not Path(self.config.csv_path).exists():
            raise FileNotFoundError(f"Missing CSV: {self.config.csv_path}")
        return pd.read_csv(self.config.csv_path)

    @pipeline_stage("Transformation & Intelligence")
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates().copy()
        
        # Business Intelligence: Lifetime Value
        df["LifetimeValue"] = df["MonthlyCharges"] * df["Tenure"].clip(lower=1)
        
        # Intelligence: High-Value Churn Risk
        high_threshold = df["MonthlyCharges"].quantile(0.80)
        df["IsCriticalRisk"] = ((df["Churn"] == 1) & (df["MonthlyCharges"] >= high_threshold)).astype(int)
        
        return df

    @pipeline_stage("Loading to SQL")
    def load(self, df: pd.DataFrame):
        df.to_sql("customers", self.conn, if_exists="replace", index=False)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_churn ON customers(Churn)")
        self.conn.commit()

    @pipeline_stage("Running Analytics Queries")
    def run_analytics(self):
        queries = {
            "churn_summary": """
                SELECT ROUND(AVG(Churn)*100, 2) as churn_pct, 
                       SUM(Churn) as churned_count, 
                       COUNT(*) as total 
                FROM customers
            """,
            "revenue_loss": """
                SELECT ROUND(SUM(MonthlyCharges * 6), 2) as loss_6m 
                FROM customers WHERE Churn = 1
            """,
            "critical_customers": """
                SELECT CustomerID, MonthlyCharges, Tenure, LifetimeValue 
                FROM customers WHERE IsCriticalRisk = 1 LIMIT 10
            """
        }
        for name, sql in queries.items():
            self.results[name] = pd.read_sql(sql, self.conn)

    def generate_dashboard(self):
        # Final Terminal Output
        print("\n" + "═"*60)
        print("   CHURN INTELLIGENCE FINAL DASHBOARD")
        print("═"*60)
        
        for name, df in self.results.items():
            print(f"\n[+] {name.upper().replace('_', ' ')}")
            print(df.to_string(index=False))

        # Alert Section (Unicode Safe - No Rupee Symbol)
        churn_rate = self.results["churn_summary"]["churn_pct"].iloc[0]
        if churn_rate > self.config.churn_alert_threshold:
            loss = self.results["revenue_loss"]["loss_6m"].iloc[0]
            alert_msg = f"CRITICAL: High Churn ({churn_rate}%). Est. 6-Month Loss: Rs. {loss:,.2f}"
            print(f"\n[ALERT] {alert_msg}")
            self.logger.warning(alert_msg)

    def start(self):
        try:
            self.connect_db()
            data = self.extract()
            data = self.transform(data)
            self.load(data)
            self.run_analytics()
            self.generate_dashboard()
        except Exception as e:
            self.logger.critical(f"Pipeline crashed: {e}", exc_info=True)
        finally:
            if self.conn: self.conn.close()

if __name__ == "__main__":
    # Ensure data folder exists
    Path("data").mkdir(exist_ok=True)
    
    # Execute Pipeline
    ChurnPipeline(PipelineConfig()).start()