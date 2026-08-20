import os
import numpy as np
import pandas as pd
from src.env import Env


class DataBentoImporter:
    def __init__(self):
        self.api_key = os.getenv("DATABENTO_API_KEY")

        
        self.raw_data_path = Env.path("raw") / "prices.csv"

       
        self.mapping = {
            "timestamp": ["date", "Date", "timestamp", "Time", "DateTime", "ts_event"],
            "symbol": ["symbol", "ticker", "instrument_id", "code"],
            "close": ["close", "Close", "C"],
            "volume": ["volume", "Volume", "vol", "V"],
            "ret": ["ret", "return", "returns", "Ret"],
            "rv20": ["rv20", "RV20", "realized_vol_20d"],
        }

    def _smart_rename(self, df: pd.DataFrame) -> pd.DataFrame:
        rename_dict = {}
        for std_name, variants in self.mapping.items():
            for v in variants:
                if v in df.columns:
                    rename_dict[v] = std_name
                    break
        return df.rename(columns=rename_dict)

    def fetch_all_as_mirror(self) -> pd.DataFrame:
        
        if self.api_key:
            print("Use API Key (fallback to local in this project)")
            return self._fetch_from_api()
        else:
            print(f"No API Key, From Local: {self.raw_data_path}")
            return self._fetch_from_local_csv()

    def _fetch_from_local_csv(self) -> pd.DataFrame:
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Fail: No Local File {self.raw_data_path}")

        df = pd.read_csv(self.raw_data_path, header=0, low_memory=False)
        df = self._smart_rename(df)

        
        if "timestamp" not in df.columns:
            raise ValueError("Your CSV must have a date column (e.g., 'date'/'Date')")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

        return df

    def _fetch_from_api(self) -> pd.DataFrame:
        try:
            import src.data.importers.strategy_data_builder as db  # noqa: F401
            print("API not used here; fallback to local")
            return self._fetch_from_local_csv()
        except ImportError:
            print("No databento; fallback to local")
            return self._fetch_from_local_csv()


    def save_processed_parquet(
        self,
        out_name: str = "ES_Daily_Processed.parquet",
        use_log_return: bool = True,
        winsor_q: float = 0.01,
    ) -> pd.DataFrame:
        """
        Output columns required by backtest engine:
          - returns
          - returns_clean
          - is_roll_day
        """
        df = self.fetch_all_as_mirror()

        
        if "ret" in df.columns:
            ret_simple = pd.to_numeric(df["ret"], errors="coerce")
        elif "close" in df.columns:
            close = pd.to_numeric(df["close"], errors="coerce")
            ret_simple = close.pct_change()
        else:
            raise ValueError("Need 'ret' or 'close' in CSV")

        
        returns = np.log1p(ret_simple) if use_log_return else ret_simple

        
        q_low = returns.quantile(winsor_q)
        q_high = returns.quantile(1 - winsor_q)
        returns_clean = returns.clip(lower=q_low, upper=q_high)

        out = pd.DataFrame(
            {
                "returns": returns,
                "returns_clean": returns_clean,
                "is_roll_day": False,
            },
            index=df.index,
        ).dropna()

        out_path = Env.path("processed") / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path)

        print(f"Saved processed dataset: {out_path} | rows={len(out)}")
        return out
