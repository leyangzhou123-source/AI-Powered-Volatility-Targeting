import pandas as pd
from pathlib import Path


class DataInterface:
    @staticmethod
    def prices(symbol: str, source: str = "local_csv", start_date=None, end_date=None) -> pd.DataFrame:
        """
        Load prices for `symbol` and return a DataFrame indexed by date with a `close` column.
        Expected CSV: data/raw/prices.csv with columns including:
          - ts_event (or date/datetime/time)
          - symbol (or ticker)
          - close (or adj_close)
        """
        if source != "local_csv":
            raise ValueError(f"Unknown data source: {source}")

        path = Path("data/raw/prices.csv")
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path.resolve()}")

        df = pd.read_csv(path)

        # map columns (case-insensitive)
        lower_map = {c.lower(): c for c in df.columns}

        ts_col = lower_map.get("ts_event") or lower_map.get("datetime") or lower_map.get("date") or lower_map.get("time")
        sym_col = lower_map.get("symbol") or lower_map.get("ticker")
        close_col = lower_map.get("close") or lower_map.get("adj_close") or lower_map.get("adj close")

        if ts_col is None or sym_col is None or close_col is None:
            raise ValueError(
                "CSV must contain timestamp, symbol, close columns. "
                f"Found columns: {list(df.columns)}"
            )

        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col])

        df = df[df[sym_col] == symbol].copy()
        if df.empty:
            raise ValueError(f"No rows found for symbol='{symbol}' in {path}")

        df = df.set_index(ts_col).sort_index()

        # intraday -> daily close
        daily = (
            df[close_col]
            .resample("1D")
            .last()
            .dropna()
            .to_frame(name="close")
        )

        if start_date is not None:
            daily = daily.loc[pd.to_datetime(start_date):]
        if end_date is not None:
            daily = daily.loc[:pd.to_datetime(end_date)]

        return daily
