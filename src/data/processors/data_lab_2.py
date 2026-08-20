import pandas as pd
from src.data.processors.data_lab_1 import DataLab1


class DataLab2(DataLab1):
    def __init__(self):
        super().__init__()

    def run_comprehensive_cleaning(self):
        print("=" * 50)
        print("🚀 Data Lab 2: Starting Multi-Period Refinement Process...")

        df = self.load_raw_data()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        initial_count = len(df)
        df = df[~df.index.duplicated(keep='first')]
        df = df[df['close'] > 0]
        print(f"  - Static Cleaning Complete: Removed {initial_count - len(df)} invalid records")

        if 'symbol' in df.columns:
            es_min_df = df[df['symbol'].str.contains('ES', na=False)].copy()
        else:
            es_min_df = df.copy()

        self.save_processed_data(es_min_df, "ES_minute_Raw.parquet")
        print(f"  - Exported Minutely Data: ES_minute_Raw.parquet ({len(es_min_df)} rows)")

        es_daily_df = es_min_df.resample('B').last().dropna(subset=['close'])

        self.save_processed_data(es_daily_df, "ES_Daily_Raw.parquet")
        print(f"  - Exported Daily Data: ES_Daily_Raw.parquet ({len(es_daily_df)} rows)")

        return es_min_df, es_daily_df


if __name__ == "__main__":
    lab = DataLab2()
    min_df, daily_df = lab.run_comprehensive_cleaning()

    print("\n" + "=" * 50)
    print("📊 Data Summary")
    print("-" * 50)

    for name, df in [("ES_minute_Raw", min_df), ("ES_Daily_Raw", daily_df)]:
        print(f"File Name: {name}.parquet")
        print(f"Rows: {len(df):,}")
        print(f"Start Time: {df.index.min()}")
        print(f"End Time: {df.index.max()}")
        print(f"Columns: {list(df.columns)}")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        print("-" * 50)


        self.save_processed_data(df, "ES_minute_Raw.parquet")
        es_daily_df = df.resample('B').last().dropna(subset=['close'])
        self.save_processed_data(es_daily_df, "ES_Daily_Raw.parquet")