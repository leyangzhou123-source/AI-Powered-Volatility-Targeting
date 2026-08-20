import os
import numpy as np
import pandas as pd
from src.data.processors.data_lab_2 import DataLab2

class DataLab3(DataLab2):
    def __init__(self):
        super().__init__()

    def build_continuous_strategy_data(self):
        print("=" * 50)
        print("🚀 Data Lab 3: Starting Strategy-Level Data Processing...")

        input_file = os.path.join(self.processed_path, "ES_Daily_Raw.parquet")
        df = pd.read_parquet(input_file)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df['is_roll_day'] = False
        df['price_diff'] = 0.0

        if 'symbol' in df.columns:
            roll_mask = df['symbol'] != df['symbol'].shift(-1)
            df.loc[roll_mask, 'is_roll_day'] = True

            df.loc[roll_mask, 'price_diff'] = df['close'].shift(-1) - df['close']
            df['adjustment'] = df.loc[::-1, 'price_diff'].shift(1).fillna(0).cumsum()[::-1]
            df['close_adj'] = df['close'] + df['adjustment']
            print(f"  - Roll Processing Complete: Marked {df['is_roll_day'].sum()} cost deduction points.")
        else:
            df['close_adj'] = df['close']
            print("  - [Warning] Symbol column not detected, skipping back-adjustment and cost marking.")

        full_calendar = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        df = df.reindex(full_calendar)

        df['close'] = df['close'].ffill()
        df['close_adj'] = df['close_adj'].ffill()
        df['is_roll_day'] = df['is_roll_day'].fillna(False)
        if 'symbol' in df.columns:
            df['symbol'] = df['symbol'].ffill()

        df['returns'] = np.log(df['close_adj'] / df['close_adj'].shift(1))

        rolling_std = df['returns'].rolling(window=252, min_periods=20).std()
        df['upper_bound'] = 5 * rolling_std
        df['lower_bound'] = -5 * rolling_std

        df['returns_clean'] = df['returns'].combine(df['upper_bound'], min)
        df['returns_clean'] = df['returns_clean'].combine(df['lower_bound'], max)

        df = df.drop(columns=['upper_bound', 'lower_bound', 'price_diff', 'adjustment'])
        print(f"  - Dynamic Winsorization Complete: Protecting capital from extreme outliers.")

        final_filename = "ES_Daily_Processed.parquet"
        self.save_processed_data(df, final_filename)
        print(f"  - Strategy Truth File Generated: {final_filename}")

        return df

if __name__ == "__main__":
    lab = DataLab3()
    final_df = lab.build_continuous_strategy_data()