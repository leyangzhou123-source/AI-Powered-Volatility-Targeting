

class DataLab1:
    def __init__(self, input_filename="prices.parquet"):
        self.processed_path = str(Env.path("processed"))
        self.raw_path = str(Env.path("prices", "databento"))
        self.input_path = Env.path("prices", "databento") / input_filename

        os.makedirs(self.processed_path, exist_ok=True)

    def load_raw_data(self):
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        print(f"📂 Loading raw data: {self.input_path}...")
        df = pd.read_parquet(self.input_path)
        return df.sort_index()

    def save_processed_data(self, df, filename):
        path = os.path.join(self.processed_path, filename)
        df.to_parquet(path)
        print(f"💾 File saved to: {path}")