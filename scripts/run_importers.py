import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.data.importers.strategy_data_builder import DataBentoImporter
from src.env import Env


def run_mirror_import():
    print("From CSV >> Parquet ")

    importer = DataBentoImporter()
    source = "databento"

    out_dir = Env.path("prices", source)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = importer.fetch_all_as_mirror()

        save_path = out_dir / "prices.parquet"
        df.to_parquet(save_path)

        print(f"\n ✅Successfully imported {save_path}")
        print(f" Scale: {len(df)}  row")
        print(f" Head: {list(df.columns)}")

    except Exception as e:
        print(f"\n Fail to Transfer from Csv to Parquet: {e}")


if __name__ == "__main__":
    run_mirror_import()