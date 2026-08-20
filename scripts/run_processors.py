import sys
import os
from pathlib import Path
import os
import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.data.processors.data_lab_2 import DataLab2
from src.data.processors.data_lab_3 import DataLab3

def main():
    print("JPM Strategy Pipeline Automation Scheduler Starting...")

    pipeline = [
        {
            "name": "Basic Cleaning & Resampling",
            "lab_class": DataLab2,
            "target_file": "ES_Daily_Raw.parquet",
            "action": "run_comprehensive_cleaning"
        },
        {
            "name": "Continuous Contract & Back-Adjustment",
            "lab_class": DataLab3,
            "target_file": "ES_Daily_Processed.parquet",
            "action": "build_continuous_strategy_data"
        },
    ]

    active_lab_class = pipeline[-1]["lab_class"]
    handler = active_lab_class()

    for step in pipeline:
        file_path = f"{handler.processed_path}/{step['target_file']}"

        print(f"\n▶️ Checking Stage: {step['name']}")

        if os.path.exists(file_path):
            print(f"✅ Cache Hit: {step['target_file']} exists, skipping.")
        else:
            print(f"🚧 Cache Miss: Executing {step['action']}...")
            func = getattr(handler, step['action'])
            func()
            print(f"✨ Stage Complete: {step['target_file']} generated.")

    print("\n" + "=" * 40)
    print("🎉 Success! All pipeline stages are up to date.")

if __name__ == "__main__":
    main()