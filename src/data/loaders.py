from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "master.xlsx"

def load_master_dataframe(sheet_name: str = "experimental") -> pd.DataFrame:
    """Load the thesis master dataset from the repository."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    return pd.read_excel(DATA_PATH, sheet_name=sheet_name)
