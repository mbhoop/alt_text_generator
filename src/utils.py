import logging
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def load_dataset(csv_path: str) -> pd.DataFrame:
    """The single CSV loader shared by every script."""
    return pd.read_csv(csv_path)
