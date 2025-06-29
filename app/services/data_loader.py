import pandas as pd
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
import re
import warnings

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.logger import logger
from core.performance import timeit

warnings.filterwarnings("ignore", message=".*match groups.*")
    
class DataLoader:
    def __init__(self, file_path: str):
        if not file_path:
            raise ValueError("File path cannot be empty or None")

        self.file_path = Path(file_path)
        self.df: Optional[pd.DataFrame] = None
        self.raw_data: Optional[Union[dict, list]] = None

        if not self.file_path.exists():
            logger.error(f"❌ File not found: {self.file_path}")
            raise FileNotFoundError(f"{self.file_path} not found")

        if self.file_path.suffix.lower() != '.json':
            logger.warning(f"⚠️ Unexpected file extension: {self.file_path.suffix}")

        logger.info(f"📂 DataLoader initialized for file: {self.file_path}")

    @timeit
    def load_data(self, orient: str = 'records', **kwargs) -> pd.DataFrame:
        try:
            logger.info(f"📥 Loading data from: {self.file_path}")
            with open(self.file_path, 'r', encoding='utf-8') as file:
                self.raw_data = json.load(file)

            logger.info("🧪 JSON loaded, converting to DataFrame...")
            self.df = self._json_to_dataframe(self.raw_data, orient, **kwargs)
            logger.info(f"✅ DataFrame shape: {self.df.shape}")
            return self.df

        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise

    def _json_to_dataframe(self, data: Union[dict, list], orient: str, **kwargs) -> pd.DataFrame:
        try:
            if isinstance(data, list):
                if orient == 'records' or not orient:
                    if data and isinstance(data[0], dict):
                        return pd.json_normalize(data, **kwargs)
                    else:
                        return pd.DataFrame(data, **kwargs)
                else:
                    return pd.DataFrame(data, **kwargs)
            elif isinstance(data, dict):
                for key in ['data', 'records', 'items', 'results']:
                    if key in data and isinstance(data[key], list):
                        logger.info(f"📦 Found list under key: {key}")
                        return pd.json_normalize(data[key], **kwargs)
                return pd.json_normalize([data], **kwargs)
            else:
                raise ValueError(f"Unsupported JSON type: {type(data)}")

        except Exception as e:
            logger.error(f"❌ Error converting to DataFrame: {e}")
            raise

    @timeit
    def get_json_structure_info(self) -> Dict[str, Any]:
        if self.raw_data is None:
            logger.error("❌ No raw data loaded yet.")
            raise ValueError("No data loaded.")

        info = {
            'type': type(self.raw_data).__name__,
            'size': len(self.raw_data) if hasattr(self.raw_data, '__len__') else 'N/A'
        }

        if isinstance(self.raw_data, dict):
            info['keys'] = list(self.raw_data.keys())
        elif isinstance(self.raw_data, list) and self.raw_data:
            info['first_item_type'] = type(self.raw_data[0]).__name__
            if isinstance(self.raw_data[0], dict):
                info['first_item_keys'] = list(self.raw_data[0].keys())

        return info

    @timeit
    def get_basic_info(self) -> Dict[str, Any]:
        if self.df is None:
            logger.error("❌ Data not loaded yet.")
            raise ValueError("Call load_data() first.")

        file_size_mb = self.file_path.stat().st_size / (1024 * 1024)
        memory_usage_mb = self.df.memory_usage(deep=True).sum() / (1024 * 1024)
        null_counts = self.df.isnull().sum().to_dict()

        info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.apply(str).to_dict(),
            'null_counts': null_counts,
            'total_nulls': sum(null_counts.values()),
            'memory_usage_mb': round(memory_usage_mb, 2),
            'file_size_mb': round(file_size_mb, 2)
        }

        logger.info("📊 Basic info generated.")
        return info

    def get_dataframe(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Call load_data() first.")
        return self.df

    def get_sample(self, n: int = 5, random_state: Optional[int] = None) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Call load_data() first.")
        if n <= 0:
            raise ValueError("Sample size must be positive.")
        if n >= len(self.df):
            logger.warning("⚠️ Sample size >= dataset size, returning full DataFrame")
            return self.df
        return self.df.sample(n=n, random_state=random_state)
if __name__ == "__main__":
    loader = DataLoader("data/BiztelAI_DS_Dataset_V1.json")
    df = loader.load_data()
    info = loader.get_basic_info()
    print(info)
