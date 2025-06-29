import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
import re

class DataCleaner:
    """
    A comprehensive data cleaning class that handles common data preprocessing tasks.
    
    Features:
    - Remove null values and duplicates
    - Convert data types (especially string timestamps to datetime)
    - Trim whitespace in text fields
    - Configurable cleaning pipeline
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the DataCleaner with a DataFrame and optional configuration.
        
        Args:
            df (pd.DataFrame): The DataFrame to clean
            config (dict, optional): Configuration dictionary for cleaning parameters
        """
        self.original_df = df.copy()
        self.df = df.copy()
        self.config = config or {}
        self.cleaning_log = []
        
        # Default configuration
        self.default_config = {
            'remove_nulls': True,
            'remove_duplicates': True,
            'trim_whitespace': True,
            'convert_datatypes': True,
            'datetime_columns': [],  # Columns to convert to datetime
            'numeric_columns': [],   # Columns to convert to numeric
            'categorical_columns': [], # Columns to convert to category
            'keep_first_duplicate': True,  # Keep first occurrence of duplicates
            'null_threshold': 0.5,   # Remove columns with > 50% nulls
            'infer_datetime_format': True
        }
        
        # Merge user config with defaults
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value
    
    def _log_action(self, action: str, details: str = ""):
        """Log cleaning actions for tracking."""
        log_entry = f"{action}: {details}"
        self.cleaning_log.append(log_entry)
        logging.info(log_entry)
    
    def remove_null_values(self, threshold: Optional[float] = None) -> 'DataCleaner':
        """
        Remove null values based on specified strategy.
        
        Args:
            threshold (float, optional): Remove columns with null percentage above threshold
            
        Returns:
            DataCleaner: Self for method chaining
        """
        if not self.config['remove_nulls']:
            return self
        
        initial_shape = self.df.shape
        threshold = threshold or self.config['null_threshold']
        
        # Remove columns with high null percentage
        null_percentages = self.df.isnull().sum() / len(self.df)
        cols_to_drop = null_percentages[null_percentages > threshold].index.tolist()
        
        if cols_to_drop:
            self.df = self.df.drop(columns=cols_to_drop)
            self._log_action("Removed high-null columns", f"{cols_to_drop}")
        
        # Remove rows with any null values in remaining columns
        self.df = self.df.dropna()
        
        final_shape = self.df.shape
        self._log_action("Removed null values", 
                        f"Shape: {initial_shape} → {final_shape}")
        
        return self
    
    def remove_duplicates(self, keep: str = 'first') -> 'DataCleaner':
        """
        Remove duplicate rows.
        
        Args:
            keep (str): Which duplicate to keep ('first', 'last', False)
            
        Returns:
            DataCleaner: Self for method chaining
        """
        if not self.config['remove_duplicates']:
            return self
        
        initial_count = len(self.df)
        keep_option = 'first' if self.config['keep_first_duplicate'] else keep
        
        self.df = self.df.drop_duplicates(keep=keep_option)
        
        final_count = len(self.df)
        duplicates_removed = initial_count - final_count
        
        self._log_action("Removed duplicates", 
                        f"{duplicates_removed} duplicates removed")
        
        return self
    
    def trim_whitespace(self) -> 'DataCleaner':
        """
        Trim whitespace from string columns.
        
        Returns:
            DataCleaner: Self for method chaining
        """
        if not self.config['trim_whitespace']:
            return self
        
        string_columns = self.df.select_dtypes(include=['object']).columns
        trimmed_columns = []
        
        for col in string_columns:
            if self.df[col].dtype == 'object':
                # Check if column contains strings
                sample_non_null = self.df[col].dropna().iloc[0] if not self.df[col].dropna().empty else None
                if isinstance(sample_non_null, str):
                    self.df[col] = self.df[col].astype(str).str.strip()
                    trimmed_columns.append(col)
        
        if trimmed_columns:
            self._log_action("Trimmed whitespace", f"Columns: {trimmed_columns}")
        
        return self
    
    def convert_data_types(self) -> 'DataCleaner':
        """
        Convert data types based on configuration and automatic detection.
        
        Returns:
            DataCleaner: Self for method chaining
        """
        if not self.config['convert_datatypes']:
            return self
        
        converted_columns = []
        
        # Convert specified datetime columns
        datetime_cols = self.config.get('datetime_columns', [])
        for col in datetime_cols:
            if col in self.df.columns:
                self.df[col] = self._convert_to_datetime(col)
                converted_columns.append(f"{col} → datetime")
        
        # Convert specified numeric columns
        numeric_cols = self.config.get('numeric_columns', [])
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = self._convert_to_numeric(col)
                converted_columns.append(f"{col} → numeric")
        
        # Convert specified categorical columns
        categorical_cols = self.config.get('categorical_columns', [])
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
                converted_columns.append(f"{col} → category")
        
        # Auto-detect and convert datetime columns
        self._auto_detect_datetime_columns()
        
        # Auto-detect and convert numeric columns
        self._auto_detect_numeric_columns()
        
        if converted_columns:
            self._log_action("Converted data types", f"{converted_columns}")
        
        return self
    
    def _convert_to_datetime(self, column: str) -> pd.Series:
        """Convert a column to datetime with error handling."""
        try:
            if self.config['infer_datetime_format']:
                return pd.to_datetime(self.df[column], infer_datetime_format=True, errors='coerce')
            else:
                return pd.to_datetime(self.df[column], errors='coerce')
        except Exception as e:
            self._log_action("DateTime conversion failed", f"{column}: {str(e)}")
            return self.df[column]
    
    def _convert_to_numeric(self, column: str) -> pd.Series:
        """Convert a column to numeric with error handling."""
        try:
            return pd.to_numeric(self.df[column], errors='coerce')
        except Exception as e:
            self._log_action("Numeric conversion failed", f"{column}: {str(e)}")
            return self.df[column]
    
    def _auto_detect_datetime_columns(self):
        """Automatically detect and convert datetime columns."""
        object_columns = self.df.select_dtypes(include=['object']).columns
        
        for col in object_columns:
            sample_values = self.df[col].dropna().head(10)
            if len(sample_values) == 0:
                continue
            
            # Check if values look like timestamps
            datetime_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            ]
            
            is_datetime = False
            for pattern in datetime_patterns:
                if sample_values.astype(str).str.contains(pattern, regex=True).any():
                    is_datetime = True
                    break
            
            if is_datetime:
                try:
                    converted = pd.to_datetime(self.df[col], errors='coerce', infer_datetime_format=True)
                    # If more than 50% of values were successfully converted, keep the conversion
                    if converted.notna().sum() / len(converted) > 0.5:
                        self.df[col] = converted
                        self._log_action("Auto-detected datetime", f"Column: {col}")
                except:
                    pass
    
    def _auto_detect_numeric_columns(self):
        """Automatically detect and convert numeric columns stored as strings."""
        object_columns = self.df.select_dtypes(include=['object']).columns
        
        for col in object_columns:
            # Skip if already processed as datetime
            if self.df[col].dtype == 'datetime64[ns]':
                continue
            
            sample_values = self.df[col].dropna().head(10)
            if len(sample_values) == 0:
                continue
            
            # Check if values look numeric
            numeric_pattern = r'^-?\d*\.?\d+$'
            if sample_values.astype(str).str.contains(numeric_pattern, regex=True).all():
                try:
                    converted = pd.to_numeric(self.df[col], errors='coerce')
                    # If more than 80% of values were successfully converted, keep the conversion
                    if converted.notna().sum() / len(converted) > 0.8:
                        self.df[col] = converted
                        self._log_action("Auto-detected numeric", f"Column: {col}")
                except:
                    pass
    
    def clean(self) -> pd.DataFrame:
        """
        Execute the complete cleaning pipeline.
        
        Returns:
            pd.DataFrame: The cleaned DataFrame
        """
        self._log_action("Starting data cleaning", f"Initial shape: {self.df.shape}")
        
        # Execute cleaning steps in order
        (self
         .trim_whitespace()
         .convert_data_types()
         .remove_null_values()
         .remove_duplicates())
        
        self._log_action("Cleaning completed", f"Final shape: {self.df.shape}")
        
        return self.df
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """
        Get a summary of cleaning operations performed.
        
        Returns:
            dict: Summary of cleaning operations
        """
        return {
            'original_shape': self.original_df.shape,
            'final_shape': self.df.shape,
            'rows_removed': self.original_df.shape[0] - self.df.shape[0],
            'columns_removed': self.original_df.shape[1] - self.df.shape[1],
            'cleaning_log': self.cleaning_log,
            'data_types_before': dict(self.original_df.dtypes),
            'data_types_after': dict(self.df.dtypes)
        }
    
    def reset(self):
        """Reset the DataFrame to its original state."""
        self.df = self.original_df.copy()
        self.cleaning_log = []
        self._log_action("Reset to original state", "")


# Example usage and utility functions
def load_and_clean_dataset(file_path: str, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Load dataset from JSON file and clean it.
    
    Args:
        file_path (str): Path to the dataset file
        config (dict, optional): Cleaning configuration
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    try:
        # Load the dataset
        df = pd.read_json(file_path)
        
        # Initialize cleaner with config
        cleaner = DataCleaner(df, config)
        
        # Clean the data
        cleaned_df = cleaner.clean()
        
        # Print summary
        summary = cleaner.get_cleaning_summary()
        print(f"Dataset cleaning completed:")
        print(f"Original shape: {summary['original_shape']}")
        print(f"Final shape: {summary['final_shape']}")
        print(f"Rows removed: {summary['rows_removed']}")
        print(f"Columns removed: {summary['columns_removed']}")
        
        return cleaned_df
        
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return None


if __name__ == "__main__":
    # Example usage with the specified dataset
    dataset_path = "data/BiztelAI_DS_Dataset_V1.json"
    
    # Custom configuration example
    cleaning_config = {
        'remove_nulls': True,
        'remove_duplicates': True,
        'trim_whitespace': True,
        'convert_datatypes': True,
        'null_threshold': 0.3,  # Remove columns with >30% nulls
        'datetime_columns': [],  # Specify datetime columns if known
        'numeric_columns': [],   # Specify numeric columns if known
    }
    
    # Load and clean the dataset
    cleaned_data = load_and_clean_dataset(dataset_path, cleaning_config)
    
    if cleaned_data is not None:
        print("\nData types after cleaning:")
        print(cleaned_data.dtypes)
        print("\nFirst few rows:")
        print(cleaned_data.head())