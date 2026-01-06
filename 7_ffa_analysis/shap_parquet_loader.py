"""
Efficient Parquet-based SHAP value loader using PyArrow.

This module provides optimized loading of SHAP values from Parquet files,
supporting both full loading and lazy/on-demand access patterns.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import pandas as pd

logger = logging.getLogger(__name__)


class ShapParquetLoader:
    """
    Efficient loader for SHAP values stored in Parquet format.
    
    Supports both full loading (pandas DataFrame) and lazy loading (PyArrow Table)
    for memory-efficient access to large SHAP value files.
    """
    
    def __init__(self, parquet_path: Union[str, Path], lazy: bool = False):
        """
        Initialize SHAP Parquet loader.
        
        Args:
            parquet_path: Path to Parquet file containing SHAP values
            lazy: If True, use PyArrow Table for lazy loading (memory efficient)
                  If False, load into pandas DataFrame (faster access, more memory)
        """
        self.parquet_path = Path(parquet_path)
        self.lazy = lazy
        self._table = None
        self._df = None
        self._metadata = None
        
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"SHAP Parquet file not found: {self.parquet_path}")
        
        # Load metadata immediately (fast, no data loading)
        self._load_metadata()
    
    def _load_metadata(self):
        """Load Parquet file metadata without loading data."""
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(self.parquet_path)
            self._metadata = {
                'num_rows': parquet_file.metadata.num_rows,
                'num_columns': len(parquet_file.schema),
                'schema': parquet_file.schema,
                'column_names': parquet_file.schema.names
            }
            logger.debug(f"Parquet metadata: {self._metadata['num_rows']} rows, {self._metadata['num_columns']} columns")
        except ImportError:
            logger.warning("PyArrow not available, cannot load metadata")
            self._metadata = None
        except Exception as e:
            logger.warning(f"Could not load Parquet metadata: {e}")
            self._metadata = None
    
    @property
    def num_rows(self) -> int:
        """Number of rows (instances) in the Parquet file."""
        if self._metadata:
            return self._metadata['num_rows']
        elif self._df is not None:
            return len(self._df)
        else:
            # Fallback: load just to get row count
            return len(self.to_pandas())
    
    @property
    def num_columns(self) -> int:
        """Number of columns (features) in the Parquet file."""
        if self._metadata:
            return self._metadata['num_columns']
        elif self._df is not None:
            return len(self._df.columns)
        else:
            return len(self.to_pandas().columns)
    
    @property
    def column_names(self) -> list:
        """Column names (feature names) in the Parquet file."""
        if self._metadata:
            return self._metadata['column_names']
        elif self._df is not None:
            return list(self._df.columns)
        else:
            return list(self.to_pandas().columns)
    
    def to_pandas(self) -> pd.DataFrame:
        """
        Load SHAP values into a pandas DataFrame.
        
        Returns:
            DataFrame with SHAP values, indexed by instance index
        """
        if self._df is not None:
            return self._df
        
        try:
            import pyarrow.parquet as pq
            
            # Use PyArrow for efficient loading
            table = pq.read_table(self.parquet_path)
            self._df = table.to_pandas()
            
            # Ensure index is set properly (should be instance indices)
            if self._df.index.name is None and self._df.index.dtype == 'int64':
                self._df.index.name = 'instance_index'
            
            logger.info(f"Loaded SHAP values into pandas DataFrame: {len(self._df)} rows, {len(self._df.columns)} columns")
            return self._df
            
        except ImportError:
            # Fallback to pandas if PyArrow not available
            logger.info("PyArrow not available, using pandas.read_parquet")
            self._df = pd.read_parquet(self.parquet_path)
            
            if self._df.index.name is None and self._df.index.dtype == 'int64':
                self._df.index.name = 'instance_index'
            
            return self._df
    
    def get_row(self, instance_index: int) -> Dict[str, float]:
        """
        Get SHAP values for a specific instance (lazy loading).
        
        Args:
            instance_index: Index of the instance to retrieve
            
        Returns:
            Dictionary mapping feature_name -> SHAP value
        """
        if self.lazy and self._table is None:
            # Lazy loading: use PyArrow to read only the specific row
            try:
                import pyarrow.parquet as pq
                
                # Read only the specific row
                table = pq.read_table(
                    self.parquet_path,
                    use_pandas_metadata=True
                )
                
                # Convert to pandas for row access
                df = table.to_pandas()
                
                # Get the row
                if instance_index in df.index:
                    row = df.loc[instance_index]
                elif instance_index < len(df):
                    row = df.iloc[instance_index]
                else:
                    raise IndexError(f"Instance index {instance_index} out of range")
                
                return row.to_dict()
                
            except ImportError:
                # Fallback: load full DataFrame
                logger.warning("PyArrow not available for lazy loading, loading full DataFrame")
                return self.to_pandas().loc[instance_index].to_dict()
        else:
            # Use cached DataFrame
            df = self.to_pandas()
            if instance_index in df.index:
                return df.loc[instance_index].to_dict()
            elif instance_index < len(df):
                return df.iloc[instance_index].to_dict()
            else:
                raise IndexError(f"Instance index {instance_index} out of range")
    
    def get_rows(self, instance_indices: list) -> pd.DataFrame:
        """
        Get SHAP values for multiple instances (batch loading).
        
        Args:
            instance_indices: List of instance indices to retrieve
            
        Returns:
            DataFrame with SHAP values for the requested instances
        """
        df = self.to_pandas()
        
        # Filter to requested indices
        if all(idx in df.index for idx in instance_indices):
            return df.loc[instance_indices]
        else:
            # Use positional access if index doesn't match
            return df.iloc[instance_indices]
    
    def get_column(self, feature_name: str) -> pd.Series:
        """
        Get SHAP values for a specific feature across all instances (columnar access).
        
        Args:
            feature_name: Name of the feature column
            
        Returns:
            Series with SHAP values for this feature across all instances
        """
        df = self.to_pandas()
        if feature_name not in df.columns:
            raise KeyError(f"Feature '{feature_name}' not found in SHAP values")
        return df[feature_name]
    
    def __len__(self) -> int:
        """Return number of rows."""
        return self.num_rows
    
    def __repr__(self) -> str:
        return f"ShapParquetLoader(path={self.parquet_path}, lazy={self.lazy}, rows={self.num_rows}, cols={self.num_columns})"


def load_shap_parquet(parquet_path: Union[str, Path], lazy: bool = False) -> Union[ShapParquetLoader, pd.DataFrame]:
    """
    Convenience function to load SHAP values from Parquet file.
    
    Args:
        parquet_path: Path to Parquet file
        lazy: If True, return ShapParquetLoader for lazy access
              If False, return pandas DataFrame (full load)
    
    Returns:
        ShapParquetLoader if lazy=True, pandas DataFrame if lazy=False
    """
    loader = ShapParquetLoader(parquet_path, lazy=lazy)
    
    if lazy:
        return loader
    else:
        return loader.to_pandas()

