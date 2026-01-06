"""
Efficient Parquet-based SHAP value loader using DuckDB.

This module provides optimized loading of SHAP values from Parquet files
using DuckDB for efficient columnar queries without loading entire files into memory.
Only converts to pandas at the final step for compatibility.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import pandas as pd

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

logger = logging.getLogger(__name__)


class ShapParquetLoader:
    """
    Efficient loader for SHAP values stored in Parquet format using DuckDB.
    
    Uses DuckDB to query Parquet files directly without loading entire files into memory.
    Only converts to pandas at the final step for compatibility with existing code.
    """
    
    def __init__(self, parquet_path: Union[str, Path]):
        """
        Initialize SHAP Parquet loader with DuckDB.
        
        Args:
            parquet_path: Path to Parquet file containing SHAP values
        """
        if not DUCKDB_AVAILABLE:
            raise ImportError("DuckDB is required for efficient Parquet access. Install with: pip install duckdb")
        
        self.parquet_path = Path(parquet_path)
        self._conn = None
        self._metadata = None
        
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"SHAP Parquet file not found: {self.parquet_path}")
        
        # Initialize DuckDB connection
        self._conn = duckdb.connect()
        
        # Load metadata immediately (fast, no data loading)
        self._load_metadata()
    
    def _load_metadata(self):
        """Load Parquet file metadata using DuckDB (fast, no data loading)."""
        try:
            # Use DuckDB to get metadata without loading data
            result = self._conn.execute(f"""
                SELECT COUNT(*) as num_rows
                FROM read_parquet('{self.parquet_path}')
            """).fetchone()
            
            # Get column names
            result_cols = self._conn.execute(f"""
                DESCRIBE SELECT * FROM read_parquet('{self.parquet_path}')
            """).fetchall()
            
            self._metadata = {
                'num_rows': result[0] if result else 0,
                'num_columns': len(result_cols),
                'column_names': [col[0] for col in result_cols]
            }
            logger.debug(f"Parquet metadata: {self._metadata['num_rows']} rows, {self._metadata['num_columns']} columns")
        except Exception as e:
            logger.warning(f"Could not load Parquet metadata: {e}")
            self._metadata = None
    
    @property
    def num_rows(self) -> int:
        """Number of rows (instances) in the Parquet file."""
        if self._metadata:
            return self._metadata['num_rows']
        else:
            # Query DuckDB directly
            result = self._conn.execute(f"""
                SELECT COUNT(*) FROM read_parquet('{self.parquet_path}')
            """).fetchone()
            return result[0] if result else 0
    
    @property
    def num_columns(self) -> int:
        """Number of columns (features) in the Parquet file."""
        if self._metadata:
            return self._metadata['num_columns']
        else:
            result = self._conn.execute(f"""
                DESCRIBE SELECT * FROM read_parquet('{self.parquet_path}')
            """).fetchall()
            return len(result)
    
    @property
    def column_names(self) -> list:
        """Column names (feature names) in the Parquet file."""
        if self._metadata:
            return self._metadata['column_names']
        else:
            result = self._conn.execute(f"""
                DESCRIBE SELECT * FROM read_parquet('{self.parquet_path}')
            """).fetchall()
            return [col[0] for col in result]
    
    def to_pandas(self) -> pd.DataFrame:
        """
        Load SHAP values into a pandas DataFrame (only for final output).
        
        This should only be called when you need the full DataFrame.
        For individual row access, use get_row() instead.
        
        Returns:
            DataFrame with SHAP values, indexed by instance index
        """
        # Use DuckDB to read Parquet and convert to pandas
        df = self._conn.execute(f"""
            SELECT * FROM read_parquet('{self.parquet_path}')
        """).df()
        
        # Set index if first column looks like an index
        if df.index.name is None and len(df) > 0:
            # Check if first column is integer index
            first_col = df.columns[0]
            if first_col.lower() in ['index', 'instance_index', 'row_number'] or df[first_col].dtype == 'int64':
                df = df.set_index(first_col)
                df.index.name = 'instance_index'
        
        logger.info(f"Loaded SHAP values into pandas DataFrame: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def get_row(self, instance_index: int, index_column: Optional[str] = None) -> Dict[str, float]:
        """
        Get SHAP values for a specific instance using DuckDB (efficient, no full load).
        
        Args:
            instance_index: Index of the instance to retrieve
            index_column: Name of the index column (if None, uses row_number)
            
        Returns:
            Dictionary mapping feature_name -> SHAP value
        """
        # Use DuckDB to query only the specific row
        if index_column:
            # If we know the index column name, use it
            query = f"""
                SELECT * FROM read_parquet('{self.parquet_path}')
                WHERE {index_column} = {instance_index}
            """
        else:
            # Use row_number() for positional access
            query = f"""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) - 1 as row_idx
                    FROM read_parquet('{self.parquet_path}')
                ) WHERE row_idx = {instance_index}
            """
        
        result = self._conn.execute(query).df()
        
        if len(result) == 0:
            raise IndexError(f"Instance index {instance_index} not found in SHAP values")
        
        # Convert single row to dict (exclude row_idx if present)
        row = result.iloc[0]
        if 'row_idx' in row.index:
            row = row.drop('row_idx')
        
        return row.to_dict()
    
    def get_rows(self, instance_indices: list, index_column: Optional[str] = None) -> pd.DataFrame:
        """
        Get SHAP values for multiple instances using DuckDB (efficient batch loading).
        
        Args:
            instance_indices: List of instance indices to retrieve
            index_column: Name of the index column (if None, uses row_number)
            
        Returns:
            DataFrame with SHAP values for the requested instances
        """
        if not instance_indices:
            return pd.DataFrame()
        
        # Use DuckDB to query only the requested rows
        indices_str = ','.join(map(str, instance_indices))
        
        if index_column:
            query = f"""
                SELECT * FROM read_parquet('{self.parquet_path}')
                WHERE {index_column} IN ({indices_str})
            """
        else:
            # Use row_number() for positional access
            query = f"""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) - 1 as row_idx
                    FROM read_parquet('{self.parquet_path}')
                ) WHERE row_idx IN ({indices_str})
            """
        
        result = self._conn.execute(query).df()
        
        # Remove row_idx if present
        if 'row_idx' in result.columns:
            result = result.drop(columns=['row_idx'])
        
        return result
    
    def get_column(self, feature_name: str) -> pd.Series:
        """
        Get SHAP values for a specific feature across all instances using DuckDB (columnar access).
        
        This is very efficient as it only reads one column from Parquet.
        
        Args:
            feature_name: Name of the feature column
            
        Returns:
            Series with SHAP values for this feature across all instances
        """
        # Use DuckDB to read only the specific column (very efficient)
        result = self._conn.execute(f"""
            SELECT {feature_name} FROM read_parquet('{self.parquet_path}')
        """).df()
        
        if feature_name not in result.columns:
            raise KeyError(f"Feature '{feature_name}' not found in SHAP values")
        
        return result[feature_name]
    
    def __len__(self) -> int:
        """Return number of rows."""
        return self.num_rows
    
    def __repr__(self) -> str:
        return f"ShapParquetLoader(path={self.parquet_path}, rows={self.num_rows}, cols={self.num_columns})"
    
    def close(self):
        """Close DuckDB connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def load_shap_parquet(parquet_path: Union[str, Path], return_loader: bool = False) -> Union[ShapParquetLoader, pd.DataFrame]:
    """
    Convenience function to load SHAP values from Parquet file.
    
    Args:
        parquet_path: Path to Parquet file
        return_loader: If True, return ShapParquetLoader for efficient access
                      If False, return pandas DataFrame (full load, only for final output)
    
    Returns:
        ShapParquetLoader if return_loader=True, pandas DataFrame if return_loader=False
    """
    loader = ShapParquetLoader(parquet_path)
    
    if return_loader:
        return loader
    else:
        df = loader.to_pandas()
        loader.close()
        return df

