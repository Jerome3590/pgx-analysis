"""
File Resolver - Universal file path resolution for pgx-analysis project

Provides a centralized pattern for finding files across multiple locations:
- Local project directories (outputs, cached)
- Data root (NVMe/local storage)  
- S3 (with automatic download and caching)

This standardizes file resolution across all notebooks and scripts.

Usage Examples:
    from py_helpers.file_resolver import FileResolver
    
    # Create resolver for a specific file type
    resolver = FileResolver(
        file_type="cohort_feature_importance",
        project_root=Path.cwd(),
        cohort="opioid_ed",
        age_band="65-74"
    )
    
    # Resolve path (returns Path or None)
    path = resolver.resolve()
    
    # Load file directly (with automatic format detection)
    df = resolver.load()
"""

import io
import json
import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any, Callable
import logging

import pandas as pd
import duckdb

logger = logging.getLogger(__name__)


def age_band_to_fname(age_band: str) -> str:
    """Convert age_band with hyphens to underscore format (e.g., '65-74' -> '65_74')."""
    return age_band.replace("-", "_")


class FileResolver:
    """
    Universal file resolver for the pgx-analysis project.
    
    Handles resolution of files across multiple locations with fallback logic:
    1. Local project outputs/cache directories
    2. Environment variable overrides (e.g., PGX_FEATURE_IMPORTANCE_OUTPUTS)
    3. Data root (NVMe/local storage)
    4. S3 (with automatic download and local caching)
    
    Attributes:
        file_type: Type of file to resolve (e.g., "cohort_feature_importance", "aggregated_fi")
        project_root: Root directory of the project
        cohort: Optional cohort name (e.g., "opioid_ed", "non_opioid_ed")
        age_band: Optional age band (e.g., "65-74", "0-12")
        event_year: Optional event year (e.g., 2024)
        custom_paths: Optional list of custom paths to check
        auto_download: Whether to automatically download from S3 if not found locally
    """
    
    # File type configurations defining search patterns and S3 locations
    FILE_TYPE_CONFIGS: Dict[str, Dict[str, Any]] = {
        "cohort_feature_importance": {
            "filename_pattern": "{cohort}_{age_band_fname}_cohort_feature_importance.csv",
            "local_paths": [
                "3b_feature_importance_eda/outputs/{cohort}/{age_band_fname}",
                "3b_feature_importance_eda/outputs/{cohort}/{age_band}",
            ],
            "data_root_paths": [
                "gold/feature_importance/{cohort}/{age_band}",
                "gold/feature_importance/{cohort}/{age_band_fname}",
            ],
            "s3_key": "gold/feature_importance/{cohort}/{age_band}/{filename}",
            "cache_dir": "3b_feature_importance_eda/outputs/{cohort}",
        },
        "aggregated_feature_importance": {
            "filename_pattern": "{cohort}_{age_band_fname}_aggregated_feature_importance.csv",
            "env_var": "PGX_FEATURE_IMPORTANCE_OUTPUTS",
            "env_path": "{cohort}",
            "local_paths": [
                "3a_feature_importance/{cohort}",
                "3a_feature_importance/{cohort}/{age_band}",
                "3a_feature_importance/from_s3/by_cohort/{cohort}/{age_band}",
            ],
            "data_root_paths": [
                "gold/feature_importance/{cohort}/{age_band}",
            ],
            "s3_key": "gold/feature_importance/{cohort}/{age_band}/{filename}",
            "cache_dir": "3a_feature_importance/{cohort}",
        },
        "bupar_post_target_analysis": {
            "filename_pattern": "{cohort}_{age_band_fname}_bupar_post_target_analysis.csv",
            "local_paths": [
                "3b_feature_importance_eda/outputs/{cohort}/{age_band_fname}",
                "3b_feature_importance_eda/outputs/{cohort}/{age_band}",
            ],
            "data_root_paths": [
                "gold/feature_importance/{cohort}/{age_band}",
            ],
            "s3_key": "gold/feature_importance/{cohort}/{age_band}/{filename}",
            "cache_dir": "3b_feature_importance_eda/outputs/{cohort}",
        },
        "cohort_parquet": {
            "filename_pattern": "cohort.parquet",
            "local_paths": [
                "data/gold/cohorts/cohort_name={cohort}/event_year={event_year}/age_band={age_band}",
            ],
            "data_root_paths": [
                "gold/cohorts/cohort_name={cohort}/event_year={event_year}/age_band={age_band}",
            ],
            "s3_key": "gold/cohorts/cohort_name={cohort}/event_year={event_year}/age_band={age_band}/{filename}",
        },
        "model_data": {
            "filename_pattern": "model_events.parquet",
            "local_paths": [
                "data/gold/model_data/{cohort}/{age_band}/{event_year}",
            ],
            "data_root_paths": [
                "gold/model_data/{cohort}/{age_band}/{event_year}",
            ],
            "s3_key": "gold/model_data/{cohort}/{age_band}/{event_year}/{filename}",
        },
        "final_model": {
            "filename_pattern": "{model_type}.{extension}",  # e.g., xgboost.json, catboost.joblib
            "local_paths": [
                "6_final_model/outputs/{cohort}/{age_band}",
            ],
            "data_root_paths": [
                "gold/models/{cohort}/{age_band}",
            ],
            "s3_key": "gold/models/{cohort}/{age_band}/{filename}",
        },
        "administrative_codes_lookup": {
            "filename_pattern": "administrative_codes_lookup.json",
            "local_paths": [
                "1b_apcd_event_filter",
                "4b_event_filter",
                "3b_feature_importance_eda/0_icd_cpt_check",
            ],
        },
    }
    
    def __init__(
        self,
        file_type: str,
        project_root: Union[str, Path],
        cohort: Optional[str] = None,
        age_band: Optional[str] = None,
        event_year: Optional[int] = None,
        model_type: Optional[str] = None,
        extension: Optional[str] = None,
        custom_paths: Optional[List[Union[str, Path]]] = None,
        auto_download: bool = True,
        s3_bucket: Optional[str] = None,
    ):
        """
        Initialize FileResolver.
        
        Args:
            file_type: Type of file to resolve (must be in FILE_TYPE_CONFIGS)
            project_root: Root directory of the project
            cohort: Cohort name (e.g., "opioid_ed")
            age_band: Age band (e.g., "65-74")
            event_year: Event year (e.g., 2024)
            model_type: Model type (e.g., "xgboost", "catboost")
            extension: File extension (e.g., "json", "joblib")
            custom_paths: Additional paths to check
            auto_download: Whether to download from S3 if not found locally
            s3_bucket: S3 bucket name (defaults to PGX_S3_BUCKET env or "pgxdatalake")
        """
        self.file_type = file_type
        self.project_root = Path(project_root)
        self.cohort = cohort
        self.age_band = age_band
        self.age_band_fname = age_band_to_fname(age_band) if age_band else None
        self.event_year = event_year
        self.model_type = model_type
        self.extension = extension
        self.custom_paths = [Path(p) for p in custom_paths] if custom_paths else []
        self.auto_download = auto_download
        
        # Get S3 bucket
        self.s3_bucket = s3_bucket or os.environ.get("PGX_S3_BUCKET", "pgxdatalake")
        
        # Get file type configuration
        if file_type not in self.FILE_TYPE_CONFIGS:
            raise ValueError(
                f"Unknown file_type '{file_type}'. "
                f"Available types: {', '.join(self.FILE_TYPE_CONFIGS.keys())}"
            )
        self.config = self.FILE_TYPE_CONFIGS[file_type]
        
        # Get data root
        self.data_root = self._get_data_root()
    
    def _get_data_root(self) -> Optional[Path]:
        """Get data root from environment or defaults."""
        try:
            from py_helpers.env_utils import get_data_root
            return get_data_root()
        except ImportError:
            # Fallback defaults
            data_root_env = os.environ.get("DATA_ROOT") or os.environ.get("LOCAL_DATA_PATH")
            if data_root_env:
                return Path(data_root_env)
            
            # Common defaults
            for default in ["/nvme", "/nvme/pgx-analysis/data", "C:/Projects/pgx-analysis/data"]:
                if Path(default).exists():
                    return Path(default)
            
            return None
    
    def _format_path(self, template: str, *, include_filename: bool = True) -> str:
        """Format a path template with current parameters.
        When include_filename is False, {filename} is not expanded (avoids recursion when formatting the filename pattern).
        """
        kwargs = dict(
            cohort=self.cohort or "",
            age_band=self.age_band or "",
            age_band_fname=self.age_band_fname or "",
            event_year=self.event_year or "",
            model_type=self.model_type or "",
            extension=self.extension or "",
        )
        if include_filename:
            kwargs["filename"] = self._get_filename()
        else:
            kwargs["filename"] = ""
        return template.format(**kwargs)
    
    def _get_filename(self) -> str:
        """Get the filename based on the config pattern."""
        pattern = self.config.get("filename_pattern", "")
        return self._format_path(pattern, include_filename=False)
    
    def get_candidate_paths(self) -> List[Path]:
        """Return all paths that resolve() checks (for inclusion in FileNotFoundError messages)."""
        return self._get_candidate_paths()

    def _get_candidate_paths(self) -> List[Path]:
        """Get all candidate paths to check in order."""
        candidates = []
        
        # 1. Custom paths (highest priority)
        for path in self.custom_paths:
            if not path.is_absolute():
                path = self.project_root / path
            candidates.append(path / self._get_filename() if path.is_dir() else path)
        
        # 2. Environment variable override
        env_var = self.config.get("env_var")
        if env_var:
            env_path = os.environ.get(env_var)
            if env_path:
                env_subpath = self.config.get("env_path", "")
                full_path = Path(env_path) / self._format_path(env_subpath) / self._get_filename()
                candidates.append(full_path)
        
        # 3. Local project paths
        for path_template in self.config.get("local_paths", []):
            path = self.project_root / self._format_path(path_template) / self._get_filename()
            candidates.append(path)
        
        # 4. Data root paths
        if self.data_root:
            for path_template in self.config.get("data_root_paths", []):
                path = self.data_root / self._format_path(path_template) / self._get_filename()
                candidates.append(path)
        
        return candidates
    
    def resolve(self, download_if_missing: Optional[bool] = None) -> Optional[Path]:
        """
        Resolve file path, optionally downloading from S3 if not found locally.
        
        Args:
            download_if_missing: Whether to download from S3 (overrides auto_download)
        
        Returns:
            Path to file if found, None otherwise
        """
        # Check local paths first
        for path in self._get_candidate_paths():
            if path.exists():
                logger.debug(f"Resolved {self.file_type} at: {path}")
                return path
        
        # Try S3 download if enabled
        should_download = download_if_missing if download_if_missing is not None else self.auto_download
        if should_download and "s3_key" in self.config:
            downloaded_path = self._download_from_s3()
            if downloaded_path:
                return downloaded_path
        
        logger.debug(f"Could not resolve {self.file_type} for {self.cohort}/{self.age_band}")
        return None
    
    def _download_from_s3(self) -> Optional[Path]:
        """Download file from S3 and cache locally."""
        try:
            # Import S3 client
            try:
                from py_helpers.common_imports import s3_client
            except ImportError:
                import boto3
                s3_client = boto3.client("s3")
            
            # Get S3 key
            s3_key = self._format_path(self.config["s3_key"])
            
            # Get cache directory
            cache_template = self.config.get("cache_dir", "cache/{cohort}")
            cache_dir = self.project_root / self._format_path(cache_template)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            cache_path = cache_dir / self._get_filename()
            
            # Download from S3
            logger.info(f"Downloading {self.file_type} from S3: {s3_key}")
            obj = s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            
            # Write to cache
            with open(cache_path, "wb") as f:
                f.write(obj["Body"].read())
            
            logger.info(f"Cached {self.file_type} at: {cache_path}")
            return cache_path
            
        except Exception as e:
            logger.warning(f"Failed to download from S3: {e}")
            return None
    
    def load(self, **kwargs) -> Optional[Union[pd.DataFrame, Dict, Any]]:
        """
        Load file with automatic format detection.
        
        Args:
            **kwargs: Additional arguments passed to the loader (e.g., pd.read_csv options)
        
        Returns:
            Loaded data (DataFrame, dict, etc.) or None if file not found
        
        Raises:
            ValueError: If file is empty or invalid
        """
        path = self.resolve()
        if not path:
            return None
        
        # Determine file format and load
        suffix = path.suffix.lower()
        
        if suffix == ".csv":
            df = pd.read_csv(path, **kwargs)
            if df.empty:
                raise ValueError(f"File is empty: {path}")
            return df
        
        elif suffix == ".parquet":
            # Try DuckDB first (faster for large files)
            try:
                con = duckdb.connect()
                path_esc = str(path).replace("'", "''")
                df = con.execute(f"SELECT * FROM read_parquet('{path_esc}')").df()
                con.close()
            except Exception:
                # Fallback to pandas
                df = pd.read_parquet(path, **kwargs)
            
            if df.empty:
                raise ValueError(f"File is empty: {path}")
            return df
        
        elif suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)
            return data
        
        elif suffix == ".joblib":
            import joblib
            return joblib.load(path)
        
        else:
            # Return path for other formats
            logger.warning(f"No loader for {suffix} files, returning path")
            return path
    
    def exists(self) -> bool:
        """Check if file exists (without downloading)."""
        return self.resolve(download_if_missing=False) is not None
    
    def __repr__(self) -> str:
        return (
            f"FileResolver(file_type='{self.file_type}', cohort={self.cohort}, "
            f"age_band={self.age_band}, event_year={self.event_year})"
        )


# Convenience functions for common file types

def resolve_cohort_fi_path(
    cohort: str,
    age_band: str,
    project_root: Path,
) -> Optional[Path]:
    """
    Resolve path to Step 3b refined cohort_feature_importance CSV.
    
    Convenience wrapper for FileResolver.
    """
    resolver = FileResolver(
        file_type="cohort_feature_importance",
        project_root=project_root,
        cohort=cohort,
        age_band=age_band,
    )
    return resolver.resolve()


def load_cohort_feature_importance(
    cohort: str,
    age_band: str,
    project_root: Path,
) -> pd.DataFrame:
    """
    Load cohort feature importance CSV.
    
    Raises:
        FileNotFoundError: If file not found
        ValueError: If file is empty
    """
    resolver = FileResolver(
        file_type="cohort_feature_importance",
        project_root=project_root,
        cohort=cohort,
        age_band=age_band,
    )
    df = resolver.load()
    if df is None:
        paths_checked = resolver.get_candidate_paths()
        paths_str = "\n  ".join(str(p) for p in paths_checked) if paths_checked else "(none)"
        raise FileNotFoundError(
            f"Could not find cohort_feature_importance for {cohort}/{age_band}. "
            f"Checked:\n  {paths_str}\n"
            "Check that Step 3b has completed for this cohort/age_band."
        )

    # Polypharmacy cohort: use only drug-name features (drop ICD/CPT)
    if cohort == "non_opioid_ed":
        from py_helpers.feature_utils import filter_fi_to_drug_only
        df = filter_fi_to_drug_only(df, feature_col="feature")
        if df.empty:
            raise ValueError(
                f"Cohort feature importance for {cohort}/{age_band} has no drug-name features. "
                "Polypharmacy cohort requires drug features only."
            )

    return df


def resolve_aggregated_fi_path(
    cohort: str,
    age_band: str,
    project_root: Path,
) -> Optional[Path]:
    """
    Resolve path to Step 3a aggregated feature importance CSV.
    
    Convenience wrapper for FileResolver.
    """
    resolver = FileResolver(
        file_type="aggregated_feature_importance",
        project_root=project_root,
        cohort=cohort,
        age_band=age_band,
    )
    return resolver.resolve()


def load_aggregated_feature_importance(
    cohort: str,
    age_band: str,
    project_root: Path,
) -> pd.DataFrame:
    """
    Load aggregated feature importance from Step 3a.
    
    Raises:
        FileNotFoundError: If file not found
        ValueError: If file is empty
    """
    resolver = FileResolver(
        file_type="aggregated_feature_importance",
        project_root=project_root,
        cohort=cohort,
        age_band=age_band,
    )
    df = resolver.load()
    if df is None:
        paths_checked = resolver.get_candidate_paths()
        paths_str = "\n  ".join(str(p) for p in paths_checked) if paths_checked else "(none)"
        raise FileNotFoundError(
            f"Could not find aggregated_feature_importance for {cohort}/{age_band}. "
            f"Checked:\n  {paths_str}\n"
            "Run Step 3a for this cohort/age_band to produce the file."
        )
    
    # Validate structure
    if "feature" not in df.columns and len(df.columns) < 2:
        raise ValueError(
            f"Invalid aggregated feature importance structure. "
            f"Expected columns: feature, importance_mean/importance_scaled_by_model_sum. "
            f"Found: {list(df.columns)}"
        )

    # Polypharmacy cohort: use only drug-name features (drop ICD/CPT)
    if cohort == "non_opioid_ed":
        from py_helpers.feature_utils import filter_fi_to_drug_only
        df = filter_fi_to_drug_only(df, feature_col="feature")
        if df.empty:
            raise ValueError(
                f"Aggregated feature importance for {cohort}/{age_band} has no drug-name features. "
                "Polypharmacy cohort requires drug features only."
            )

    return df


def get_administrative_lookup_path(project_root: Path) -> Optional[Path]:
    """Get path to administrative_codes_lookup.json."""
    resolver = FileResolver(
        file_type="administrative_codes_lookup",
        project_root=project_root,
    )
    return resolver.resolve(download_if_missing=False)


def load_administrative_codes(project_root: Path) -> Dict[str, List[str]]:
    """Load administrative codes (ICD/CPT/HCPCS) lookup."""
    path = get_administrative_lookup_path(project_root)
    if not path:
        return {"icd": [], "cpt": [], "hcpcs": []}
    
    with open(path) as f:
        data = json.load(f)
    
    return data.get("administrative_codes", {})
