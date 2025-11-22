"""
Optimized configuration for HP Omen laptop
CPU: Intel 14 cores / 20 threads
RAM: 32GB (12.8GB available)
GPU: RTX 3080 Ti (16GB) - Currently unused (FP-Growth is CPU-only)
"""

# Optimal configuration based on hardware
OPTIMIZED_CONFIG = {
    # Global FPGrowth - processes all data at once (memory intensive)
    'global': {
        'MIN_SUPPORT': 0.003,  # Lower for more patterns
        'MIN_CONFIDENCE': 0.01,
        'TOP_K': 100,
        'MAX_WORKERS': 1,  # Global is sequential (large dataset)
    },
    
    # Cohort FPGrowth - parallel processing of independent cohorts
    'cohort': {
        'MIN_SUPPORT': 0.05,  # Higher for cohort-specific patterns
        'MIN_CONFIDENCE': 0.3,
        'TOP_K': 50,
        'MAX_WORKERS': 10,  # Aggressive: use 10 of 14 cores
        'TIMEOUT_SECONDS': 600,  # Increase timeout for complex patterns
    }
}

# Memory considerations
# - Each worker may use 1-2GB RAM
# - With 12.8GB available, 10 workers = ~10-12GB usage
# - Leaves ~2GB for system

# GPU Notes:
# - mlxtend FP-Growth is CPU-only
# - Data loading (DuckDB) is CPU-only  
# - No GPU acceleration available for this workload
# - GPU will sit idle during this analysis

# Recommendation: Focus on CPU parallelization
# Your 14-core CPU is perfect for cohort parallel processing!

print(f"Optimized configuration loaded:")
print(f"  Global analysis: Single-threaded (large dataset)")
print(f"  Cohort analysis: {OPTIMIZED_CONFIG['cohort']['MAX_WORKERS']} parallel workers")
print(f"  Expected memory usage: ~10-12 GB")
print(f"  GPU: Idle (FP-Growth is CPU-only)")

