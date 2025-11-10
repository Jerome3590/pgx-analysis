"""
Modular pipeline phases for cohort creation.

Each phase is in its own file for better maintainability.
"""

from .phase1_data_preparation import run_phase1_data_preparation
from .phase2_event_processing import (
    run_phase2_step1_event_fact_table,
    run_phase2_step2_drug_exposure
)
from .phase3_cohort_creation import run_phase3_step3_final_cohort_fact
from .phase4_finalization import run_phase4_complete_pipeline

__all__ = [
    'run_phase1_data_preparation',
    'run_phase2_step1_event_fact_table',
    'run_phase2_step2_drug_exposure',
    'run_phase3_step3_final_cohort_fact',
    'run_phase4_complete_pipeline',
]

