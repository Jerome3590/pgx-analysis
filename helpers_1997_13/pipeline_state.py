"""
Pipeline state management for tracking progress across all stages.

This module provides a centralized way to track pipeline progress,
enabling resume functionality and avoiding redundant work.
"""

import json
import boto3
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

S3_BUCKET = "pgx-repository"
STATE_PREFIX = "pgx-pipeline-status"  # Central checkpoint location


class PipelineState:
    """Track pipeline execution state for resume and skip functionality."""
    
    def __init__(self, pipeline_name: str, entity_id: str, logger: Optional[logging.Logger] = None):
        """
        Initialize pipeline state tracker.
        
        Args:
            pipeline_name: Name of pipeline (e.g., 'pharmacy_clean', 'create_cohort')
            entity_id: Unique identifier (e.g., '65-74/2020', 'opioid_ed_65-74_2020')
            logger: Logger instance
        """
        self.pipeline_name = pipeline_name
        self.entity_id = entity_id.replace('/', '_')  # S3 safe
        self.logger = logger or logging.getLogger(__name__)
        self.s3_client = boto3.client('s3')
        self.state_key = f"{STATE_PREFIX}/{pipeline_name}/{self.entity_id}/state.json"
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load existing state from S3 or return empty state."""
        try:
            response = self.s3_client.get_object(Bucket=S3_BUCKET, Key=self.state_key)
            state = json.loads(response['Body'].read().decode('utf-8'))
            self.logger.info(f"📂 Loaded pipeline state: {len(state.get('completed_steps', []))} steps completed")
            return state
        except self.s3_client.exceptions.NoSuchKey:
            self.logger.info(f"📂 No existing state found, starting fresh")
            return {
                'pipeline_name': self.pipeline_name,
                'entity_id': self.entity_id,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'status': 'running',
                'completed_steps': [],
                'failed_steps': [],
                'metadata': {}
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load state: {e}, starting fresh")
            return {
                'pipeline_name': self.pipeline_name,
                'entity_id': self.entity_id,
                'created_at': datetime.utcnow().isoformat(),
                'status': 'running',
                'completed_steps': [],
                'failed_steps': [],
                'metadata': {}
            }
    
    def _save_state(self):
        """Save current state to S3."""
        try:
            self.state['updated_at'] = datetime.utcnow().isoformat()
            self.s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=self.state_key,
                Body=json.dumps(self.state, indent=2),
                ContentType='application/json'
            )
            self.logger.debug(f"💾 Saved pipeline state to s3://{S3_BUCKET}/{self.state_key}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save state: {e}")
    
    def is_step_completed(self, step_name: str) -> bool:
        """
        Check if a step has been completed.
        
        Checks both:
        1. Local state (in-memory)
        2. S3 checkpoint (source of truth)
        """
        # Check in-memory state first
        completed = any(s['step_name'] == step_name for s in self.state['completed_steps'])
        
        # If not in memory, check S3 checkpoint (handles recovery scenarios)
        if not completed:
            completed = self._check_step_checkpoint_exists(step_name)
            if completed:
                self.logger.info(f"✅ Found existing checkpoint for '{step_name}' in S3")
                # Add to local state to avoid re-checking
                self.state['completed_steps'].append({
                    'step_name': step_name,
                    'completed_at': datetime.utcnow().isoformat(),
                    'metadata': {'recovered_from_checkpoint': True}
                })
        
        if completed:
            self.logger.info(f"⏭️  Step '{step_name}' already completed, skipping")
        return completed
    
    def _check_step_checkpoint_exists(self, step_name: str) -> bool:
        """Check if step checkpoint exists in S3."""
        try:
            checkpoint_key = f"{STATE_PREFIX}/{self.pipeline_name}/{self.entity_id}/checkpoints/{step_name}.json"
            self.s3_client.head_object(Bucket=S3_BUCKET, Key=checkpoint_key)
            return True
        except:
            return False
    
    def mark_step_completed(self, step_name: str, metadata: Optional[Dict] = None):
        """Mark a step as completed with optional metadata."""
        if not self.is_step_completed(step_name):
            step_data = {
                'step_name': step_name,
                'completed_at': datetime.utcnow().isoformat(),
                'metadata': metadata or {}
            }
            self.state['completed_steps'].append(step_data)
            self._save_state()
            self._save_step_checkpoint(step_name, metadata)
            self.logger.info(f"✅ Marked step '{step_name}' as completed")
    
    def _save_step_checkpoint(self, step_name: str, metadata: Optional[Dict] = None):
        """
        Save individual step checkpoint to S3.
        
        This creates a separate checkpoint file for each step at:
        s3://pgx-repository/pgx-pipeline-status/{pipeline}/{entity}/checkpoints/{step_name}.json
        
        This makes it easy to check if specific steps are complete.
        """
        try:
            checkpoint_key = f"{STATE_PREFIX}/{self.pipeline_name}/{self.entity_id}/checkpoints/{step_name}.json"
            checkpoint_data = {
                'pipeline_name': self.pipeline_name,
                'entity_id': self.entity_id,
                'step_name': step_name,
                'completed_at': datetime.utcnow().isoformat(),
                'status': 'completed',
                'metadata': metadata or {}
            }
            
            self.s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=checkpoint_key,
                Body=json.dumps(checkpoint_data, indent=2),
                ContentType='application/json'
            )
            self.logger.debug(f"💾 Saved step checkpoint: s3://{S3_BUCKET}/{checkpoint_key}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not save step checkpoint for '{step_name}': {e}")
    
    def mark_step_failed(self, step_name: str, error: str):
        """Mark a step as failed with error details."""
        step_data = {
            'step_name': step_name,
            'failed_at': datetime.utcnow().isoformat(),
            'error': str(error)
        }
        self.state['failed_steps'].append(step_data)
        self.state['status'] = 'failed'
        self._save_state()
        self._save_step_failure_checkpoint(step_name, error)
        self.logger.error(f"❌ Marked step '{step_name}' as failed: {error}")
    
    def _save_step_failure_checkpoint(self, step_name: str, error: str):
        """Save failure checkpoint to S3 for debugging."""
        try:
            checkpoint_key = f"{STATE_PREFIX}/{self.pipeline_name}/{self.entity_id}/failures/{step_name}.json"
            checkpoint_data = {
                'pipeline_name': self.pipeline_name,
                'entity_id': self.entity_id,
                'step_name': step_name,
                'failed_at': datetime.utcnow().isoformat(),
                'status': 'failed',
                'error': str(error)
            }
            
            self.s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=checkpoint_key,
                Body=json.dumps(checkpoint_data, indent=2),
                ContentType='application/json'
            )
            self.logger.debug(f"💾 Saved failure checkpoint: s3://{S3_BUCKET}/{checkpoint_key}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not save failure checkpoint for '{step_name}': {e}")
    
    def mark_pipeline_completed(self, metadata: Optional[Dict] = None):
        """Mark entire pipeline as completed."""
        self.state['status'] = 'completed'
        self.state['completed_at'] = datetime.utcnow().isoformat()
        if metadata:
            self.state['metadata'].update(metadata)
        self._save_state()
        self.logger.info(f"🎉 Pipeline '{self.pipeline_name}' completed for {self.entity_id}")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress summary."""
        return {
            'pipeline_name': self.pipeline_name,
            'entity_id': self.entity_id,
            'status': self.state['status'],
            'completed_steps': len(self.state['completed_steps']),
            'failed_steps': len(self.state['failed_steps']),
            'step_names': [s['step_name'] for s in self.state['completed_steps']]
        }
    
    def reset(self):
        """Reset pipeline state (use with caution)."""
        self.state = {
            'pipeline_name': self.pipeline_name,
            'entity_id': self.entity_id,
            'created_at': datetime.utcnow().isoformat(),
            'status': 'running',
            'completed_steps': [],
            'failed_steps': [],
            'metadata': {}
        }
        self._save_state()
        self.logger.warning(f"🔄 Reset pipeline state for {self.entity_id}")
    
    @staticmethod
    def check_output_exists(s3_path: str) -> bool:
        """
        Check if output file exists in S3.
        
        This is the ultimate source of truth - if the final output exists,
        the step is complete regardless of state file.
        """
        try:
            s3_client = boto3.client('s3')
            # Parse S3 path
            if s3_path.startswith('s3://'):
                s3_path = s3_path[5:]
            parts = s3_path.split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
            
            s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except:
            return False


class GlobalPipelineTracker:
    """Track progress across all entities in a pipeline run."""
    
    def __init__(self, pipeline_name: str, logger: Optional[logging.Logger] = None):
        self.pipeline_name = pipeline_name
        self.logger = logger or logging.getLogger(__name__)
        self.s3_client = boto3.client('s3')
        self.tracker_key = f"{STATE_PREFIX}/{pipeline_name}/global_tracker.json"
        self.entities = self._load_tracker()
    
    def _load_tracker(self) -> Dict[str, Any]:
        """Load global tracker from S3."""
        try:
            response = self.s3_client.get_object(Bucket=S3_BUCKET, Key=self.tracker_key)
            return json.loads(response['Body'].read().decode('utf-8'))
        except:
            return {
                'pipeline_name': self.pipeline_name,
                'started_at': datetime.utcnow().isoformat(),
                'entities': {}
            }
    
    def _save_tracker(self):
        """Save global tracker to S3."""
        try:
            self.entities['updated_at'] = datetime.utcnow().isoformat()
            self.s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=self.tracker_key,
                Body=json.dumps(self.entities, indent=2),
                ContentType='application/json'
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to save global tracker: {e}")
    
    def register_entity(self, entity_id: str, metadata: Optional[Dict] = None):
        """Register an entity (cohort, age_band/year) for tracking."""
        entity_key = entity_id.replace('/', '_')
        if entity_key not in self.entities.get('entities', {}):
            self.entities.setdefault('entities', {})[entity_key] = {
                'entity_id': entity_id,
                'status': 'pending',
                'started_at': datetime.utcnow().isoformat(),
                'metadata': metadata or {}
            }
            self._save_tracker()
    
    def update_entity_status(self, entity_id: str, status: str, metadata: Optional[Dict] = None):
        """Update entity status (pending, running, completed, failed)."""
        entity_key = entity_id.replace('/', '_')
        if entity_key in self.entities.get('entities', {}):
            self.entities['entities'][entity_key]['status'] = status
            self.entities['entities'][entity_key]['updated_at'] = datetime.utcnow().isoformat()
            if metadata:
                self.entities['entities'][entity_key]['metadata'].update(metadata)
            if status == 'completed':
                self.entities['entities'][entity_key]['completed_at'] = datetime.utcnow().isoformat()
            self._save_tracker()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all entities."""
        entities = self.entities.get('entities', {})
        statuses = [e['status'] for e in entities.values()]
        return {
            'pipeline_name': self.pipeline_name,
            'total_entities': len(entities),
            'pending': statuses.count('pending'),
            'running': statuses.count('running'),
            'completed': statuses.count('completed'),
            'failed': statuses.count('failed'),
            'entities': list(entities.keys())
        }


# Example Usage in Pipeline
"""
from helpers_1997_13.pipeline_state import PipelineState

# Example 1: Pharmacy Cleaning Pipeline
def build_optimized_pipeline(age_band, event_year, ...):
    # Initialize state tracker
    entity_id = f"{age_band}/{event_year}"
    state = PipelineState('pharmacy_clean', entity_id, logger)
    
    # Checkpoints will be saved to:
    # s3://pgx-repository/pgx-pipeline-status/pharmacy_clean/55_64_2020/state.json
    # s3://pgx-repository/pgx-pipeline-status/pharmacy_clean/55_64_2020/checkpoints/load_pharmacy_data.json
    # s3://pgx-repository/pgx-pipeline-status/pharmacy_clean/55_64_2020/checkpoints/normalize_data.json
    # etc.
    
    # Check if already completed (ultimate check)
    output_path = f"s3://pgxdatalake/gold/pharmacy/age_band={age_band.replace('-','_')}/event_year={event_year}/data.parquet"
    if PipelineState.check_output_exists(output_path):
        logger.info("✅ Output already exists, skipping entire pipeline")
        state.mark_pipeline_completed({'output': output_path, 'skipped': True})
        return
    
    # Step 1: Load data
    if not state.is_step_completed('load_pharmacy_data'):
        logger.info("📊 Step 1: Loading pharmacy data...")
        # ... do work ...
        state.mark_step_completed('load_pharmacy_data', {'rows': 1000000})
    
    # Step 2: Data normalization
    if not state.is_step_completed('normalize_data'):
        try:
            logger.info("🔧 Step 2: Normalizing data...")
            # ... do work ...
            state.mark_step_completed('normalize_data', {'cleaned_rows': 950000})
        except Exception as e:
            state.mark_step_failed('normalize_data', str(e))
            raise
    
    # Step 3: Deduplication
    if not state.is_step_completed('deduplication'):
        logger.info("🔍 Step 3: Deduplication...")
        # ... do work ...
        state.mark_step_completed('deduplication', {'unique_rows': 900000})
    
    # Final step
    state.mark_pipeline_completed({'final_rows': 900000, 'output': output_path})


# Example 2: Cohort Creation Pipeline
def create_cohort(cohort_name, age_band, event_year, ...):
    entity_id = f"{cohort_name}_{age_band}_{event_year}"
    state = PipelineState('create_cohort', entity_id, logger)
    
    # Checkpoints will be saved to:
    # s3://pgx-repository/pgx-pipeline-status/create_cohort/opioid_ed_65_74_2020/state.json
    # s3://pgx-repository/pgx-pipeline-status/create_cohort/opioid_ed_65_74_2020/checkpoints/load_medical.json
    # s3://pgx-repository/pgx-pipeline-status/create_cohort/opioid_ed_65_74_2020/checkpoints/load_pharmacy.json
    # etc.
    
    if not state.is_step_completed('load_medical'):
        logger.info("📊 Loading medical data...")
        # ... do work ...
        state.mark_step_completed('load_medical', {'rows': 500000})
    
    if not state.is_step_completed('load_pharmacy'):
        logger.info("💊 Loading pharmacy data...")
        # ... do work ...
        state.mark_step_completed('load_pharmacy', {'rows': 300000})
    
    if not state.is_step_completed('identify_cases'):
        logger.info("🔍 Identifying cases...")
        # ... do work ...
        state.mark_step_completed('identify_cases', {'cases': 1000})
    
    state.mark_pipeline_completed({'cases': 1000, 'controls': 5000})


# Example 3: Global Pipeline Tracking (across all entities)
from helpers_1997_13.pipeline_state import GlobalPipelineTracker

def run_all_cohorts():
    tracker = GlobalPipelineTracker('create_cohort', logger)
    
    cohorts = [
        ('opioid_ed', '65-74', 2020),
        ('opioid_ed', '55-64', 2020),
        # ... more cohorts
    ]
    
    # Register all entities
    for cohort_name, age_band, year in cohorts:
        entity_id = f"{cohort_name}_{age_band}_{year}"
        tracker.register_entity(entity_id)
    
    # Process each entity
    for cohort_name, age_band, year in cohorts:
        entity_id = f"{cohort_name}_{age_band}_{year}"
        tracker.update_entity_status(entity_id, 'running')
        
        try:
            create_cohort(cohort_name, age_band, year, ...)
            tracker.update_entity_status(entity_id, 'completed')
        except Exception as e:
            tracker.update_entity_status(entity_id, 'failed', {'error': str(e)})
    
    # Get summary
    summary = tracker.get_summary()
    logger.info(f"Pipeline complete: {summary['completed']}/{summary['total_entities']} succeeded")
"""

