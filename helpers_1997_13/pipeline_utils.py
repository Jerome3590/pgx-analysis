"""
Pipeline utilities and pipeline state tracking merged into one module.

This file contains multiprocessing helpers (get_multiprocessing_context,
persist/load mappings) and the PipelineState + GlobalPipelineTracker classes.
"""

import os
import sys
import json
import tempfile
import multiprocessing as mp
from typing import Dict, Optional, Tuple, Any
import boto3
from datetime import datetime
import logging

# Default S3 bucket used for pipeline state metadata
S3_BUCKET = os.environ.get("PGX_S3_BUCKET", "pgx-repository")
STATE_PREFIX = "pgx-pipeline-status"


def get_multiprocessing_context():
	"""
	Get the appropriate multiprocessing context.
	Returns (context, method_name).
	"""
	mp_start_method = os.getenv('PGX_MP_START_METHOD', '').lower()
	if mp_start_method in ('fork', 'spawn'):
		try:
			return mp.get_context(mp_start_method), mp_start_method
		except (ValueError, RuntimeError) as e:
			print(f"⚠️ Warning: Requested start method '{mp_start_method}' not available: {e}, falling back to 'spawn'")
			return mp.get_context('spawn'), 'spawn'

	return mp.get_context('spawn'), 'spawn'


def persist_mappings_to_temp(icd_map: Dict[str, str], cpt_map: Dict[str, str],
							   icd_target_map: Dict[str, str], drug_map: Optional[Dict[str, str]] = None) -> Optional[str]:
	"""
	Persist mappings to temporary JSON files to reduce memory duplication in spawn mode.
	Returns temp directory path if successful, None otherwise.
	"""
	try:
		mapping_temp_dir = tempfile.mkdtemp(prefix='pgx_mappings_')

		if icd_map:
			with open(os.path.join(mapping_temp_dir, 'icd_map.json'), 'w') as f:
				json.dump(icd_map, f)
		if cpt_map:
			with open(os.path.join(mapping_temp_dir, 'cpt_map.json'), 'w') as f:
				json.dump(cpt_map, f)
		if icd_target_map:
			with open(os.path.join(mapping_temp_dir, 'icd_target_map.json'), 'w') as f:
				json.dump(icd_target_map, f)
		if drug_map:
			with open(os.path.join(mapping_temp_dir, 'drug_map.json'), 'w') as f:
				json.dump(drug_map, f)

		return mapping_temp_dir
	except Exception as e:
		print(f"⚠️ Warning: Could not persist mappings to temp files: {e}")
		return None


def load_mappings_from_temp(mapping_temp_dir: Optional[str]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str], Dict[str, str]]:
	"""
	Load mappings from temporary JSON files.
	Returns (icd_map, cpt_map, icd_target_map, drug_map) tuple.
	"""
	if not mapping_temp_dir or not os.path.exists(mapping_temp_dir):
		return {}, {}, {}, {}

	try:
		icd_map = {}
		cpt_map = {}
		icd_target_map = {}
		drug_map = {}

		icd_path = os.path.join(mapping_temp_dir, 'icd_map.json')
		if os.path.exists(icd_path):
			with open(icd_path, 'r') as f:
				icd_map = json.load(f)

		cpt_path = os.path.join(mapping_temp_dir, 'cpt_map.json')
		if os.path.exists(cpt_path):
			with open(cpt_path, 'r') as f:
				cpt_map = json.load(f)

		target_path = os.path.join(mapping_temp_dir, 'icd_target_map.json')
		if os.path.exists(target_path):
			with open(target_path, 'r') as f:
				icd_target_map = json.load(f)

		drug_path = os.path.join(mapping_temp_dir, 'drug_map.json')
		if os.path.exists(drug_path):
			with open(drug_path, 'r') as f:
				drug_map = json.load(f)

		return icd_map, cpt_map, icd_target_map, drug_map
	except Exception as e:
		print(f"⚠️ Warning: Could not load mappings from temp files: {e}")
		return {}, {}, {}, {}


def get_retry_attempts() -> int:
	"""Get retry attempts from environment, default to 3."""
	return int(os.getenv('PGX_RETRY_ATTEMPTS', '3'))


def get_timeout_seconds() -> int:
	"""Get timeout seconds from environment, default to 3600 (1 hour)."""
	return int(os.getenv('PGX_TIMEOUT_SECONDS', '3600'))


# ----------------------------- pipeline state classes -----------------------------


class PipelineState:
	"""Track pipeline execution state for resume and skip functionality."""

	def __init__(self, pipeline_name: str, entity_id: str, logger: Optional[logging.Logger] = None):
		self.pipeline_name = pipeline_name
		self.entity_id = entity_id.replace('/', '_')  # S3 safe
		self.logger = logger or logging.getLogger(__name__)
		self.s3_client = boto3.client('s3')
		self.state_key = f"{STATE_PREFIX}/{pipeline_name}/{self.entity_id}/state.json"
		self.state = self._load_state()

	def _load_state(self) -> Dict[str, Any]:
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
		completed = any(s['step_name'] == step_name for s in self.state['completed_steps'])
		if not completed:
			completed = self._check_step_checkpoint_exists(step_name)
			if completed:
				self.logger.info(f"✅ Found existing checkpoint for '{step_name}' in S3")
				self.state['completed_steps'].append({
					'step_name': step_name,
					'completed_at': datetime.utcnow().isoformat(),
					'metadata': {'recovered_from_checkpoint': True}
				})
		if completed:
			self.logger.info(f"⏭️  Step '{step_name}' already completed, skipping")
		return completed

	def _check_step_checkpoint_exists(self, step_name: str) -> bool:
		try:
			checkpoint_key = f"{STATE_PREFIX}/{self.pipeline_name}/{self.entity_id}/checkpoints/{step_name}.json"
			self.s3_client.head_object(Bucket=S3_BUCKET, Key=checkpoint_key)
			return True
		except:
			return False

	def mark_step_completed(self, step_name: str, metadata: Optional[Dict] = None):
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
		self.state['status'] = 'completed'
		self.state['completed_at'] = datetime.utcnow().isoformat()
		if metadata:
			self.state['metadata'].update(metadata)
		self._save_state()
		self.logger.info(f"🎉 Pipeline '{self.pipeline_name}' completed for {self.entity_id}")

	def get_progress(self) -> Dict[str, Any]:
		return {
			'pipeline_name': self.pipeline_name,
			'entity_id': self.entity_id,
			'status': self.state['status'],
			'completed_steps': len(self.state['completed_steps']),
			'failed_steps': len(self.state['failed_steps']),
			'step_names': [s['step_name'] for s in self.state['completed_steps']]
		}

	def reset(self):
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
		try:
			s3_client = boto3.client('s3')
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

#!/usr/bin/env python3
"""
Pipeline utilities for worker processing and multiprocessing coordination.
"""


