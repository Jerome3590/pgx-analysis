#!/usr/bin/env python3
"""
SQS Utilities for FP-Growth Job Queue Management

This module provides utilities for managing FP-Growth jobs through SQS queues
to prevent CPU overload and ensure controlled job execution.
"""

import boto3
import json
import time
import logging
from typing import Dict, List, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

# User's existing SQS queue configuration
QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/535362115856/cohorts.fifo"
SQS_REGION = "us-east-1"

class FPGrowthJobQueue:
    """
    Manages FP-Growth jobs through SQS FIFO queue to prevent CPU overload.
    """
    
    def __init__(self, queue_url: str = QUEUE_URL, region: str = SQS_REGION, logger=None):
        """
        Initialize FP-Growth job queue.
        
        Args:
            queue_url: URL of the SQS FIFO queue
            region: AWS region
            logger: Logger instance
        """
        self.queue_url = queue_url
        self.region = region
        self.logger = logger or logging.getLogger(__name__)
        
        # Get SQS client
        self.sqs_client = boto3.client('sqs', region_name=region)
        
    def enqueue_jobs(self, jobs: List[Dict[str, Any]]) -> int:
        """
        Add jobs to the FIFO queue.
        
        Args:
            jobs: List of job dictionaries
            
        Returns:
            Number of jobs enqueued
        """
        enqueued = 0
        
        for job in jobs:
            try:
                # Create cohort from age_band and event_year
                cohort = f"{job['age_band']}-{job['event_year']}"
                
                message = {
                    "QueueUrl": self.queue_url,
                    "MessageBody": json.dumps(job)
                }
                
                # Add FIFO-specific attributes
                if self.queue_url.endswith(".fifo"):
                    # Create unique group and deduplication IDs using cohort
                    message["MessageGroupId"] = f"group-{cohort}"
                    message["MessageDeduplicationId"] = cohort
                
                response = self.sqs_client.send_message(**message)
                enqueued += 1
                self.logger.info(f"Enqueued job {job['age_band']}/{job['event_year']}: {response['MessageId']}")
            except Exception as e:
                self.logger.error(f"Failed to enqueue job {job['age_band']}/{job['event_year']}: {e}")
        
        self.logger.info(f"Enqueued {enqueued}/{len(jobs)} jobs to SQS FIFO queue")
        return enqueued
    
    def dequeue_jobs(self, max_messages: int = 10, wait_time: int = 20) -> List[Dict[str, Any]]:
        """
        Receive jobs from the FIFO queue.
        
        Args:
            max_messages: Maximum number of messages to receive
            wait_time: Long polling wait time in seconds
            
        Returns:
            List of job dictionaries with receipt handles
        """
        try:
            response = self.sqs_client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=wait_time,
                AttributeNames=['All'],
                MessageAttributeNames=['All']
            )
            
            messages = response.get('Messages', [])
            jobs = []
            
            for message in messages:
                try:
                    job = json.loads(message['Body'])
                    job['receipt_handle'] = message['ReceiptHandle']
                    job['message_id'] = message['MessageId']
                    jobs.append(job)
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse message {message['MessageId']}: {e}")
                    # Delete malformed message
                    self._delete_message(message['ReceiptHandle'])
            
            if jobs:
                self.logger.info(f"Dequeued {len(jobs)} jobs from SQS FIFO queue")
            
            return jobs
            
        except Exception as e:
            self.logger.error(f"Failed to dequeue jobs: {e}")
            return []
    
    def delete_message(self, receipt_handle: str) -> bool:
        """Delete a message from the queue."""
        try:
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=receipt_handle
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete message: {e}")
            return False
    
    def _delete_message(self, receipt_handle: str) -> bool:
        """Internal method to delete message."""
        return self.delete_message(receipt_handle)
    
    def get_queue_attributes(self) -> Dict[str, Any]:
        """Get queue attributes for monitoring."""
        try:
            response = self.sqs_client.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=['All']
            )
            return response['Attributes']
        except Exception as e:
            self.logger.error(f"Failed to get queue attributes: {e}")
            return {}
    
    def purge_queue(self) -> bool:
        """Purge all messages from the queue."""
        try:
            self.sqs_client.purge_queue(QueueUrl=self.queue_url)
            self.logger.info("Purged all messages from SQS FIFO queue")
            return True
        except Exception as e:
            self.logger.error(f"Failed to purge queue: {e}")
            return False


def create_fpgrowth_job_queue(queue_url: str = QUEUE_URL, logger=None) -> FPGrowthJobQueue:
    """
    Create an FP-Growth job queue using the existing FIFO queue.
    
    Args:
        queue_url: URL of the SQS FIFO queue
        logger: Logger instance
        
    Returns:
        FPGrowthJobQueue instance
    """
    return FPGrowthJobQueue(queue_url, logger=logger)


def enqueue_fpgrowth_jobs(jobs: List[Dict[str, Any]], queue_url: str = QUEUE_URL, logger=None) -> int:
    """
    Enqueue FP-Growth jobs to the existing SQS FIFO queue.
    
    Args:
        jobs: List of job dictionaries
        queue_url: URL of the SQS FIFO queue
        logger: Logger instance
        
    Returns:
        Number of jobs enqueued
    """
    queue = create_fpgrowth_job_queue(queue_url, logger)
    return queue.enqueue_jobs(jobs)


def process_fpgrowth_jobs_from_queue(
    job_processor_func,
    queue_url: str = QUEUE_URL,
    max_concurrent: int = 4,
    logger=None
) -> List[Dict[str, Any]]:
    """
    Process FP-Growth jobs from the existing SQS FIFO queue with controlled concurrency.
    
    Args:
        job_processor_func: Function to process individual jobs
        queue_url: URL of the SQS FIFO queue
        max_concurrent: Maximum concurrent jobs
        logger: Logger instance
        
    Returns:
        List of job results
    """
    queue = create_fpgrowth_job_queue(queue_url, logger)
    results = []
    
    # Get queue attributes for monitoring
    attributes = queue.get_queue_attributes()
    approximate_number_of_messages = int(attributes.get('ApproximateNumberOfMessages', 0))
    
    if approximate_number_of_messages == 0:
        logger.info("No jobs in FIFO queue")
        return results
    
    logger.info(f"Processing {approximate_number_of_messages} jobs from FIFO queue (max concurrent: {max_concurrent})")
    
    # Process jobs with true concurrency using ProcessPoolExecutor
    completed_jobs = 0

    with ProcessPoolExecutor(max_workers=max_concurrent) as executor:
        # Submit initial batch of jobs
        futures = {}
        active_jobs = 0

        while True:
            # Submit new jobs if we have capacity
            while active_jobs < max_concurrent:
                # Dequeue a single job
                jobs = queue.dequeue_jobs(max_messages=1, wait_time=5)

                if not jobs:
                    # No more jobs in queue
                    if active_jobs == 0:
                        break
                    else:
                        # Wait for active jobs to complete
                        time.sleep(1)
                        break

                job = jobs[0]
                # Submit job to process pool
                future = executor.submit(process_single_job, job, job_processor_func, queue_url, SQS_REGION, "fpgrowth_worker")
                futures[future] = job
                active_jobs += 1

                logger.info(f"Started job {job['age_band']}/{job['event_year']} (Active: {active_jobs})")

            # Check for completed jobs
            for future in as_completed(futures, timeout=1):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        completed_jobs += 1
                        logger.info(f"[{completed_jobs}] Completed job {result['age_band']}/{result['event_year']} (Active: {active_jobs - 1})")
                except Exception as e:
                    job = futures[future]
                    logger.error(f"Failed to process job {job['age_band']}/{job['event_year']}: {e}")

                # Remove completed job from futures
                del futures[future]
                active_jobs -= 1

                # Break out of the loop to submit new jobs
                break

            # If no jobs completed, wait a bit
            if active_jobs >= max_concurrent:
                time.sleep(0.1)
    
    logger.info(f"Completed processing {completed_jobs} jobs from FIFO queue")
    return results


def process_single_job(job: Dict[str, Any], job_processor_func, queue_url, region, logger_name) -> Dict[str, Any]:
    """
    Process a single job and handle cleanup.
    This function is designed to work with ProcessPoolExecutor.

    Args:
        job: Job dictionary with receipt_handle
        job_processor_func: Function to process the job
        queue_url: SQS queue URL
        region: AWS region
        logger_name: Name for the logger

    Returns:
        Job result with age_band and event_year added
    """
    # Create fresh logger for this process
    import logging
    logger = logging.getLogger(logger_name)

    # Create fresh queue instance for this process
    queue = create_fpgrowth_job_queue(queue_url, logger)

    try:
        # Process job
        result = job_processor_func(job)
        result['age_band'] = job['age_band']
        result['event_year'] = job['event_year']

        # Delete message from queue
        queue.delete_message(job['receipt_handle'])

        return result

    except Exception as e:
        logger.error(f"Failed to process job {job['age_band']}/{job['event_year']}: {e}")
        # Don't delete message - let it be retried
        return None


def create_fpgrowth_job_message(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a properly formatted message for the FIFO queue.
    
    Args:
        job: Job dictionary with age_band and event_year
        
    Returns:
        Formatted message dictionary
    """
    # Create cohort from age_band and event_year
    cohort = f"{job['age_band']}-{job['event_year']}"
    
    message = {
        "QueueUrl": QUEUE_URL,
        "MessageBody": json.dumps(job)
    }
    
    # Add FIFO-specific attributes
    if QUEUE_URL.endswith(".fifo"):
        message["MessageGroupId"] = f"group-{cohort}"
        message["MessageDeduplicationId"] = cohort
    
    return message


def enqueue_jobs_with_existing_format(jobs: List[Dict[str, Any]], logger=None) -> int:
    """
    Enqueue jobs using the existing message format and queue.
    
    Args:
        jobs: List of job dictionaries with age_band and event_year
        logger: Logger instance
        
    Returns:
        Number of jobs enqueued
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    sqs = boto3.client("sqs", region_name=SQS_REGION)
    count = 0
    
    for job in jobs:
        try:
            message = create_fpgrowth_job_message(job)
            response = sqs.send_message(**message)
            
            logger.info(f"→ Pushed job: {job['age_band']}/{job['event_year']} - {response['MessageId']}")
            count += 1
            
        except Exception as e:
            logger.error(f"Failed to enqueue job {job['age_band']}/{job['event_year']}: {e}")
    
    logger.info(f"\n✓ Done. {count} FP-Growth jobs queued to SQS FIFO.")
    return count 