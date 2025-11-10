#!/usr/bin/env python3
"""
Signal 15 (SIGTERM) Debugging Script

This script helps debug Signal 15 issues by providing detailed logging,
resource monitoring, and stack trace capture when signals are received.
"""

import os
import sys
import signal
import time
import traceback
import psutil
import logging
from datetime import datetime

# Set up project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from helpers_1997_13.aws_utils import enable_signal_debugging, get_system_resource_status

def setup_debug_logging():
    """Set up detailed logging for signal debugging"""
    logger = logging.getLogger('signal_debug')
    logger.setLevel(logging.DEBUG)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(f'signal_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def monitor_resources(logger, interval=30):
    """Monitor system resources continuously"""
    logger.info("🔍 Starting resource monitoring...")
    
    while True:
        try:
            status = get_system_resource_status()
            logger.info(f"📊 Resource Status: {status}")
            
            # Check for potential issues
            if status.get('process_memory_mb', 0) > 1000:  # > 1GB
                logger.warning(f"⚠️ High memory usage: {status['process_memory_mb']:.2f} MB")
            
            if status.get('system_memory_percent', 0) > 90:
                logger.error(f"🚨 Critical system memory: {status['system_memory_percent']:.1f}%")
            
            if status.get('open_files', 0) > 1000:
                logger.warning(f"⚠️ High file descriptor count: {status['open_files']}")
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            logger.info("🔍 Resource monitoring stopped")
            break
        except Exception as e:
            logger.error(f"🔍 Error in resource monitoring: {e}")

def create_signal_test(logger):
    """Create a test that can trigger Signal 15 for debugging"""
    logger.info("🧪 Creating signal test environment...")
    
    def memory_intensive_task():
        """Task that uses a lot of memory to potentially trigger OOM"""
        logger.info("🧪 Starting memory-intensive task...")
        large_list = []
        for i in range(1000000):  # 1M items
            large_list.append(f"item_{i}" * 100)  # Large strings
            if i % 100000 == 0:
                logger.info(f"🧪 Created {i} items, memory: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
        return len(large_list)
    
    def file_intensive_task():
        """Task that opens many files"""
        logger.info("🧪 Starting file-intensive task...")
        files = []
        for i in range(1000):
            try:
                f = open(f"temp_file_{i}.txt", "w")
                files.append(f)
                f.write(f"Test content {i}")
            except Exception as e:
                logger.error(f"🧪 Error creating file {i}: {e}")
                break
        return len(files)
    
    return memory_intensive_task, file_intensive_task

def main():
    """Main debugging function"""
    print("🔍 Signal 15 Debugging Tool")
    print("=" * 50)
    
    # Set up logging
    logger = setup_debug_logging()
    logger.info("🔍 Signal debugging tool started")
    
    # Enable signal debugging
    enable_signal_debugging(logger)
    
    # Get initial system status
    initial_status = get_system_resource_status()
    logger.info(f"📊 Initial system status: {initial_status}")
    
    # Create test functions
    memory_task, file_task = create_signal_test(logger)
    
    print("\n🔍 Available debugging options:")
    print("1. Monitor resources continuously")
    print("2. Run memory-intensive task (may trigger OOM)")
    print("3. Run file-intensive task (may trigger limits)")
    print("4. Run your actual FP-Growth pipeline with debugging")
    print("5. Exit")
    
    while True:
        try:
            choice = input("\n🔍 Enter your choice (1-5): ").strip()
            
            if choice == "1":
                logger.info("🔍 Starting resource monitoring...")
                monitor_resources(logger)
                
            elif choice == "2":
                logger.info("🧪 Running memory-intensive task...")
                try:
                    result = memory_task()
                    logger.info(f"🧪 Memory task completed: {result} items")
                except Exception as e:
                    logger.error(f"🧪 Memory task failed: {e}")
                    
            elif choice == "3":
                logger.info("🧪 Running file-intensive task...")
                try:
                    result = file_task()
                    logger.info(f"🧪 File task completed: {result} files")
                except Exception as e:
                    logger.error(f"🧪 File task failed: {e}")
                    
            elif choice == "4":
                logger.info("🔍 Running FP-Growth pipeline with debugging...")
                # Import and run your actual pipeline
                try:
                    from run_fpgrowth import execute_parallel_global_fpgrowth
                    logger.info("🔍 Starting FP-Growth pipeline...")
                    execute_parallel_global_fpgrowth(num_workers=2)  # Reduced workers for debugging
                except Exception as e:
                    logger.error(f"🔍 FP-Growth pipeline failed: {e}")
                    logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
                    
            elif choice == "5":
                logger.info("🔍 Exiting debugging tool")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            logger.info("🔍 Interrupted by user")
            break
        except Exception as e:
            logger.error(f"🔍 Unexpected error: {e}")
            logger.error(f"🔍 Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 