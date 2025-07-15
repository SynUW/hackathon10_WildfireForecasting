#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Parallel Trainer - Simplified
Automatically monitors model completion status and intelligently starts the next task
"""

import subprocess
import time
import os
import signal
import sys
from datetime import datetime
import pandas as pd
import argparse

# =============================================================================
# Configuration Parameters
# =============================================================================

# Import configuration parameters
from train_all_models_combined import (
    DEFAULT_MAX_PARALLEL_PER_GPU,
    get_all_models,
    is_model_trained,  # Add checkpoint resume function
    filter_trained_models  # Add checkpoint resume function
)

class SmartParallelTrainer:
    """Smart Parallel Trainer"""
    
    def __init__(self, max_parallel_per_gpu=2):
        self.max_parallel_per_gpu = max_parallel_per_gpu
        # Generate a log directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = f"./trash/smart_parallel_logs_{timestamp}"
        self.running_tasks = {}  # {task_id: task_info}
        self.completed_tasks = []
        self.failed_tasks = []
        self.task_queue = []
        self.gpu_counts = {0: 0, 1: 0}  # GPU task count
        self.task_counter = 0
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"📁 Log directory: {self.log_dir}")
        
        # Set signal handling
        signal.signal(signal.SIGINT, self.stop_all)
        signal.signal(signal.SIGTERM, self.stop_all)
    
    def add_tasks(self, model_configs):
        """Add training tasks"""
        for config in model_configs:
            task = {
                'model': config['model'],
                'type': config.get('type', 'standard'),
                'gpu': config.get('gpu', 0),
                'id': f"task_{self.task_counter}",
                'status': 'pending'
            }
            self.task_queue.append(task)
            self.task_counter += 1
            print(f"📋 Added task: {task['model']}({task['type']}) -> GPU{task['gpu']}")
    
    def get_available_gpu(self, preferred_gpu):
        """Get available GPU"""
        # First check preferred GPU
        if self.gpu_counts[preferred_gpu] < self.max_parallel_per_gpu:
            return preferred_gpu
        
        # Check other GPUs
        for gpu_id in [0, 1]:
            if self.gpu_counts[gpu_id] < self.max_parallel_per_gpu:
                return gpu_id
        
        return None
    
    def start_task(self, task):
        """Start a single task"""
        available_gpu = self.get_available_gpu(task['gpu'])
        if available_gpu is None:
            return False
        
        task['gpu'] = available_gpu  # May reassign GPU
        
        # Prepare command
        cmd = [
            "python", "train_single_model.py",
            "--model", task['model'],
            "--type", task['type'],
            "--gpu", str(task['gpu']),
            "--log-dir", self.log_dir
        ]
        
        # Log file
        log_file = os.path.join(self.log_dir, f"{task['model']}_{task['type']}_gpu{task['gpu']}.log")
        
        try:
            # Start process
            with open(log_file, 'w') as f:
                process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            
            # Record task information
            task.update({
                'process': process,
                'pid': process.pid,
                'start_time': datetime.now(),
                'log_file': log_file,
                'status': 'running'
            })
            
            self.running_tasks[task['id']] = task
            self.gpu_counts[task['gpu']] += 1
            
            print(f"🚀 Started: {task['model']}({task['type']}) on GPU{task['gpu']} (PID: {task['pid']})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start: {task['model']} - {e}")
            return False
    
    def check_completed_tasks(self):
        """Check completed tasks"""
        completed_ids = []
        
        for task_id, task in self.running_tasks.items():
            return_code = task['process'].poll()
            if return_code is not None:
                # Task completed
                end_time = datetime.now()
                duration = end_time - task['start_time']
                
                # Update GPU count
                self.gpu_counts[task['gpu']] -= 1
                
                if return_code == 0:
                    task['status'] = 'completed'
                    self.completed_tasks.append(task)
                    print(f"✅ Completed: {task['model']}({task['type']}) Duration: {duration}")
                else:
                    task['status'] = 'failed'
                    self.failed_tasks.append(task)
                    print(f"❌ Failed: {task['model']}({task['type']}) Return code: {return_code}")
                
                completed_ids.append(task_id)
        
        # Remove completed tasks
        for task_id in completed_ids:
            del self.running_tasks[task_id]
    
    def start_pending_tasks(self):
        """Start pending tasks"""
        started_count = 0
        remaining_tasks = []
        
        for task in self.task_queue:
            if task['status'] == 'pending':
                if self.start_task(task):
                    started_count += 1
                else:
                    remaining_tasks.append(task)
            else:
                remaining_tasks.append(task)
        
        self.task_queue = remaining_tasks
        return started_count
    
    def print_status(self):
        """Print status"""
        pending = len([t for t in self.task_queue if t['status'] == 'pending'])
        running = len(self.running_tasks)
        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)
        
        print(f"\r📊 Pending({pending}) Running({running}) Completed({completed}) Failed({failed}) | "
              f"GPU0: {self.gpu_counts[0]}/{self.max_parallel_per_gpu} "
              f"GPU1: {self.gpu_counts[1]}/{self.max_parallel_per_gpu}", end="", flush=True)
    
    def run(self):
        """Run the training queue"""
        print(f"🚀 Smart Parallel Trainer started")
        print(f"📋 Total tasks: {len(self.task_queue)}")
        print(f"⚙️ Max parallel per GPU: {self.max_parallel_per_gpu}")
        print("💡 Press Ctrl+C to stop all tasks\n")
        
        try:
            while True:
                # Check completed tasks
                self.check_completed_tasks()
                
                # Start new tasks
                self.start_pending_tasks()
                
                # Print status
                self.print_status()
                
                # Check if all tasks are completed
                if (len(self.task_queue) == 0 and 
                    len(self.running_tasks) == 0 and 
                    (len(self.completed_tasks) + len(self.failed_tasks)) > 0):
                    break
                
                time.sleep(15)  # Check every 15 seconds, reduce check frequency
            
            print(f"\n\n🎉 All tasks completed!")
            self.print_summary()
            
        except KeyboardInterrupt:
            print(f"\n⏹️ Received stop signal...")
            self.stop_all()
    
    def stop_all(self, signum=None, frame=None):
        """Stop all tasks"""
        print(f"\n🛑 Stopping all running tasks...")
        
        for task in self.running_tasks.values():
            try:
                task['process'].terminate()
                print(f"🛑 Stopped: {task['model']}({task['type']})")
            except:
                pass
        
        # Wait for processes to end
        time.sleep(2)
        
        # Force kill
        for task in self.running_tasks.values():
            try:
                task['process'].kill()
            except:
                pass
        
        if signum:
            sys.exit(0)
    
    def test_completed_models(self):
        """Collect and summarize test results for all successfully completed models"""
        if not self.completed_tasks:
            print("⚠️ No successfully completed models to collect results for")
            return
            
        print(f"\n📊 Collecting test results for {len(self.completed_tasks)} completed models...")
        
        # Collect standard model results
        standard_results = {}
        
        for task in self.completed_tasks:
            model_name = task['model']
            model_type = task['type']
            
            # Check corresponding CSV result file
            csv_file = os.path.join(self.log_dir, f"{model_name}_{model_type}_results.csv")
            
            if os.path.exists(csv_file):
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file)
                    
                    if len(df) > 0:
                        # Convert DataFrame to dictionary format
                        row = df.iloc[0]
                        model_result = {
                            'f1': {
                                'precision': self._safe_float(row.get('best_f1_precision')),
                                'recall': self._safe_float(row.get('best_f1_recall')),
                                'f1': self._safe_float(row.get('best_f1_f1')),
                                'pr_auc': self._safe_float(row.get('best_f1_pr_auc'))
                            },
                            'recall': {
                                'precision': self._safe_float(row.get('best_recall_precision')),
                                'recall': self._safe_float(row.get('best_recall_recall')),
                                'f1': self._safe_float(row.get('best_recall_f1')),
                                'pr_auc': self._safe_float(row.get('best_recall_pr_auc'))
                            },
                            'pr_auc': {
                                'precision': self._safe_float(row.get('best_pr_auc_precision')),
                                'recall': self._safe_float(row.get('best_pr_auc_recall')),
                                'f1': self._safe_float(row.get('best_pr_auc_f1')),
                                'pr_auc': self._safe_float(row.get('best_pr_auc_pr_auc'))
                            },
                            'final_epoch': {
                                'precision': self._safe_float(row.get('final_epoch_precision')),
                                'recall': self._safe_float(row.get('final_epoch_recall')),
                                'f1': self._safe_float(row.get('final_epoch_f1')),
                                'pr_auc': self._safe_float(row.get('final_epoch_pr_auc'))
                            }
                        }
                        
                        standard_results[model_name] = model_result
                        
                        print(f"✅ Collected test results for {model_name}({model_type})")
                    else:
                        print(f"⚠️ CSV file is empty for {model_name}({model_type}): {csv_file}")
                        
                except Exception as e:
                    print(f"❌ Failed to read CSV file {csv_file}: {e}")
            else:
                print(f"❌ CSV result file not found: {csv_file}")
        
        # Generate combined CSV file
        if standard_results:
            self._save_combined_csv(standard_results, 'standard')
        
        print(f"\n📊 Result collection complete!")
        print(f"   Standard models: {len(standard_results)}")
        
        return len(standard_results)
    
    def _safe_float(self, value):
        """Safely convert to float, handling N/A values"""
        if pd.isna(value) or value == 'N/A':
            return None
        try:
            return float(value)
        except:
            return None
    
    def _save_combined_csv(self, results, model_type):
        """Save combined CSV results"""
        if not results:
            return
        
        # Prepare CSV data
        csv_data = []
        
        # CSV column names
        columns = ['Model']
        metric_types = ['f1', 'recall', 'pr_auc', 'final_epoch']
        metric_names = ['precision', 'recall', 'f1', 'pr_auc']
        
        for metric_type in metric_types:
            for metric_name in metric_names:
                display_type = "final_epoch" if metric_type == 'final_epoch' else f"best_{metric_type}"
                columns.append(f"{display_type}_{metric_name}")
        
        # Add data rows
        for model_name, model_results in results.items():
            row = [model_name]
            
            for metric_type in metric_types:
                for metric_name in metric_names:
                    value = model_results[metric_type][metric_name]
                    if value is not None:
                        row.append(f"{value:.6f}")
                    else:
                        row.append("N/A")
            
            csv_data.append(row)
        
        # Save to CSV file
        import pandas as pd
        df = pd.DataFrame(csv_data, columns=columns)
        
        csv_filename = f"combined_results_{model_type}.csv"
        csv_path = os.path.join(self.log_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        
        print(f"📊 {model_type} model summary results saved to: {csv_path}")
        print(f"   Total {len(csv_data)} model test results included")
        print(f"   CSV format: Model name + 4 save versions × 4 metrics = {len(columns)} columns")
    
    def print_summary(self):
        """Print summary"""
        total = len(self.completed_tasks) + len(self.failed_tasks)
        success_rate = len(self.completed_tasks) / total * 100 if total > 0 else 0
        
        print(f"📊 Execution Summary:")
        print(f"   Total tasks: {total}")
        print(f"   Success: {len(self.completed_tasks)}")
        print(f"   Failed: {len(self.failed_tasks)}")
        print(f"   Success rate: {success_rate:.1f}%")
        
        if self.completed_tasks:
            print(f"\n✅ Successfully completed models:")
            for task in self.completed_tasks:
                duration = task.get('end_time', datetime.now()) - task['start_time']
                print(f"   {task['model']}({task['type']}) - {duration}")
        
        if self.failed_tasks:
            print(f"\n❌ Failed models:")
            for task in self.failed_tasks:
                print(f"   {task['model']}({task['type']})")
        
        # Automatically test completed models
        if self.completed_tasks:
            self.test_completed_models()

def get_all_available_models():
    """Get all available models, filtering out models that cannot be loaded"""
    # Use functions from train_all_models_combined.py to get model list
    models = get_all_models('model_zoo')
    
    # Filter out models requiring special dependencies
    available_models = []
    for model_name in models:
        if model_name == 'Mamba':
            try:
                import mamba_ssm
                available_models.append(model_name)
                print(f"✅ {model_name} available (mamba_ssm installed)")
            except ImportError:
                print(f"⚠️ Skipping {model_name} (mamba_ssm library missing)")
                continue
        else:
            available_models.append(model_name)
    
    return sorted(available_models)

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Smart Parallel Trainer')
    
    parser.add_argument('--max-parallel', type=int, default=DEFAULT_MAX_PARALLEL_PER_GPU,
                       help=f'Max parallel tasks per GPU (default: {DEFAULT_MAX_PARALLEL_PER_GPU})')
    parser.add_argument('--models', nargs='+',
                       help='Specify a list of models to train (if not specified, all models are trained)')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retrain all models, ignoring existing model files')
    
    args = parser.parse_args()
    
    print("🚀 Smart Parallel Trainer")
    print("💡 Automatically skips models that have already been trained (checks final_epoch.pth files)")
    print("    Use --force-retrain to force retrain all models")
    print()
    
    # Display configuration status
    from train_all_models_combined import print_config_status
    print_config_status()
    print()
    
    # Create trainer
    trainer = SmartParallelTrainer(max_parallel_per_gpu=args.max_parallel)
    
    # Get all available models
    all_models = get_all_available_models()
    
    # If specific models are specified, use only those models
    if args.models:
        specified_models = [m for m in args.models if m in all_models]
        missing_models = [m for m in args.models if m not in all_models]
        if missing_models:
            print(f"⚠️ The following models are not available: {', '.join(missing_models)}")
        all_models = specified_models
    
    print(f"🔍 Found {len(all_models)} available models: {', '.join(all_models)}")
    
    # Train standard models
    train_standard = True
    
    print("📋 Training Plan: Standard Models ✅")
    
    # Generate model configurations
    model_configs = []
    
    # First batch: Core models (prioritize training)
    priority_models = ["DLinear", "CrossLinear", "iTransformer", "s_mamba"]
    
    # Initialize statistics
    models_to_train_standard = []
    total_skipped = 0
    
    # Add standard models - use checkpoint resume functionality
    if train_standard:
        # Filter already trained standard models
        models_to_train_standard, trained_models_standard = filter_trained_models(all_models, 'standard', force_retrain=args.force_retrain)
        
        # Add priority models
        for i, model in enumerate(priority_models):
            if model in models_to_train_standard:
                model_configs.append({
                    "model": model,
                    "type": "standard", 
                    "gpu": i % 2,  # Alternate GPU assignment
                    "priority": 1
                })
        
        # Add other standard models
        other_models = [m for m in models_to_train_standard if m not in priority_models]
        for i, model in enumerate(other_models):
            model_configs.append({
                "model": model,
                "type": "standard",
                "gpu": i % 2,  # Alternate GPU assignment
                "priority": 2
            })
    

    
    print(f"📋 Total {len(model_configs)} training tasks configured")
    
    # Statistics
    if train_standard:
        standard_count = len([c for c in model_configs if c['type'] == 'standard'])
        skipped_standard = len(all_models) - len(models_to_train_standard)
        total_skipped += skipped_standard
        print(f"   Standard models: {standard_count} to train, {skipped_standard} completed")
    
    if total_skipped > 0 and not args.force_retrain:
        print(f"💡 Skipped {total_skipped} already trained models (use --force-retrain to force retrain)")
    elif args.force_retrain:
        print(f"🔄 Force retrain mode: will train all models")
    
    if not model_configs:
        print("❌ No models to train, please check configuration")
        return
    
    # Add tasks and run
    trainer.add_tasks(model_configs)
    trainer.run()

if __name__ == "__main__":
    main() 