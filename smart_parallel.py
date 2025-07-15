#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能并行训练器 - 简化版
自动监测模型完成状态，智能启动下一个任务
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
# 配置参数
# =============================================================================

# 导入配置参数
from train_all_models_combined import (
    ENABLE_10X_TRAINING, 
    DEFAULT_MAX_PARALLEL_PER_GPU,
    get_all_models,
    is_model_trained,  # 添加断点续传函数
    filter_trained_models  # 添加断点续传函数
)

class SmartParallelTrainer:
    """智能并行训练器"""
    
    def __init__(self, max_parallel_per_gpu=2):
        self.max_parallel_per_gpu = max_parallel_per_gpu
        # 生成带时间戳的日志目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = f"./trash/smart_parallel_logs_{timestamp}"
        self.running_tasks = {}  # {task_id: task_info}
        self.completed_tasks = []
        self.failed_tasks = []
        self.task_queue = []
        self.gpu_counts = {0: 0, 1: 0}  # GPU任务计数
        self.task_counter = 0
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        print(f"📁 日志目录: {self.log_dir}")
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.stop_all)
        signal.signal(signal.SIGTERM, self.stop_all)
    
    def add_tasks(self, model_configs):
        """添加训练任务"""
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
            print(f"📋 添加任务: {task['model']}({task['type']}) -> GPU{task['gpu']}")
    
    def get_available_gpu(self, preferred_gpu):
        """获取可用的GPU"""
        # 先检查首选GPU
        if self.gpu_counts[preferred_gpu] < self.max_parallel_per_gpu:
            return preferred_gpu
        
        # 检查其他GPU
        for gpu_id in [0, 1]:
            if self.gpu_counts[gpu_id] < self.max_parallel_per_gpu:
                return gpu_id
        
        return None
    
    def start_task(self, task):
        """启动单个任务"""
        available_gpu = self.get_available_gpu(task['gpu'])
        if available_gpu is None:
            return False
        
        task['gpu'] = available_gpu  # 可能重新分配GPU
        
        # 准备命令
        cmd = [
            "python", "train_single_model.py",
            "--model", task['model'],
            "--type", task['type'],
            "--gpu", str(task['gpu']),
            "--log-dir", self.log_dir
        ]
        
        # 日志文件
        log_file = os.path.join(self.log_dir, f"{task['model']}_{task['type']}_gpu{task['gpu']}.log")
        
        try:
            # 启动进程
            with open(log_file, 'w') as f:
                process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            
            # 记录任务信息
            task.update({
                'process': process,
                'pid': process.pid,
                'start_time': datetime.now(),
                'log_file': log_file,
                'status': 'running'
            })
            
            self.running_tasks[task['id']] = task
            self.gpu_counts[task['gpu']] += 1
            
            print(f"🚀 启动: {task['model']}({task['type']}) on GPU{task['gpu']} (PID: {task['pid']})")
            return True
            
        except Exception as e:
            print(f"❌ 启动失败: {task['model']} - {e}")
            return False
    
    def check_completed_tasks(self):
        """检查已完成的任务"""
        completed_ids = []
        
        for task_id, task in self.running_tasks.items():
            return_code = task['process'].poll()
            if return_code is not None:
                # 任务完成
                end_time = datetime.now()
                duration = end_time - task['start_time']
                
                # 更新GPU计数
                self.gpu_counts[task['gpu']] -= 1
                
                if return_code == 0:
                    task['status'] = 'completed'
                    self.completed_tasks.append(task)
                    print(f"✅ 完成: {task['model']}({task['type']}) 耗时: {duration}")
                else:
                    task['status'] = 'failed'
                    self.failed_tasks.append(task)
                    print(f"❌ 失败: {task['model']}({task['type']}) 返回码: {return_code}")
                
                completed_ids.append(task_id)
        
        # 移除已完成的任务
        for task_id in completed_ids:
            del self.running_tasks[task_id]
    
    def start_pending_tasks(self):
        """启动等待中的任务"""
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
        """打印状态"""
        pending = len([t for t in self.task_queue if t['status'] == 'pending'])
        running = len(self.running_tasks)
        completed = len(self.completed_tasks)
        failed = len(self.failed_tasks)
        
        print(f"\r📊 等待({pending}) 运行({running}) 完成({completed}) 失败({failed}) | "
              f"GPU0: {self.gpu_counts[0]}/{self.max_parallel_per_gpu} "
              f"GPU1: {self.gpu_counts[1]}/{self.max_parallel_per_gpu}", end="", flush=True)
    
    def run(self):
        """运行训练队列"""
        print(f"🚀 智能并行训练器启动")
        print(f"📋 总任务数: {len(self.task_queue)}")
        print(f"⚙️ 每GPU最大并行: {self.max_parallel_per_gpu}")
        print("💡 按 Ctrl+C 停止所有任务\n")
        
        try:
            while True:
                # 检查完成的任务
                self.check_completed_tasks()
                
                # 启动新任务
                self.start_pending_tasks()
                
                # 打印状态
                self.print_status()
                
                # 检查是否全部完成
                if (len(self.task_queue) == 0 and 
                    len(self.running_tasks) == 0 and 
                    (len(self.completed_tasks) + len(self.failed_tasks)) > 0):
                    break
                
                time.sleep(15)  # 每15秒检查一次，减少检查频率
            
            print(f"\n\n🎉 所有任务完成!")
            self.print_summary()
            
        except KeyboardInterrupt:
            print(f"\n⏹️ 收到停止信号...")
            self.stop_all()
    
    def stop_all(self, signum=None, frame=None):
        """停止所有任务"""
        print(f"\n🛑 停止所有运行中的任务...")
        
        for task in self.running_tasks.values():
            try:
                task['process'].terminate()
                print(f"🛑 停止: {task['model']}({task['type']})")
            except:
                pass
        
        # 等待进程结束
        time.sleep(2)
        
        # 强制杀死
        for task in self.running_tasks.values():
            try:
                task['process'].kill()
            except:
                pass
        
        if signum:
            sys.exit(0)
    
    def test_completed_models(self):
        """收集并汇总所有成功完成的模型的测试结果"""
        if not self.completed_tasks:
            print("⚠️ 没有成功完成的模型需要收集结果")
            return
            
        print(f"\n📊 收集 {len(self.completed_tasks)} 个已完成模型的测试结果...")
        
        # 分别收集标准模型和10x模型的结果
        standard_results = {}
        tenx_results = {}
        
        for task in self.completed_tasks:
            model_name = task['model']
            model_type = task['type']
            
            # 检查对应的CSV结果文件
            csv_file = os.path.join(self.log_dir, f"{model_name}_{model_type}_results.csv")
            
            if os.path.exists(csv_file):
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file)
                    
                    if len(df) > 0:
                        # 将DataFrame转换为字典格式
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
                        
                        if model_type == 'standard':
                            standard_results[model_name] = model_result
                        else:
                            tenx_results[model_name] = model_result
                        
                        print(f"✅ 收集 {model_name}({model_type}) 的测试结果")
                    else:
                        print(f"⚠️ {model_name}({model_type}) 的CSV文件为空")
                        
                except Exception as e:
                    print(f"❌ 读取CSV文件失败 {csv_file}: {e}")
            else:
                print(f"❌ 未找到CSV结果文件: {csv_file}")
        
        # 生成汇总的CSV文件
        if standard_results:
            self._save_combined_csv(standard_results, 'standard')
        if tenx_results:
            self._save_combined_csv(tenx_results, '10x')
        
        print(f"\n📊 结果收集完成!")
        print(f"   标准模型: {len(standard_results)} 个")
        print(f"   10x模型: {len(tenx_results)} 个")
        
        return len(standard_results) + len(tenx_results)
    
    def _safe_float(self, value):
        """安全地转换为浮点数，处理N/A值"""
        if pd.isna(value) or value == 'N/A':
            return None
        try:
            return float(value)
        except:
            return None
    
    def _save_combined_csv(self, results, model_type):
        """保存合并的CSV结果"""
        if not results:
            return
        
        # 准备CSV数据
        csv_data = []
        
        # CSV列名
        columns = ['Model']
        metric_types = ['f1', 'recall', 'pr_auc', 'final_epoch']
        metric_names = ['precision', 'recall', 'f1', 'pr_auc']
        
        for metric_type in metric_types:
            for metric_name in metric_names:
                display_type = "final_epoch" if metric_type == 'final_epoch' else f"best_{metric_type}"
                columns.append(f"{display_type}_{metric_name}")
        
        # 添加数据行
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
        
        # 保存到CSV文件
        import pandas as pd
        df = pd.DataFrame(csv_data, columns=columns)
        
        csv_filename = f"combined_results_{model_type}.csv"
        csv_path = os.path.join(self.log_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        
        print(f"📊 {model_type}模型汇总结果已保存到: {csv_path}")
        print(f"   共包含 {len(csv_data)} 个模型的测试结果")
        print(f"   CSV格式: 模型名 + 4种保存版本 × 4个指标 = {len(columns)} 列")
    
    def print_summary(self):
        """打印摘要"""
        total = len(self.completed_tasks) + len(self.failed_tasks)
        success_rate = len(self.completed_tasks) / total * 100 if total > 0 else 0
        
        print(f"📊 执行摘要:")
        print(f"   总任务: {total}")
        print(f"   成功: {len(self.completed_tasks)}")
        print(f"   失败: {len(self.failed_tasks)}")
        print(f"   成功率: {success_rate:.1f}%")
        
        if self.completed_tasks:
            print(f"\n✅ 成功完成的模型:")
            for task in self.completed_tasks:
                duration = task.get('end_time', datetime.now()) - task['start_time']
                print(f"   {task['model']}({task['type']}) - {duration}")
        
        if self.failed_tasks:
            print(f"\n❌ 失败的模型:")
            for task in self.failed_tasks:
                print(f"   {task['model']}({task['type']})")
        
        # 自动测试完成的模型
        if self.completed_tasks:
            self.test_completed_models()

def get_all_available_models():
    """获取所有可用的模型，过滤掉无法加载的模型"""
    # 使用train_all_models_combined.py中的函数获取模型列表
    models = get_all_models('model_zoo')
    
    # 过滤掉需要特殊依赖的模型
    available_models = []
    for model_name in models:
        if model_name == 'Mamba':
            try:
                import mamba_ssm
                available_models.append(model_name)
                print(f"✅ {model_name} 可用 (mamba_ssm已安装)")
            except ImportError:
                print(f"⚠️ 跳过 {model_name} (缺少mamba_ssm库)")
                continue
        else:
            available_models.append(model_name)
    
    return sorted(available_models)

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='智能并行训练器')
    parser.add_argument('--skip-10x', action='store_true', 
                       help='跳过10x模型训练，只训练标准模型')
    parser.add_argument('--only-10x', action='store_true',
                       help='只训练10x模型，跳过标准模型')
    parser.add_argument('--max-parallel', type=int, default=DEFAULT_MAX_PARALLEL_PER_GPU,
                       help=f'每个GPU的最大并行任务数 (默认: {DEFAULT_MAX_PARALLEL_PER_GPU})')
    parser.add_argument('--models', nargs='+',
                       help='指定要训练的模型列表 (如果不指定则训练所有模型)')
    parser.add_argument('--force-retrain', action='store_true',
                       help='强制重新训练所有模型，忽略已存在的模型文件')
    
    args = parser.parse_args()
    
    print("🚀 智能并行训练器")
    print("💡 支持自动跳过已训练完成的模型 (检查final_epoch.pth文件)")
    print("   使用 --force-retrain 可强制重新训练所有模型")
    print()
    
    # 显示配置状态
    from train_all_models_combined import print_config_status
    print_config_status()
    print()
    
    # 创建训练器
    trainer = SmartParallelTrainer(max_parallel_per_gpu=args.max_parallel)
    
    # 获取所有可用模型
    all_models = get_all_available_models()
    
    # 如果指定了特定模型，则只使用指定的模型
    if args.models:
        specified_models = [m for m in args.models if m in all_models]
        missing_models = [m for m in args.models if m not in all_models]
        if missing_models:
            print(f"⚠️ 以下模型不可用: {', '.join(missing_models)}")
        all_models = specified_models
    
    print(f"🔍 发现 {len(all_models)} 个可用模型: {', '.join(all_models)}")
    
    # 根据配置和命令行参数决定训练哪些模型
    train_standard = not args.only_10x
    train_10x = ENABLE_10X_TRAINING and not args.skip_10x
    
    if args.skip_10x:
        print("📋 已选择跳过10x模型训练")
        train_10x = False
    elif args.only_10x:
        print("📋 已选择只训练10x模型")
        train_standard = False
    elif not ENABLE_10X_TRAINING:
        print("⚠️ 10x模型训练已在配置中禁用")
        train_10x = False
    
    print(f"📋 训练计划: 标准模型={'✅' if train_standard else '❌'}, 10x模型={'✅' if train_10x else '❌'}")
    
    # 生成模型配置
    model_configs = []
    
    # 第一批：核心模型（优先训练）
    priority_models = ["DLinear", "CrossLinear", "iTransformer", "s_mamba"]
    
    # 初始化统计变量
    models_to_train_standard = []
    models_to_train_10x = []
    total_skipped = 0
    
    # 添加标准模型 - 使用断点续传功能
    if train_standard:
        # 过滤已训练的标准模型
        models_to_train_standard, trained_models_standard = filter_trained_models(all_models, 'standard', force_retrain=args.force_retrain)
        
        # 添加优先模型
        for i, model in enumerate(priority_models):
            if model in models_to_train_standard:
                model_configs.append({
                    "model": model,
                    "type": "standard", 
                    "gpu": i % 2,  # 轮流分配GPU
                    "priority": 1
                })
        
        # 添加其他标准模型
        other_models = [m for m in models_to_train_standard if m not in priority_models]
        for i, model in enumerate(other_models):
            model_configs.append({
                "model": model,
                "type": "standard",
                "gpu": i % 2,  # 轮流分配GPU
                "priority": 2
            })
    
    # 添加10x模型 - 使用断点续传功能
    if train_10x:
        # 过滤已训练的10x模型
        models_to_train_10x, trained_models_10x = filter_trained_models(all_models, '10x', force_retrain=args.force_retrain)
        
        for i, model in enumerate(models_to_train_10x):
            model_configs.append({
                "model": model,
                "type": "10x",
                "gpu": i % 2,  # 轮流分配GPU
                "priority": 3
            })
    
    print(f"📋 总共配置了 {len(model_configs)} 个训练任务")
    
    # 统计信息
    if train_standard:
        standard_count = len([c for c in model_configs if c['type'] == 'standard'])
        skipped_standard = len(all_models) - len(models_to_train_standard)
        total_skipped += skipped_standard
        print(f"   标准模型: {standard_count} 个待训练, {skipped_standard} 个已完成")
    if train_10x:
        tenx_count = len([c for c in model_configs if c['type'] == '10x'])
        skipped_10x = len(all_models) - len(models_to_train_10x)
        total_skipped += skipped_10x
        print(f"   10x模型: {tenx_count} 个待训练, {skipped_10x} 个已完成")
    
    if total_skipped > 0 and not args.force_retrain:
        print(f"💡 总共跳过 {total_skipped} 个已训练模型 (使用 --force-retrain 可强制重新训练)")
    elif args.force_retrain:
        print(f"🔄 强制重新训练模式：将训练所有模型")
    
    if not model_configs:
        print("❌ 没有可训练的模型，请检查配置")
        return
    
    # 添加任务并运行
    trainer.add_tasks(model_configs)
    trainer.run()

if __name__ == "__main__":
    main() 