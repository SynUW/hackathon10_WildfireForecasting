#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行训练脚本 - 在多个GPU上同时训练不同模型
支持在2个GPU上并行运行4个训练任务
"""

import subprocess
import threading
import time
import os
import sys
from datetime import datetime
import json

# 配置参数
PARALLEL_CONFIG = {
    'gpu_devices': [0, 1],           # 使用的GPU设备
    'tasks_per_gpu': 2,              # 每个GPU上运行的任务数
    'train_script': 'train_all_models_combined.py',  # 训练脚本名称
    'log_dir': 'parallel_logs',      # 日志目录
    'wait_interval': 5,              # 检查任务状态的间隔（秒）
}

# 获取所有可用模型
def get_available_models():
    """获取所有可用的模型列表"""
    try:
        # 从train_all_models_combined.py中获取模型列表
        import importlib.util
        spec = importlib.util.spec_from_file_location("train_module", PARALLEL_CONFIG['train_script'])
        train_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(train_module)
        
        standard_models = train_module.MODEL_LIST_STANDARD
        models_10x = train_module.MODEL_LIST_10X
        
        return standard_models, models_10x
    except Exception as e:
        print(f"⚠️ 无法获取模型列表: {e}")
        # 备用模型列表
        standard_models = ['Autoformer', 'Autoformer_M', 'iTransformer', 's_mamba']
        models_10x = ['Autoformer', 'Autoformer_M', 'iTransformer', 's_mamba']
        return standard_models, models_10x

class ParallelTrainer:
    """并行训练管理器"""
    
    def __init__(self):
        self.running_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.start_time = None
        
        # 创建日志目录
        os.makedirs(PARALLEL_CONFIG['log_dir'], exist_ok=True)
        
    def create_single_model_script(self, model_name, model_type, gpu_id):
        """为单个模型创建训练脚本"""
        script_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单模型训练脚本 - {model_name} ({model_type})
自动生成的并行训练脚本
"""

import os
import sys
import torch

# 设置CUDA设备
os.environ['CUDA_VISIBLE_DEVICES'] = '{gpu_id}'

# 导入训练模块
from train_all_models_combined import *

def train_single_model_only():
    """只训练指定的单个模型"""
    print(f"🚀 开始训练单个模型: {{model_name}} ({{model_type}}) on GPU {{gpu_id}}")
    
    # 初始化
    set_seed(TRAINING_CONFIG['seed'])
    device = torch.device('cuda:0')  # 由于设置了CUDA_VISIBLE_DEVICES，这里总是0
    print(f"🖥️  使用设备: GPU {{gpu_id}} (映射为cuda:0)")
    
    # 准备数据
    train_dataset, val_dataset, test_dataset, data_loader_obj = prepare_data_loaders()
    
    # 初始化FIRMS归一化器
    print("🔧 初始化FIRMS归一化器...")
    firms_normalizer = FIRMSNormalizer(
        method='log1p_minmax',
        firms_min=DATA_CONFIG['firms_min'],
        firms_max=DATA_CONFIG['firms_max']
    )
    
    # 为归一化拟合创建临时数据加载器
    temp_loader = DataLoader(
        train_dataset, batch_size=512, shuffle=False, 
        num_workers=2, collate_fn=data_loader_obj.dataset.custom_collate_fn
    )
    firms_normalizer.fit(temp_loader)
    
    # 创建数据加载器
    config_key = '{model_type}'
    train_config = TRAINING_CONFIG[config_key]
    
    train_loader = DataLoader(
        train_dataset, batch_size=train_config['batch_size'], shuffle=True, 
        num_workers=2, collate_fn=data_loader_obj.dataset.custom_collate_fn, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_config['batch_size'], shuffle=False,
        num_workers=2, collate_fn=data_loader_obj.dataset.custom_collate_fn, worker_init_fn=worker_init_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=train_config['batch_size'], shuffle=False,
        num_workers=2, collate_fn=data_loader_obj.dataset.custom_collate_fn, worker_init_fn=worker_init_fn
    )
    
    # 训练模型
    try:
        result = train_single_model(
            '{model_name}', device, train_loader, val_loader, test_loader, firms_normalizer, '{model_type}'
        )
        
        if result is not None:
            print(f"✅ {{model_name}} ({{model_type}}) 训练完成")
            
            # 测试所有保存的模型
            test_results = []
            for metric_name, metric_info in result.items():
                if metric_info['path'] is not None:
                    print(f"🧪 测试 {{metric_name}} 模型...")
                    test_result = test_model('{model_name}', metric_info['path'], device, test_loader, firms_normalizer, '{model_type}')
                    if test_result:
                        test_results.append((test_result, metric_name))
            
            # 保存结果
            results_file = f"{{PARALLEL_CONFIG['log_dir']}}/{model_name}_{model_type}_results.json"
            with open(results_file, 'w') as f:
                json.dump({{
                    'model_name': '{model_name}',
                    'model_type': '{model_type}',
                    'gpu_id': {gpu_id},
                    'training_metrics': result,
                    'test_results': test_results,
                    'status': 'completed'
                }}, f, indent=2, default=str)
            
            print(f"📊 结果已保存到: {{results_file}}")
        else:
            print(f"❌ {{model_name}} ({{model_type}}) 训练失败")
            
    except Exception as e:
        print(f"❌ 训练过程中出现错误: {{e}}")
        # 保存错误信息
        error_file = f"{{PARALLEL_CONFIG['log_dir']}}/{model_name}_{model_type}_error.json"
        with open(error_file, 'w') as f:
            json.dump({{
                'model_name': '{model_name}',
                'model_type': '{model_type}',
                'gpu_id': {gpu_id},
                'error': str(e),
                'status': 'failed'
            }}, f, indent=2)
        raise

if __name__ == "__main__":
    train_single_model_only()
'''
        
        # 保存脚本文件
        script_filename = f"{PARALLEL_CONFIG['log_dir']}/train_{model_name}_{model_type}_gpu{gpu_id}.py"
        with open(script_filename, 'w') as f:
            f.write(script_content)
        
        return script_filename
    
    def run_task(self, model_name, model_type, gpu_id, task_id):
        """运行单个训练任务"""
        print(f"🚀 启动任务 {task_id}: {model_name} ({model_type}) on GPU {gpu_id}")
        
        # 创建单模型训练脚本
        script_path = self.create_single_model_script(model_name, model_type, gpu_id)
        
        # 设置日志文件
        log_file = f"{PARALLEL_CONFIG['log_dir']}/train_{model_name}_{model_type}_gpu{gpu_id}.log"
        
        # 运行训练脚本
        try:
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    [sys.executable, script_path],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
                )
                
                # 等待任务完成
                return_code = process.wait()
                
                if return_code == 0:
                    print(f"✅ 任务 {task_id} 完成: {model_name} ({model_type})")
                    self.completed_tasks.append((task_id, model_name, model_type, gpu_id))
                else:
                    print(f"❌ 任务 {task_id} 失败: {model_name} ({model_type}) - 返回码: {return_code}")
                    self.failed_tasks.append((task_id, model_name, model_type, gpu_id, return_code))
                    
        except Exception as e:
            print(f"❌ 任务 {task_id} 异常: {model_name} ({model_type}) - 错误: {e}")
            self.failed_tasks.append((task_id, model_name, model_type, gpu_id, str(e)))
        
        finally:
            # 从运行列表中移除
            self.running_tasks = [(tid, mn, mt, gid) for tid, mn, mt, gid in self.running_tasks 
                                 if not (tid == task_id and mn == model_name and mt == model_type)]
    
    def create_task_queue(self, models_standard, models_10x):
        """创建任务队列"""
        tasks = []
        task_id = 1
        
        # 添加标准模型任务
        for model in models_standard:
            tasks.append((task_id, model, 'standard'))
            task_id += 1
            
        # 添加10x模型任务
        for model in models_10x:
            tasks.append((task_id, model, '10x'))
            task_id += 1
            
        return tasks
    
    def run_parallel_training(self, models_standard, models_10x):
        """运行并行训练"""
        self.start_time = datetime.now()
        print(f"🔥 开始并行训练 - {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📋 配置: {len(PARALLEL_CONFIG['gpu_devices'])} GPU, 每GPU {PARALLEL_CONFIG['tasks_per_gpu']} 任务")
        print(f"📊 标准模型: {len(models_standard)} 个")
        print(f"📊 10x模型: {len(models_10x)} 个")
        
        # 创建任务队列
        task_queue = self.create_task_queue(models_standard, models_10x)
        total_tasks = len(task_queue)
        print(f"📋 总任务数: {total_tasks}")
        
        # 计算最大并行任务数
        max_parallel = len(PARALLEL_CONFIG['gpu_devices']) * PARALLEL_CONFIG['tasks_per_gpu']
        
        task_index = 0
        active_threads = []
        
        while task_index < total_tasks or active_threads:
            # 启动新任务（如果有空闲槽位）
            while len(active_threads) < max_parallel and task_index < total_tasks:
                task_id, model_name, model_type = task_queue[task_index]
                
                # 分配GPU
                gpu_id = PARALLEL_CONFIG['gpu_devices'][len(active_threads) % len(PARALLEL_CONFIG['gpu_devices'])]
                
                # 创建并启动线程
                thread = threading.Thread(
                    target=self.run_task,
                    args=(model_name, model_type, gpu_id, task_id)
                )
                thread.start()
                active_threads.append(thread)
                self.running_tasks.append((task_id, model_name, model_type, gpu_id))
                
                task_index += 1
                time.sleep(2)  # 避免同时启动过多任务
            
            # 检查完成的线程
            active_threads = [t for t in active_threads if t.is_alive()]
            
            # 显示进度
            completed = len(self.completed_tasks)
            failed = len(self.failed_tasks)
            running = len(active_threads)
            
            print(f"📊 进度: 完成={completed}, 失败={failed}, 运行中={running}, 剩余={total_tasks-completed-failed-running}")
            
            time.sleep(PARALLEL_CONFIG['wait_interval'])
        
        # 等待所有线程完成
        for thread in active_threads:
            thread.join()
        
        # 输出最终结果
        self.print_final_results()
    
    def print_final_results(self):
        """打印最终结果"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print(f"\n{'='*80}")
        print(f"🎉 并行训练完成! - {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  总用时: {duration}")
        print(f"{'='*80}")
        
        print(f"\n📊 训练结果统计:")
        print(f"✅ 成功完成: {len(self.completed_tasks)} 个任务")
        print(f"❌ 失败任务: {len(self.failed_tasks)} 个任务")
        
        if self.completed_tasks:
            print(f"\n✅ 成功完成的任务:")
            for task_id, model_name, model_type, gpu_id in self.completed_tasks:
                print(f"  - 任务{task_id}: {model_name} ({model_type}) on GPU {gpu_id}")
        
        if self.failed_tasks:
            print(f"\n❌ 失败的任务:")
            for task_id, model_name, model_type, gpu_id, error in self.failed_tasks:
                print(f"  - 任务{task_id}: {model_name} ({model_type}) on GPU {gpu_id} - {error}")
        
        print(f"\n📁 详细日志和结果文件保存在: {PARALLEL_CONFIG['log_dir']}/")
        print(f"📊 可以查看各个模型的训练日志和结果文件")

def main():
    """主函数"""
    print("🔥 野火预测模型并行训练系统")
    print("="*50)
    
    # 检查训练脚本是否存在
    if not os.path.exists(PARALLEL_CONFIG['train_script']):
        print(f"❌ 训练脚本不存在: {PARALLEL_CONFIG['train_script']}")
        return
    
    # 检查GPU可用性
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，无法进行GPU训练")
            return
        
        available_gpus = torch.cuda.device_count()
        print(f"💾 可用GPU数量: {available_gpus}")
        
        for gpu_id in PARALLEL_CONFIG['gpu_devices']:
            if gpu_id >= available_gpus:
                print(f"❌ GPU {gpu_id} 不存在，可用GPU: 0-{available_gpus-1}")
                return
            
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            print(f"  GPU {gpu_id}: {gpu_name} ({gpu_memory:.1f} GB)")
            
    except ImportError:
        print("❌ PyTorch未安装，无法检查GPU状态")
        return
    
    # 获取模型列表
    print(f"\n📋 获取可训练模型列表...")
    models_standard, models_10x = get_available_models()
    
    if not models_standard and not models_10x:
        print("❌ 没有找到可训练的模型")
        return
    
    print(f"📊 标准模型 ({len(models_standard)}): {', '.join(models_standard)}")
    print(f"📊 10x模型 ({len(models_10x)}): {', '.join(models_10x)}")
    
    # 用户确认
    total_tasks = len(models_standard) + len(models_10x)
    max_parallel = len(PARALLEL_CONFIG['gpu_devices']) * PARALLEL_CONFIG['tasks_per_gpu']
    
    print(f"\n🔧 并行训练配置:")
    print(f"  - 总任务数: {total_tasks}")
    print(f"  - 最大并行数: {max_parallel}")
    print(f"  - 使用GPU: {PARALLEL_CONFIG['gpu_devices']}")
    print(f"  - 每GPU任务数: {PARALLEL_CONFIG['tasks_per_gpu']}")
    
    response = input(f"\n是否开始并行训练? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("❌ 用户取消训练")
        return
    
    # 开始并行训练
    trainer = ParallelTrainer()
    trainer.run_parallel_training(models_standard, models_10x)

if __name__ == "__main__":
    main() 