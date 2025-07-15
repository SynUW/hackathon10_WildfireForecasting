#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置文件训练器 - 通过JSON配置文件管理训练任务
"""

import json
import argparse
from smart_parallel import SmartParallelTrainer

def load_config(config_file):
    """加载配置文件"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        return None

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='配置文件训练器')
    parser.add_argument('--config', '-c', type=str, default='training_config.json',
                        help='配置文件路径 (默认: training_config.json)')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    if not config:
        return
    
    # 获取设置
    settings = config.get('settings', {})
    max_parallel_per_gpu = settings.get('max_parallel_per_gpu', 2)
    
    # 创建训练器
    trainer = SmartParallelTrainer(max_parallel_per_gpu=max_parallel_per_gpu)
    
    # 如果配置了日志目录，更新训练器设置
    if 'log_dir' in settings:
        trainer.log_dir = settings['log_dir']
        import os
        os.makedirs(trainer.log_dir, exist_ok=True)
    
    # 获取任务列表并按优先级排序
    tasks = config.get('tasks', [])
    tasks.sort(key=lambda x: x.get('priority', 999))
    
    print(f"🔥 配置文件训练器")
    print(f"📁 配置文件: {args.config}")
    print(f"📋 任务数量: {len(tasks)}")
    print(f"⚙️ 每GPU最大并行: {max_parallel_per_gpu}")
    print("=" * 50)
    
    # 添加任务并运行
    trainer.add_tasks(tasks)
    trainer.run()

if __name__ == "__main__":
    main() 