#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的SHAP分析命令行工具
"""

import os
import sys
import argparse
from shap_analysis import SHAPAnalyzer

def main():
    parser = argparse.ArgumentParser(description='SHAP分析工具 - 生成类似SHAP摘要图的可视化')
    parser.add_argument('--model_path', type=str, required=False, default='/mnt/raid/zhengsen/pths/365to1_focal_withRegressionLoss_withfirms_baseline/s_mamba_best_recall.pth',
                       help='模型文件路径(.pth)')
    parser.add_argument('--model_name', type=str, required=False, default='s_mamba',
                       help='模型名称 (如: SCINet, Autoformer, TimeXer, s_mamba)')
    parser.add_argument('--seq_len', type=int, default=365, 
                       help='输入序列长度 (默认: 365)')
    parser.add_argument('--pred_len', type=int, default=1, 
                       help='预测长度 (默认: 1)')
    parser.add_argument('--input_channels', type=int, default=39, 
                       help='输入通道数 (默认: 40, 39基础+1position)')
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='输出目录 (默认: ./{model_name}_shap_results)')
    parser.add_argument('--num_samples', type=int, default=100, 
                       help='样本数量 (默认: 1000)')
    parser.add_argument('--fast_mode', action='store_true',
                       help='启用快速模式（使用梯度近似，适用于小样本）')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批量大小 (默认: 64)')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='计算设备 (默认: cuda)')
    
    args = parser.parse_args()
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 模型文件不存在: {args.model_path}")
        sys.exit(1)
    
    # 设置默认输出目录
    if args.output_dir is None:
        args.output_dir = f'./{args.model_name.lower()}_shap_results'
    
    print(f"🎯 开始 {args.model_name} 模型的SHAP分析")
    print(f"📁 模型文件: {args.model_path}")
    print(f"📊 输入配置: seq_len={args.seq_len}, pred_len={args.pred_len}, channels={args.input_channels}")
    print(f"📈 样本数量: {args.num_samples}")
    print(f"⚡ 快速模式: {'启用' if args.fast_mode else '禁用'}")
    print(f"📦 批量大小: {args.batch_size}")
    print(f"💾 输出目录: {args.output_dir}")
    print(f"🖥️  计算设备: {args.device}")
    print("-" * 60)
    
    try:
        # 创建SHAP分析器
        analyzer = SHAPAnalyzer(
            model_path=args.model_path,
            model_name=args.model_name,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            input_channels=args.input_channels,
            device=args.device
        )
        
        # 运行SHAP分析
        results = analyzer.run_shap_analysis(args.output_dir, args.num_samples)
        
        print("\n🎉 SHAP分析完成！")
        print(f"📊 结果保存在: {args.output_dir}")
        print("\n📋 生成的文件:")
        print(f"   - {args.model_name}_shap_summary.png (SHAP摘要图)")
        print(f"   - {args.model_name}_feature_importance.png (特征重要性图)")
        print(f"   - {args.model_name}_shap_analysis_report.md (详细分析报告)")
        print("\n🔍 分析内容包含:")
        print("   - SHAP摘要图：每个特征的SHAP值分布和特征值关系")
        print("   - 特征重要性图：按重要性排序的特征排名")
        print("   - 详细分析报告：包含统计信息、排名表格、特征分析、建议等")
        print("   - 样本级别的详细解释和影响方向分析")
        
    except Exception as e:
        print(f"❌ SHAP分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 