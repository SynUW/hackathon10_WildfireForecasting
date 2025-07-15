# 野火预测模型训练系统使用说明

## 1. 环境依赖

### 1.1 Python版本
- 推荐 Python 3.10

### 1.2 CUDA环境
- 需具备NVIDIA GPU及CUDA驱动，推荐CUDA 11.8
- 驱动与CUDA Toolkit需与PyTorch版本兼容

### 1.3 必须安装的依赖包
请严格按照如下顺序安装，确保环境一致性。

#### 1.3.1 使用conda创建新环境
```bash
conda create -n wildfire python=3.10 -y
conda activate wildfire
```

#### 1.3.2 安装PyTorch（含CUDA支持）
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 1.3.3 安装核心依赖
```bash
conda install numpy pandas scikit-learn tqdm h5py matplotlib
pip install wandb
```

#### 1.3.4 安装Mamba模型依赖（如需训练s_mamba/Mamba系列模型，必须安装）
```bash
pip install mamba-ssm
```

#### 1.3.5 其他依赖
- 推荐使用conda/pip安装所有依赖，避免版本冲突。
- 若需wandb实验监控，需提前注册并执行`wandb login`。

## 2. 数据准备

### 2.1 数据目录结构
- 所有HDF5数据文件需放置于：
  ```
  /mnt/raid/zhengsen/wildfire_dataset/self_built_materials/full_datasets/
  ```

### 2.2 数据文件命名与内容
- 文件名格式：`{year}_year_dataset.h5`，如`2020_year_dataset.h5`
- 每个文件包含所有像元的全年多通道数据
- 数据集名：`{row}_{col}`，如`37_255`
- 数据形状：(39, 365) 或 (39, 366)
- 通道0为FIRMS火点数据，float类型

### 2.3 缓存机制
- 首次运行会自动生成采样缓存文件于`full_datasets/cache/`目录，无需手动操作

## 3. 训练模型

### 3.1 单个模型训练
```bash
# 训练指定的单个模型
python train_single_model.py --model DLinear --model-type standard
python train_single_model.py --model Mamba --model-type 10x

# 查看可用模型列表
python train_single_model.py --list-models
```
- 适用于测试单个模型或调试特定模型
- 支持标准模型和10x模型类型选择

### 3.2 批量顺序训练
```bash
# 训练所有标准模型（顺序执行）
python train_all_models_combined.py --skip-10x --force-retrain

# 训练所有10x大模型
python train_all_models_combined.py --only-10x --force-retrain

# 训练所有模型（标准+10x）
python train_all_models_combined.py --force-retrain
```
- 自动训练所有模型，覆盖所有主流时序结构
- 标准模型结果保存在`/mnt/raid/zhengsen/pths/7to1_Focal_woFirms_onlyFirmsLoss_newloadertest/`
- 10x模型结果保存在`/mnt/raid/zhengsen/pths/model_pth_20epoch_MSE_10x/`

### 3.3 并行训练（推荐）
```bash
# 自动分配GPU并行训练多个模型
python smart_parallel.py

# 指定GPU数量
python smart_parallel.py --num-gpus 4

# 指定每个GPU的最大并行任务数
python smart_parallel.py --max-parallel-per-gpu 2
```
- **推荐使用**：自动分配GPU资源，显著提升训练效率
- 智能负载均衡，避免GPU资源浪费
- 支持断点续传，自动跳过已训练模型

### 3.4 训练脚本对比

| 脚本名称 | 适用场景 | 优势 | 劣势 |
|---------|---------|------|------|
| `train_single_model.py` | 单模型调试、测试 | 快速验证、参数调优 | 无法批量处理 |
| `train_all_models_combined.py` | 完整实验、结果对比 | 功能完整、结果统一 | 训练时间长、资源利用率低 |
| `smart_parallel.py` | 生产环境、大规模训练 | 高效并行、智能调度 | 配置稍复杂 |

### 3.5 训练参数说明
- 所有参数均可通过`python train_all_models_combined.py --help`查看
- 常用参数：
  - `--firms-weight`：FIRMS损失权重
  - `--other-drivers-weight`：其他驱动因素损失权重
  - `--loss-type`：损失函数类型（focal/kldiv/multitask）
  - `--enable-position-features`：启用位置信息特征
  - `--enable-future-weather`：启用未来气象特征
  - `--weather-channels`：气象通道范围（如1-12）

## 4. 可视化预测结果

### 4.1 生成预测可视化
```bash
# 可视化指定模型的预测结果
python test_and_visualize_optimized.py --model DLinear --model-type standard

# 可视化多个模型并生成对比图
python test_and_visualize_optimized.py --model Mamba --model-type 10x --save-comparison

# 生成指定时间窗口的预测结果
python test_and_visualize_optimized.py --model iTransformer --window-range 1-10

# 批量可视化所有模型结果
python test_and_visualize_full.py
```

### 4.2 可视化输出
- 自动生成TIFF格式的预测结果图像
- 包含真实值vs预测值的对比可视化
- 生成定量评价指标（精确率、召回率、F1分数等）
- 支持批量处理多个时间窗口

## 5. 运行环境变量与注意事项
- 建议在Linux服务器（如Ubuntu 20.04）下运行，确保有足够的GPU显存（推荐24GB及以上）
- 训练脚本自动检测CUDA设备，无需手动指定
- 若需指定GPU，可设置`CUDA_VISIBLE_DEVICES`环境变量
- 若需wandb监控，需提前`wandb login`，否则可在脚本中关闭WandB
- 若训练Mamba系列模型，必须在已安装`mamba-ssm`的环境下运行

## 6. 常见问题
- **数据加载慢/内存占用高**：首次采样慢，后续极快。建议使用SSD。
- **Mamba模型报错**：请确保`mamba-ssm`已正确安装，且在对应环境下运行。
- **显存不足**：可调小`batch_size`，或只训练标准模型。
- **wandb相关问题**：如不需实验监控，可在脚本中关闭WandB。
- **数据路径/命名错误**：请严格按照上述目录和命名规范准备数据。

---
如有环境或训练相关问题，请联系维护者或提交issue。 