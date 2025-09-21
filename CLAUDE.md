# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

## 项目概览

Graph-DiT 是一个用于多条件分子生成的图扩散Transformer。这是一个基于 PyTorch Lightning 的实现，使用扩散模型和 transformer 架构来生成分子图。

## 开发命令

### 训练和运行模型

训练模型的主要命令：
```bash
python graph_dit/main.py --config-name=config.yaml [附加选项]
```

不同数据集的示例命令：
```bash
# 多气体条件下的聚合物图生成
python graph_dit/main.py --config-name=config.yaml \
    model.ensure_connected=True \
    dataset.task_name='O2-N2-CO2' \
    dataset.guidance_target='O2-N2-CO2'

# 小分子生成
python graph_dit/main.py --config-name=config.yaml \
    dataset.task_name='bace_b' \
    dataset.guidance_target='Class'

# 单一气体渗透率条件
python graph_dit/main.py --config-name=config.yaml \
    dataset.task_name='O2' \
    dataset.guidance_target='O2'
```

### 依赖安装和环境设置

安装依赖：
```bash
pip install -r requirements.txt
```

分子评估所需的外部包：
```bash
pip install fcd_torch
pip install git+https://github.com/igor-krawczuk/mini-moses
```

## 代码架构

### 核心组件

1. **主入口点**: `graph_dit/main.py` - 使用 Hydra 进行配置管理，PyTorch Lightning 进行训练
2. **主要模型**: `graph_dit/diffusion_model.py` - 包含 `Graph_DiT` 类，主要的 Lightning 模块
3. **配置文件**: `configs/config.yaml` - 包含模型、训练和数据集参数的中央配置

### 关键模块

- **模型**: `graph_dit/models/`
  - `transformer.py` - 核心去噪器 transformer 架构
  - `conditions.py` - 条件引导实现
  - `layers.py` - 自定义 transformer 层

- **扩散**: `graph_dit/diffusion/`
  - `diffusion_utils.py` - 扩散过程工具
  - `noise_schedule.py` - 噪声调度和边际转换
  - `distributions.py` - 离散扩散的概率分布

- **数据集**: `graph_dit/datasets/` - 数据加载和预处理
- **指标**: `graph_dit/metrics/` - 训练和采样指标
- **分析**: `graph_dit/analysis/` - 可视化和分析工具

### 数据结构

- `data/` - 数据集存储目录
  - `raw/` - 原始数据集文件
  - `split/` - 训练/测试划分
  - `evaluator/` - 评估工具

### 配置管理

项目使用 Hydra 进行配置管理。关键配置部分：

- `general` - 训练设置、GPU、日志、采样参数
- `model` - 模型架构、扩散步骤、引导尺度
- `train` - 训练参数（轮数、批大小、学习率）
- `dataset` - 数据路径、任务名称、引导目标

### 模型架构

Graph-DiT 结合了：
- 用于图生成的离散扩散过程
- 基于 Transformer 的去噪架构
- 带引导的多条件生成
- 对节点和边特征的支持
- 分子属性条件化

## 关键实现要点

- 使用 PyTorch Lightning 作为训练基础设施
- 使用 YAML 配置的 Hydra 配置系统
- 带边际转换的离散扩散
- 用于条件生成的无分类器引导
- GPU/CPU 自动检测和通过 DDP 的多GPU支持
- 模型检查点和恢复功能

## 测试和模型恢复

从检查点恢复训练：
```bash
python graph_dit/main.py --config-name=config.yaml general.resume=path/to/checkpoint.ckpt
```

仅测试模式：
```bash
python graph_dit/main.py --config-name=config.yaml general.test_only=path/to/checkpoint.ckpt
```

代码库需要 Python 3.9.16、PyTorch 2.0.0、PyTorch Geometric 2.3.0 和 PyTorch Lightning 2.0.1 以获得最佳兼容性。

# 模型实现的过程
● 基于对代码的深入分析，我来详细说明Graph-DiT扩散模型的具体实现原理：

  1. 离散扩散模型核心原理

  Graph-DiT实现了一个离散扩散模型，专门针对分子图生成。与传统的连续扩散不同，它处理的是离散的原子类型和键类型：

  噪声调度机制

  - 余弦调度：diffusion_utils.py:14-23使用余弦噪声调度，提供平滑的噪声添加过程
  - 转移矩阵：通过MarginalTransition类实现状态转移，定义了如何在不同原子/键类型之间转换
  - 边际分布：基于数据统计的先验分布，确保生成的分子符合化学规律

  前向过程（加噪）

  # 一步转移矩阵：Qt = (1 - β_t) * I + β_t * U
  q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes)
  其中u_x是基于边际分布的均匀转移矩阵。

  2. DiT架构设计

  Transformer作为去噪器

  Graph-DiT的核心创新是使用**Diffusion Transformer (DiT)**架构：

  - 输入处理：transformer.py:87将节点特征和边特征拼接后映射到隐藏空间
  - 自注意力层：SELayer类实现了条件化的自注意力，能处理图结构
  - AdaLN调制：使用自适应层归一化，将时间步和条件信息注入到每一层

  关键技术实现

  # 条件调制机制
  shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=1)
  x = x + gate * modulate(self.norm(self.attn(x)), shift, scale)

  3. 多条件引导机制

  条件嵌入系统

  Graph-DiT支持多种条件输入：

  1. 时间步嵌入：TimestepEmbedder使用正弦位置编码
  2. 分类条件：CategoricalEmbedder处理离散属性（如活性分类）
  3. 连续条件：ClusterContinuousEmbedder处理连续属性（如气体渗透率）

  无分类器引导（Classifier-Free Guidance）

  - 条件丢弃：训练时随机丢弃条件信息（drop_condition参数）
  - 引导推理：推理时使用引导尺度控制条件强度：
  pred_uncond = model(x, t, y=None)  # 无条件预测
  pred_cond = model(x, t, y=y)       # 有条件预测
  pred = pred_uncond + guide_scale * (pred_cond - pred_uncond)

  4. 图结构处理

  节点-边联合建模

  - 节点特征：原子类型（118维onehot编码，映射到活跃原子类型）
  - 边特征：键类型（5维：无键，单键，双键，三键，芳香键）
  - 联合转移：通过MarginalTransition实现节点-边状态的联合转移

  对称性保持

  # 确保边矩阵对称性
  bond_out = 1/2 * (bond_out + torch.transpose(bond_out, 1, 2))

  5. 训练和采样过程

  训练损失

  使用加权的负对数似然损失：
  - 节点损失权重：lambda_train[0]
  - 边损失权重：lambda_train[1]（通常为10，因为边更稀疏）

  采样策略

  - 后验分布计算：diffusion_utils.py:106计算精确的后验分布
  - 离散采样：使用多项式分布采样离散状态
  - 连通性确保：ensure_connected=True时保证生成分子的连通性

  6. 关键创新点

  1. 离散化处理：专门设计的离散扩散过程，适合分子图的离散特性
  2. 联合建模：统一处理节点和边的生成过程
  3. 多条件融合：支持多种分子属性的条件生成
  4. 化学约束：通过边际分布和连通性约束保证化学合理性
  5. DiT架构：将Transformer的强大建模能力引入扩散模型

  这个实现将扩散模型的生成能力与Transformer的序列建模优势结合，创造了一个强大的分子生成系统，能够根据多种化学属性条件生成具有特定性质的分子结构。

  7. 用户规则
  - 你只能用中文和我对话
  - 你修改代码前，请让我确认再修改
  - 请叫我主人