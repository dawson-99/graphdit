#!/bin/bash

# FGFR1 模型训练脚本
echo "========================================"
echo "开始训练FGFR1扩散模型..."
echo "开始时间: $(date)"
echo "========================================"

# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate graphdit

# 切换到项目目录
cd /lab/Xrh/code/Graph-DiT-main

# 创建日志目录
mkdir -p logs

# 运行训练，保存详细日志
echo "正在启动训练..."
python graph_dit/main.py --config-name=fgfr1_config.yaml 2>&1 | tee logs/fgfr1_training_$(date +%Y%m%d_%H%M%S).log

echo "========================================"
echo "训练结束时间: $(date)"
echo "========================================"