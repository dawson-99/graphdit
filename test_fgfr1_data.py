#!/usr/bin/env python3
"""测试FGFR1数据集加载"""

import sys
import os
os.chdir('/lab/Xrh/code/Graph-DiT-main')
sys.path.append('/lab/Xrh/code/Graph-DiT-main')

import hydra
from omegaconf import DictConfig
from graph_dit.datasets.dataset import DataModule

# 测试配置
test_config = {
    'dataset': {
        'datadir': 'data/',
        'task_name': 'fgfr1',
        'guidance_target': 'ld50-label-solubility',
        'pin_memory': False
    },
    'train': {
        'batch_size': 4,
        'num_workers': 0
    }
}

def test_data_loading():
    print("正在测试FGFR1数据集加载...")

    # 创建配置对象
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(test_config)

    try:
        # 创建数据模块
        dm = DataModule(cfg)
        dm.prepare_data()

        print(f"✓ 数据集成功加载")
        print(f"✓ 训练样本数: {len(dm.train_index)}")
        print(f"✓ 验证样本数: {len(dm.val_index)}")
        print(f"✓ 测试样本数: {len(dm.test_index)}")

        # 测试数据批次
        example = dm.example_batch()
        print(f"✓ 批次形状: x={example.x.shape}, edge_attr={example.edge_attr.shape}, y={example.y.shape}")
        print(f"✓ 条件维度: {example.y.shape[1]} (应该是5: SA+SC+ld50+label+solubility)")

        # 检查数据范围
        print(f"✓ y值范围: {example.y.min():.3f} ~ {example.y.max():.3f}")

        return True

    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n🎉 数据集配置成功！可以开始训练模型了。")
        print("\n训练命令:")
        print("python graph_dit/main.py --config-name=fgfr1_config.yaml")
    else:
        print("\n❌ 数据集配置失败，请检查错误信息。")