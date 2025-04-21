import torch
from pathlib import Path

def analyze_checkpoint(checkpoint_path):
    """
    分析检查点文件并输出其中的关键信息。

    参数:
        checkpoint_path (str): 检查点文件的路径
    """
    # 加载检查点文件
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 输出检查点中的关键信息
    print("检查点文件路径:", checkpoint_path)
    print("\n检查点中包含的键:", list(checkpoint.keys()))

    # 输出模型的状态字典
    if 'state_dict' in checkpoint:
        print("\n模型的状态字典:")
        for key, value in checkpoint['state_dict'].items():
            print(f"{key}: ")

    # 输出优化器的状态
    if 'optimizer_states' in checkpoint:
        print("\n优化器的状态:")
        for i, state in enumerate(checkpoint['optimizer_states']):
            print(f"优化器 {i}:")
            for key, value in state.items():
                if key == 'param_groups':
                    print(f"{key} {value}")

    # 输出当前的 epoch
    if 'epoch' in checkpoint:
        print("\n当前的 epoch:", checkpoint['epoch'])

    # 输出其他信息
    if 'global_step' in checkpoint:
        print("\n全局步数:", checkpoint['global_step'])
    if 'lr_schedulers' in checkpoint:
        print("\n学习率调度器的状态:", checkpoint['lr_schedulers'])

if __name__ == '__main__':
    # 指定检查点文件路径
    checkpoint_path = './runs/04170018_3DGaussTokens_car_abstract-projection/checkpoints/309-0.ckpt'
    
    # 分析检查点文件
    analyze_checkpoint(checkpoint_path)