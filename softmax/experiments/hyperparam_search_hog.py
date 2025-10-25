"""
Softmax (HOG) 超参数搜索脚本
"""

import sys
from pathlib import Path
import numpy as np

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from softmax_classifier import SoftmaxClassifier
from cifar10_utils_hog import load_cifar10, preprocess_data

def run_search():
    print("=" * 70)
    print(" " * 15 + "HOG + Softmax 超参数搜索")
    print("=" * 70)

    # 1. 加载数据 (我们只需要训练集和验证集)
    print("加载数据...")
    X_train_raw, y_train, X_val_raw, y_val, _, _ = load_cifar10(
        data_dir='../../data/cifar-10-batches-py',
        train_samples=49000,
        val_samples=1000,
        test_samples=0  # 不加载测试集
    )
    
    print("预处理 (HOG)...")
    X_train, X_val, _ = preprocess_data(X_train_raw, X_val_raw, np.array([]))
    
    print(f"✓ 训练集: {X_train.shape}, 验证集: {X_val.shape}")
    
    # ==============================================================
    # ✨ 在这里定义你的搜索网格
    # ==============================================================
    learning_rates = [1e-3, 5e-3, 1e-2]
    # (HOG模型几乎不过拟合, 我们可以尝试更小的正则化强度)
    reg_strengths = [5e-5, 1e-4, 5e-4] 
    
    best_val_acc = -1.0
    best_params = {}
    
    print("\n" + "=" * 70)
    print(f"开始搜索 {len(learning_rates) * len(reg_strengths)} 组参数...")
    print("=" * 70)
    
    # 嵌套循环搜索
    for lr in learning_rates:
        for reg in reg_strengths:
            print(f"测试中: lr={lr}, reg={reg}")
            
            # 创建分类器
            classifier = SoftmaxClassifier(
                num_features=X_train.shape[1],
                num_classes=10,
                reg_strength=reg
            )
            
            # 训练模型 (注意: patience 设小一点, 加速搜索)
            classifier.train(
                X_train, y_train,
                X_val, y_val,
                learning_rate=lr,
                num_epochs=1000, # (早停会自动处理)
                batch_size=128,
                patience=150,     # <-- 缩短耐心值
                verbose=False     # <-- 关闭啰嗦的打印
            )
            
            # 评估验证集
            val_acc = classifier.evaluate(X_val, y_val)
            print(f"  -> 验证集准确率: {val_acc:.4f}")
            
            # 记录最佳结果
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {'lr': lr, 'reg': reg}
                print(f"  ✨ 新的最佳准确率!")

    print("\n" + "=" * 70)
    print("搜索完成!")
    print(f"🏆 最佳验证集准确率: {best_val_acc:.4f}")
    print(f"   最佳参数: {best_params}")
    print("=" * 70)

if __name__ == '__main__':
    run_search()