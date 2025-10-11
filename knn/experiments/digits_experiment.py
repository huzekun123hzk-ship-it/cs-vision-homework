# knn/experiments/digits_experiment.py

"""
使用纯 NumPy 实现的 KNNClassifier 在手写数字 (8x8) 数据集上的分类演示。

本脚本使用 sklearn.datasets.load_digits 加载数据集，数据被归一化到 [0, 1]，
然后拆分为训练集和测试集，并用我们手搓的 KNN 算法进行评估。
最终保存混淆矩阵热力图和一个样本预测网格图。
"""
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# --- 导入自定义模块 ---
# 将上级目录 (knn/) 添加到 Python 的模块搜索路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from knn_classifier import KNNClassifier, train_test_split_indices

def setup_matplotlib_font():
    """设置 Matplotlib 的中文字体。"""
    try:
        plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]
        plt.rcParams["axes.unicode_minus"] = False
    except:
        print("警告：中文字体设置失败，图表中的中文可能无法正常显示。")

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """手搓一个简单的混淆矩阵计算函数。"""
    labels = np.unique(np.concatenate([np.unique(y_true), np.unique(y_pred)]))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((labels.size, labels.size), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return labels, cm

def plot_confusion_matrix(labels: np.ndarray, cm: np.ndarray, ax: plt.Axes):
    """绘制混淆矩阵热力图。"""
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("混淆矩阵 (行=真实值, 列=预测值)")
    ax.set_xlabel("预测值")
    ax.set_ylabel("真实值")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=8)
    return im

def plot_samples_grid(images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, rows: int, cols: int, axarr):
    """绘制样本预测结果的网格图。"""
    indices_to_show = np.arange(images.shape[0])[: rows * cols]
    for k, i in enumerate(indices_to_show):
        r, c = divmod(k, cols)
        ax = axarr[r, c]
        ax.imshow(images[i], cmap="gray")
        correct = (y_true[i] == y_pred[i])
        color = "green" if correct else "red"
        ax.set_title(f"真:{y_true[i]} 预:{y_pred[i]}", color=color, fontsize=8)
        ax.axis("off")

def main():
    """脚本主入口，用于手写数字分类实验。"""
    parser = argparse.ArgumentParser(description="在 sklearn digits (8x8) 数据集上运行 K-NN")
    parser.add_argument("--k", type=int, default=5, help="邻居数量 K 值")
    parser.add_argument("--p", type=float, default=2.0, help="闵可夫斯基距离的阶数 (p=2 为欧氏距离)")
    parser.add_argument("--test-ratio", type=float, default=0.25, help="测试集所占比例")
    parser.add_argument("--seed", type=int, default=42, help="用于复现的随机种子")
    parser.add_argument("--grid-rows", type=int, default=4, help="样本网格图的行数")
    parser.add_argument("--grid-cols", type=int, default=8, help="样本网格图的列数")
    args = parser.parse_args()
    
    setup_matplotlib_font()

    # --- 1. 加载和预处理数据 ---
    print("正在加载 Digits (8x8) 数据集...")
    digits = load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(int)
    images = digits.images

    X -= X.min()
    max_val = X.max()
    if max_val > 0:
        X /= max_val
    print("✓ 数据加载和归一化完成。")

    # --- 2. 划分训练集和测试集 (使用手搓函数) ---
    print(f"正在按 {1 - args.test_ratio:.0%}/{args.test_ratio:.0%} 的比例划分数据集...")
    rng = np.random.default_rng(args.seed)
    train_idx, test_idx = train_test_split_indices(len(y), test_ratio=args.test_ratio, rng=rng)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    img_test = images[test_idx]
    print(f"  - 训练集大小: {len(X_train)}")
    print(f"  - 测试集大小: {len(X_test)}")

    # --- 3. 拟合与评估模型 ---
    print(f"\n正在使用 K={args.k} 和 p={args.p} 进行预测...")
    model = KNNClassifier(p=args.p).fit(X_train, y_train)
    y_pred = model.predict(X_test, k=args.k)
    
    acc = float(np.mean(y_pred == y_test))
    print(f"✓ Digits 测试集准确率: {acc:.4f}")

    # --- 4. 保存可视化结果 ---
    current_experiment_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_experiment_dir, "digits_results")
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制并保存混淆矩阵热力图
    labels, cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    plot_confusion_matrix(labels, cm, ax_cm)
    fig_cm.tight_layout()
    cm_path = os.path.join(save_dir, "digits_confusion_matrix.png")
    fig_cm.savefig(cm_path, dpi=150)
    print(f"✓ 混淆矩阵图已保存至: {cm_path}")

    # 绘制并保存样本预测网格图
    rows, cols = args.grid_rows, args.grid_cols
    fig_grid, axarr = plt.subplots(rows, cols, figsize=(cols * 1.4, rows * 1.4))
    
    # 【已修正】移除了多余的参数
    plot_samples_grid(img_test, y_test, y_pred, rows, cols, axarr)
    
    fig_grid.suptitle("Digits 样本预测结果展示", y=0.98)
    fig_grid.tight_layout()
    grid_path = os.path.join(save_dir, "digits_samples.png")
    fig_grid.savefig(grid_path, dpi=150)
    print(f"✓ 样本预测网格图已保存至: {grid_path}")
    plt.close('all')

if __name__ == "__main__":
    main()