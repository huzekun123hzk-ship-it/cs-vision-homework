# knn/experiments/mnist_experiment.py

"""
在从网络下载的 MNIST (28x28) 数据集上运行 K-NN 的演示脚本。

本脚本会自动下载并缓存 MNIST 数据集至 data/mnist/ 目录，将图片展平为
784 维的、值在 [0,1] 区间的特征向量。为了加速运行，脚本默认使用
训练集和测试集的子集。最终会保存混淆矩阵和样本预测网格图。
"""
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# --- 导入自定义模块 ---
# 将上级目录 (knn/) 添加到 Python 的模块搜索路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from knn_classifier import KNNClassifier
# 【已修正】使用相对导入，明确告诉 Python 从当前文件夹导入
from .mnist_utils import load_mnist, mnist_to_flat_float32

def setup_matplotlib_font():
    """设置 Matplotlib 的中文字体。"""
    try:
        plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]
        plt.rcParams["axes.unicode_minus"] = False
    except:
        print("警告：中文字体设置失败，图表中的中文可能无法正常显示。")

# (为保持脚本独立，此处保留了绘图函数的副本)
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
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=6)
    return im

def plot_samples_grid(images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, rows: int, cols: int, axarr):
    """绘制样本预测结果的网格图。"""
    idx = np.arange(images.shape[0])[: rows * cols]
    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        ax = axarr[r, c]
        ax.imshow(images[i], cmap="gray")
        correct = (y_true[i] == y_pred[i])
        color = "green" if correct else "red"
        ax.set_title(f"真:{int(y_true[i])} 预:{int(y_pred[i])}", color=color, fontsize=7)
        ax.axis("off")

def main():
    """脚本主入口，用于 MNIST 分类实验。"""
    parser = argparse.ArgumentParser(description="在 MNIST 数据集上运行 K-NN (带自动下载功能)")
    parser.add_argument("--k", type=int, default=5, help="邻居数量 K 值。")
    parser.add_argument("--p", type=float, default=2.0, help="闵可夫斯基距离的阶数。")
    parser.add_argument("--train-subset", type=int, default=10000, help="使用前 N 个训练样本 (<=60000)。")
    parser.add_argument("--test-subset", type=int, default=2000, help="使用前 N 个测试样本 (<=10000)。")
    parser.add_argument("--seed", type=int, default=42, help="用于复现的随机种子。")
    parser.add_argument("--grid-rows", type=int, default=4, help="样本网格图的行数。")
    parser.add_argument("--grid-cols", type=int, default=8, help="样本网格图的列数。")
    args = parser.parse_args()

    setup_matplotlib_font()

    # --- 1. 下载和加载数据 ---
    current_experiment_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(current_experiment_dir, '../../data/mnist')
    
    Xtr_u8, ytr, Xte_u8, yte = load_mnist(root=data_root)

    # --- 2. 数据采样和预处理 ---
    rng = np.random.default_rng(args.seed)
    tr_idx = np.arange(Xtr_u8.shape[0]); rng.shuffle(tr_idx)
    te_idx = np.arange(Xte_u8.shape[0]); rng.shuffle(te_idx)
    
    Xtr_u8 = Xtr_u8[tr_idx][: args.train_subset]
    ytr = ytr[tr_idx][: args.train_subset]
    Xte_u8 = Xte_u8[te_idx][: args.test_subset]
    yte = yte[te_idx][: args.test_subset]

    Xtr = mnist_to_flat_float32(Xtr_u8)
    Xte = mnist_to_flat_float32(Xte_u8)
    print(f"✓ 数据加载和采样完成。训练集: {len(Xtr)}, 测试集: {len(Xte)}")

    # --- 3. 拟合与评估 ---
    print(f"\n正在使用 K={args.k} 和 p={args.p} 进行预测...")
    model = KNNClassifier(p=args.p).fit(Xtr, ytr)
    y_pred = model.predict(Xte, k=args.k)
    acc = float(np.mean(y_pred == yte))
    print(f"✓ MNIST 测试集准确率: {acc:.4f}")

    # --- 4. 保存可视化结果 ---
    save_dir = os.path.join(current_experiment_dir, "mnist_results")
    os.makedirs(save_dir, exist_ok=True)

    # 绘制并保存混淆矩阵
    labels, cm = confusion_matrix(yte, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    plot_confusion_matrix(labels, cm, ax_cm)
    fig_cm.tight_layout()
    cm_path = os.path.join(save_dir, "mnist_confusion_matrix.png")
    fig_cm.savefig(cm_path, dpi=150)
    print(f"✓ 混淆矩阵图已保存至: {cm_path}")

    # 绘制并保存样本预测网格图
    rows, cols = args.grid_rows, args.grid_cols
    fig_grid, axarr = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    plot_samples_grid(Xte_u8, yte, y_pred, rows, cols, axarr)
    fig_grid.suptitle("MNIST 样本预测结果展示", y=0.98)
    fig_grid.tight_layout()
    grid_path = os.path.join(save_dir, "mnist_samples.png")
    fig_grid.savefig(grid_path, dpi=150)
    print(f"✓ 样本预测网格图已保存至: {grid_path}")
    plt.close('all')

if __name__ == "__main__":
    main()