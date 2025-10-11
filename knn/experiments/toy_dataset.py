# knn/experiments/toy_dataset.py

"""
在二维“玩具”数据集上运行 K-NN 的演示脚本。

本脚本的特点:
- 生成两个二维高斯分布的数据团（用于二元分类）。
- 训练一个 KNNClassifier 并评估其测试集准确率。
- 保存决策边界图、CSV格式的预测详情、以及一个展示预测正确性的散点图。
"""
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

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


def make_toy_data(n_per_class: int = 100, rng: np.random.Generator | None = None):
    """为二元分类任务生成两个二维高斯数据团。"""
    if rng is None:
        rng = np.random.default_rng(42)
    # 类别中心和共享的协方差矩阵
    mean0 = np.array([-2.0, -2.0])
    mean1 = np.array([+2.0, +2.0])
    cov = np.array([[1.0, 0.2], [0.2, 1.0]])

    X0 = rng.multivariate_normal(mean0, cov, size=n_per_class)
    X1 = rng.multivariate_normal(mean1, cov, size=n_per_class)
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)])
    return X, y

def plot_decision_boundary(model: KNNClassifier, X: np.ndarray, y: np.ndarray, k: int, p: float, ax: plt.Axes):
    """在一个密集的二维网格上绘制 K-NN 决策区域，并叠加训练数据点。"""
    x_min, x_max = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
    y_min, y_max = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    y_pred_grid = model.predict(grid, k=k)
    Z = y_pred_grid.reshape(xx.shape)

    cmap_bg = plt.cm.Pastel2
    cmap_pts = plt.cm.Set1

    ax.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_bg)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_pts, edgecolors="k", s=40)
    ax.set_title(f"K-NN 决策边界 (k={k}, p={p})")
    ax.set_xlabel("特征 x1")
    ax.set_ylabel("特征 x2")

def save_predictions_csv(path: str, y_true: np.ndarray, y_pred: np.ndarray):
    """将每个样本的预测结果、真实标签和是否正确保存为 CSV 文件。"""
    correct = (y_true == y_pred).astype(int)
    header = "索引,真实标签,预测标签,是否正确"
    rows = np.column_stack([np.arange(y_true.shape[0]), y_true, y_pred, correct])
    np.savetxt(path, rows, fmt="%d", delimiter=",", header=header, comments="")

def plot_test_correctness(X_test: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, ax: plt.Axes):
    """绘制测试点的散点图，并用颜色（绿/红）标注预测是否正确。"""
    correct = y_true == y_pred
    ax.scatter(X_test[correct, 0], X_test[correct, 1], c="green", edgecolors="k", s=40, label="预测正确")
    ax.scatter(X_test[~correct, 0], X_test[~correct, 1], c="red", edgecolors="k", s=40, label="预测错误")
    ax.legend(loc="best")
    ax.set_title("测试集预测正确性分布")
    ax.set_xlabel("特征 x1")
    ax.set_ylabel("特征 x2")

def main():
    """脚本主入口：在玩具数据集上训练、评估 K-NN 并保存结果。"""
    parser = argparse.ArgumentParser(description="纯 NumPy 实现的 K-NN 玩具数据集演示")
    parser.add_argument("--k", type=int, default=5, help="邻居数量 K 值。")
    parser.add_argument("--p", type=float, default=2.0, help="闵可夫斯基距离的阶数 (p=1 曼哈顿, p=2 欧氏)。")
    parser.add_argument("--n-per-class", type=int, default=150, help="每个类别的样本数量。")
    parser.add_argument("--test-ratio", type=float, default=0.3, help="测试集所占比例。")
    parser.add_argument("--seed", type=int, default=42, help="用于复现的随机种子。")
    args = parser.parse_args()

    # --- 0. 全局设置 ---
    setup_matplotlib_font()

    # --- 1. 生成数据并划分 ---
    print("正在生成玩具数据集...")
    rng = np.random.default_rng(args.seed)
    X, y = make_toy_data(n_per_class=args.n_per_class, rng=rng)
    train_idx, test_idx = train_test_split_indices(len(y), test_ratio=args.test_ratio, rng=rng)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"✓ 数据生成完毕。训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # --- 2. 拟合与预测 ---
    print(f"\n正在使用 K={args.k} 和 p={args.p} 进行预测...")
    model = KNNClassifier(p=args.p).fit(X_train, y_train)
    y_pred = model.predict(X_test, k=args.k)
    acc = float(np.mean(y_pred == y_test))
    print(f"✓ 测试集准确率: {acc:.4f}")

    # --- 3. 保存结果 ---
    save_dir = os.path.join(os.path.dirname(__file__), "toy_dataset_results")
    os.makedirs(save_dir, exist_ok=True)

    # 保存 CSV 格式的详细预测结果
    preds_csv_path = os.path.join(save_dir, "predictions.csv")
    save_predictions_csv(preds_csv_path, y_test, y_pred)
    print(f"✓ 详细预测结果已保存至: {preds_csv_path}")

    # 绘制并保存决策边界图 (在训练集上绘制)
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    plot_decision_boundary(model, X_train, y_train, k=args.k, p=args.p, ax=ax1)
    plt.tight_layout()
    boundary_path = os.path.join(save_dir, f"decision_boundary_k{args.k}_p{int(args.p)}.png")
    fig1.savefig(boundary_path, dpi=150)
    print(f"✓ 决策边界图已保存至: {boundary_path}")

    # 绘制并保存测试集正确性图
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    plot_test_correctness(X_test, y_test, y_pred, ax2)
    plt.tight_layout()
    correctness_path = os.path.join(save_dir, "test_correctness.png")
    fig2.savefig(correctness_path, dpi=150)
    print(f"✓ 预测正确性图已保存至: {correctness_path}")
    plt.close('all')

if __name__ == "__main__":
    main()