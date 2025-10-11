# knn/experiments/image_folder.py

"""
在简单的图像文件夹数据集上运行 K-NN 的演示脚本。

期望的文件夹结构如下:

root/
  class_a/*.jpg|png|jpeg|bmp
  class_b/*.jpg|png|jpeg|bmp
  ...

所有图片将被转换为灰度图，缩放到统一尺寸，展平为一维的、值在 [0, 1] 区间的
特征向量，然后用 K-NN 进行分类。
"""
import argparse
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- 导入自定义模块 ---
# 将上级目录 (knn/) 添加到 Python 的模块搜索路径中
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from knn_classifier import KNNClassifier, train_test_split_indices
# 从当前目录导入辅助工具
from .image_folder_utils import download_and_extract

def setup_matplotlib_font():
    """设置 Matplotlib 的中文字体。"""
    try:
        plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]
        plt.rcParams["axes.unicode_minus"] = False
    except:
        print("警告：中文字体设置失败，图表中的中文可能无法正常显示。")

def load_image_folder(root_dir: str, image_size=(32, 32)):
    """
    从按类别存放的子文件夹中加载图像，并转换为特征向量。
    """
    X_list = []
    y_list = []
    class_names = []
    root = Path(root_dir)

    # 自动下探一层
    direct_subdirs = [p for p in root.iterdir() if p.is_dir()]
    if len(direct_subdirs) == 1:
        inner = direct_subdirs[0]
        inner_subdirs = [p for p in inner.iterdir() if p.is_dir()]
        if len(inner_subdirs) >= 2:
            root = inner

    print(f"正在从 '{root}' 加载图片...")
    for cls_idx, cls_name in enumerate(sorted([p.name for p in root.iterdir() if p.is_dir()])):
        class_names.append(cls_name)
        cls_dir = root / cls_name
        image_count = 0
        for img_path in cls_dir.rglob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
                continue
            try:
                with Image.open(img_path) as im:
                    im = im.convert("L").resize(image_size)
                    arr = np.asarray(im, dtype=np.float32) / 255.0
                    X_list.append(arr.flatten())
                    y_list.append(cls_idx)
                    image_count += 1
            except Exception:
                continue
        print(f"  - 找到类别 '{cls_name}'，加载了 {image_count} 张图片。")

    if not X_list:
        raise RuntimeError(f"在 '{root_dir}' 中没有加载到任何图片。请确保文件夹结构正确: root/class_a/*.jpg ...")

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=int)
    return X, y, class_names, root

def save_predictions_csv(path: str, y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]):
    """将每个样本的预测结果、真实标签和是否正确保存为 CSV 文件。"""
    correct = (y_true == y_pred).astype(int)
    # 将数字标签转换为可读的类别名称
    y_true_names = [class_names[i] for i in y_true]
    y_pred_names = [class_names[i] for i in y_pred]
    
    header = "索引,真实标签,预测标签,是否正确"
    # 使用 np.savetxt 保存混合了字符串和数字的数据会比较复杂
    # 因此我们直接用 Python 的文件写入功能
    with open(path, 'w', encoding='utf-8') as f:
        f.write(header + '\n')
        for i in range(len(y_true)):
            f.write(f"{i},{y_true_names[i]},{y_pred_names[i]},{correct[i]}\n")


def confusion_matrix(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((labels.size, labels.size), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1
    return labels, cm

def plot_confusion_matrix(labels, cm, class_names, ax):
    label_names = [class_names[int(lab)] for lab in labels]
    ax.imshow(cm, cmap="Blues")
    ax.set_title("混淆矩阵 (行=真实, 列=预测)")
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(label_names, rotation=30, ha="right")
    ax.set_yticklabels(label_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=8)

def plot_samples_grid(X, y_true, y_pred, rows, cols, image_size, class_names, axarr):
    idx = np.arange(X.shape[0])[: rows * cols]
    H, W = image_size
    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        ax = axarr[r, c]
        ax.imshow(X[i].reshape(H, W), cmap="gray")
        correct = (y_true[i] == y_pred[i])
        color = "green" if correct else "red"
        t_name = class_names[int(y_true[i])]
        p_name = class_names[int(y_pred[i])]
        ax.set_title(f"真:{t_name}\n预:{p_name}", color=color, fontsize=8)
        ax.axis("off")

def main():
    """脚本主入口，用于在自定义图像文件夹上运行K-NN实验。"""
    parser = argparse.ArgumentParser(description="在自定义图像文件夹数据集上运行 K-NN")
    parser.add_argument("--data-dir", type=str, required=True, help="包含类别子文件夹的根目录。")
    parser.add_argument("--k", type=int, default=5, help="邻居数量 K 值。")
    parser.add_argument("--p", type=float, default=2.0, help="闵可夫斯基距离的阶数。")
    parser.add_argument("--test-ratio", type=float, default=0.3, help="测试集所占比例。")
    parser.add_argument("--seed", type=int, default=42, help="用于复现的随机种子。")
    parser.add_argument("--image-size", type=int, nargs=2, default=(32, 32), help="图像缩放尺寸 (高 宽)。")
    parser.add_argument("--download-url", type=str, default=None, help="一个指向数据集压缩包的可选URL。")
    parser.add_argument("--grid-rows", type=int, default=4, help="样本网格图的行数。")
    parser.add_argument("--grid-cols", type=int, default=8, help="样本网格图的列数。")
    args = parser.parse_args()
    
    setup_matplotlib_font()

    # --- 1. 下载和加载数据 ---
    if args.download_url:
        print(f"检测到下载链接，将尝试下载并解压至 '{args.data_dir}'...")
        download_and_extract(args.download_url, args.data_dir)

    X, y, class_names, root_used = load_image_folder(args.data_dir, image_size=tuple(args.image_size))
    print(f"✓ 数据加载完成。共找到 {len(class_names)} 个类别: {class_names}。总样本数: {len(X)}")

    # --- 2. 划分训练集和测试集 ---
    rng = np.random.default_rng(args.seed)
    train_idx, test_idx = train_test_split_indices(len(y), test_ratio=args.test_ratio, rng=rng)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"  - 训练集大小: {len(X_train)}")
    print(f"  - 测试集大小: {len(X_test)}")

    # --- 3. 拟合与评估 ---
    print(f"\n正在使用 K={args.k} 和 p={args.p} 进行预测...")
    model = KNNClassifier(p=args.p).fit(X_train, y_train)
    y_pred = model.predict(X_test, k=args.k)
    acc = float(np.mean(y_pred == y_test))
    dataset_tag = root_used.name
    print(f"✓ {dataset_tag} 测试集准确率: {acc:.4f}")

    # --- 4. 保存结果 ---
    current_experiment_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_experiment_dir, f"{dataset_tag}_results")
    os.makedirs(save_dir, exist_ok=True)

    # 保存 CSV 格式的详细预测结果
    preds_csv_path = os.path.join(save_dir, "predictions.csv")
    save_predictions_csv(preds_csv_path, y_test, y_pred, class_names)
    print(f"✓ 详细预测结果已保存至: {preds_csv_path}")

    # 绘制并保存混淆矩阵
    labels, cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(max(5, len(class_names)*0.8), max(4, len(class_names)*0.6)))
    plot_confusion_matrix(labels, cm, class_names, ax_cm)
    fig_cm.suptitle(f"准确率={acc:.4f} | K={args.k}, 尺寸={tuple(args.image_size)}", y=1.02, fontsize=9)
    fig_cm.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    fig_cm.savefig(cm_path, dpi=150)
    print(f"✓ 混淆矩阵图已保存至: {cm_path}")

    # 绘制并保存样本预测网格图
    rows, cols = args.grid_rows, args.grid_cols
    fig_grid, axarr = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    plot_samples_grid(X_test, y_test, y_pred, rows, cols, tuple(args.image_size), class_names, axarr)
    fig_grid.suptitle(f"{dataset_tag} 样本预测结果", y=0.98, fontsize=12)
    fig_grid.tight_layout()
    grid_path = os.path.join(save_dir, "samples_grid.png")
    fig_grid.savefig(grid_path, dpi=150)
    print(f"✓ 样本预测网格图已保存至: {grid_path}")
    plt.close('all')

if __name__ == "__main__":
    main()