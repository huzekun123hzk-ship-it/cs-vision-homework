#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化 CNN 在 CIFAR-10 测试集上的混淆矩阵。

用法示例：
    python -m cnn.visualize_confusion_cnn \
        --data-dir ./data/cifar-10-batches-py \
        --model-path ./cnn/experiments/results/cnn_cifar10_best.npz \
        --results-dir ./cnn/experiments/results
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 避免无显示环境报错
import matplotlib.pyplot as plt

from .model import Cifar10SimpleConvNet  # ✅ 注意类名

# ----------------- 兼容不同 data_utils 实现 -----------------
HAS_GET_HELPER = False
try:
    # 如果你的 data_utils 里已经有这个封装函数，就直接用
    from .data_utils import get_cifar10_data  # type: ignore
    HAS_GET_HELPER = True
except Exception:
    # 否则退回到最基础的 load_cifar10
    from .data_utils import load_cifar10  # type: ignore


CIFAR10_CLASS_NAMES = [
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def build_model():
    """
    按照训练时的结构，构建一个“空”模型，然后再加载参数覆盖。

    ⚠️ 一定要跟 experiment_cifar10_cnn.py 里创建模型的方式保持一致！
    如果你修改过那里的超参数，在这里也要同步改。
    """
    model = Cifar10SimpleConvNet(
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=3,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=1e-3,  # 如果你的 __init__ 里没有 reg 参数，就把这一行删掉
    )
    return model


def load_model_params(model, model_path):
    """
    从 npz 文件中加载参数到 model.params 里。
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"model_path 不存在：{model_path}")

    data = np.load(model_path)
    for k in model.params.keys():
        if k in data:
            model.params[k] = data[k]
        else:
            raise KeyError(f"在 npz 中找不到参数键：{k}")
    print(f"Loaded model parameters from {model_path}")


def load_cifar10_for_eval(data_dir):
    """
    封装一层：无论有没有 get_cifar10_data，最后都返回
    X_test, y_test，且做了减均值预处理。

    注意：这里先不强行改通道维度，统一在 main() 里做一次“找 size=3 的那一维搬到 axis=1”。
    """
    if HAS_GET_HELPER:
        # 你的 data_utils 里已经有这个函数，直接沿用训练时的逻辑
        data = get_cifar10_data(
            cifar10_dir=data_dir,
            num_training=49000,
            num_validation=1000,
            num_test=10000,
            subtract_mean=True,
        )
        X_test = data["X_test"]
        y_test = data["y_test"]
        print(f"[get_cifar10_data] 原始 X_test shape: {X_test.shape}")
        return X_test, y_test

    # 没有 get_cifar10_data：用最基础的 load_cifar10 自己做切分和减均值
    print("get_cifar10_data 未找到，使用 load_cifar10 手动切分数据 ……")
    X_train, y_train, X_test, y_test = load_cifar10(data_dir)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # 用训练集的均值做减均值
    mean_image = np.mean(X_train, axis=0, keepdims=True)
    X_test -= mean_image

    # 这里先不动通道维度，在 main() 再统一处理
    return X_test, y_test


def compute_confusion_matrix(y_true, y_pred, num_classes):
    """
    计算混淆矩阵：行是真实标签，列是预测标签。
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(cm, class_names, save_path,
                          normalize=True, cmap="Blues"):
    """
    绘制并保存混淆矩阵热力图。
    """
    if normalize:
        cm_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = cm.astype(np.float64) / np.maximum(cm_sum, 1)
        cm_show = cm_norm
        fmt = ".2f"
        title = "Confusion Matrix (normalized)"
    else:
        cm_show = cm
        fmt = "d"
        title = "Confusion Matrix (counts)"

    num_classes = cm.shape[0]

    plt.figure(figsize=(8, 7))
    im = plt.imshow(cm_show, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm_show.max() / 2.0
    for i in range(num_classes):
        for j in range(num_classes):
            value = cm_show[i, j]
            plt.text(
                j, i, format(value, fmt),
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if value > thresh else "black",
                fontsize=9,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix figure to {save_path}")


def ensure_nchw(X):
    """
    确保输入是 (N, C, H, W) 形式，其中 C=3。
    当前你遇到的是 (N, 32, 3, 32)，这里统一处理成 (N, 3, 32, 32)。
    """
    if X.ndim != 4:
        raise ValueError(f"期望 4D 张量，得到 shape={X.shape}")

    if X.shape[1] == 3:
        # 已经是 (N, 3, H, W)
        return X

    shape = X.shape
    if 3 not in shape:
        raise ValueError(f"在 X 的 shape={shape} 中找不到通道维度 size=3")

    # 找到哪个轴是 3，把它搬到 axis=1
    c_axis = int(np.where(np.array(shape) == 3)[0][0])
    X_moved = np.moveaxis(X, c_axis, 1)

    print(f"自动调整通道维度: 原始 shape={shape} -> 调整后 shape={X_moved.shape}")
    return X_moved


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CIFAR-10 CNN confusion matrix."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="CIFAR-10 原始 batches 的目录（cifar-10-batches-py）",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="训练好的模型参数 npz 文件路径",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./cnn/experiments/results",
        help="保存可视化结果的目录",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # 1. 加载数据（尽量和训练时预处理保持一致）
    X_test, y_test = load_cifar10_for_eval(args.data_dir)
    print(f"Loaded CIFAR-10 test data (raw): {X_test.shape}, {y_test.shape}")

    # 🔧 统一成 (N, 3, 32, 32)
    X_test = ensure_nchw(X_test)
    print(f"X_test after ensure_nchw: {X_test.shape}")

    # 2. 构建模型并加载参数
    model = build_model()
    load_model_params(model, args.model_path)

    # 3. 前向计算预测
    print("Running model on test set to get predictions...")
    scores = model.loss(X_test)  # 不传 y，只返回 scores
    y_pred = np.argmax(scores, axis=1)

    # 4. 计算整体准确率
    test_acc = np.mean(y_pred == y_test)
    print(f"Test accuracy (recomputed): {test_acc:.4f}")

    # 5. 混淆矩阵
    cm = compute_confusion_matrix(y_test, y_pred, num_classes=10)

    # 6. 按类别打印准确率
    print("Per-class accuracy:")
    for i, name in enumerate(CIFAR10_CLASS_NAMES):
        mask = (y_test == i)
        if np.sum(mask) == 0:
            acc_i = 0.0
        else:
            acc_i = np.mean(y_pred[mask] == y_test[mask])
        print(f"  {i} ({name:5s}): {acc_i:.4f}")

    # 7. 绘图并保存
    save_path_norm = os.path.join(args.results_dir, "cnn_confusion_matrix_normalized.png")
    save_path_cnt = os.path.join(args.results_dir, "cnn_confusion_matrix_counts.png")

    plot_confusion_matrix(cm, CIFAR10_CLASS_NAMES, save_path_norm, normalize=True)
    plot_confusion_matrix(cm, CIFAR10_CLASS_NAMES, save_path_cnt, normalize=False)


if __name__ == "__main__":
    main()
