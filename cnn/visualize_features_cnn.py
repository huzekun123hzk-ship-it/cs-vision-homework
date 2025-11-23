#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化训练好的 CNN 的：
1. 第一层卷积核（filters）
2. 若干样本在第一层卷积后的特征图（feature maps）

用法示例：
    python -m cnn.visualize_features_cnn \
        --data-dir ./data/cifar-10-batches-py \
        --model-path ./cnn/experiments/results/cnn_cifar10_best.npz \
        --results-dir ./cnn/experiments/results \
        --num-filters 16 \
        --num-samples 5
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 避免无显示环境报错
import matplotlib.pyplot as plt

from .model import Cifar10SimpleConvNet

# ------- data_utils 兼容：优先用 get_cifar10_data，兜底用 load_cifar10 -------
HAS_GET_HELPER = False
try:
    from .data_utils import get_cifar10_data  # type: ignore
    HAS_GET_HELPER = True
except Exception:
    from .data_utils import load_cifar10  # type: ignore


def build_model():
    """
    和训练脚本保持同样的模型结构。
    ⚠️ 要和 experiment_cifar10_cnn.py 里的超参数一致！
    """
    model = Cifar10SimpleConvNet(
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=3,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=1e-3,
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


def load_cifar10_for_features(data_dir):
    """
    返回归一化后的 X_test（还没保证是 NCHW），只用来做特征可视化。
    """
    if HAS_GET_HELPER:
        data = get_cifar10_data(
            cifar10_dir=data_dir,
            num_training=49000,
            num_validation=1000,
            num_test=10000,
            subtract_mean=True,
        )
        X_test = data["X_test"]
        print(f"[get_cifar10_data] 原始 X_test shape: {X_test.shape}")
        return X_test

    print("get_cifar10_data 未找到，使用 load_cifar10 手动切分数据 ……")
    X_train, y_train, X_test, y_test = load_cifar10(data_dir)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    mean_image = np.mean(X_train, axis=0, keepdims=True)
    X_test -= mean_image

    # 注意：这里不改通道维度，在 ensure_nchw 里统一处理
    print(f"[load_cifar10] 原始 X_test shape: {X_test.shape}")
    return X_test


def ensure_nchw(X):
    """
    确保输入是 (N, C, H, W) 且 C=3。
    比如当前你出现过 (N, 32, 3, 32)，就需要自动把 size=3 的轴移动到 axis=1。
    """
    if X.ndim != 4:
        raise ValueError(f"期望 4D 张量，得到 shape={X.shape}")

    if X.shape[1] == 3:  # 已经是 (N, 3, H, W)
        return X

    shape = X.shape
    if 3 not in shape:
        raise ValueError(f"在 X 的 shape={shape} 中找不到通道维度 size=3")

    c_axis = int(np.where(np.array(shape) == 3)[0][0])
    X_moved = np.moveaxis(X, c_axis, 1)

    print(f"自动调整通道维度: 原始 shape={shape} -> 调整后 shape={X_moved.shape}")
    return X_moved


def make_grid(data, pad=1):
    """
    把一组小图像（N, C, H, W）排成网格，输出单张大图（H_grid, W_grid, C）。

    Args:
        data: (N, C, H, W)
        pad:  padding 像素

    Returns:
        grid: (H_grid, W_grid, C)
    """
    N, C, H, W = data.shape
    grid_size = int(np.ceil(np.sqrt(N)))
    grid_H = grid_size * H + (grid_size - 1) * pad
    grid_W = grid_size * W + (grid_size - 1) * pad

    grid = np.zeros((grid_H, grid_W, C), dtype=data.dtype)

    idx = 0
    for y in range(grid_size):
        for x in range(grid_size):
            if idx >= N:
                break
            h_start = y * (H + pad)
            w_start = x * (W + pad)
            img = data[idx].transpose(1, 2, 0)  # C, H, W -> H, W, C
            grid[h_start:h_start + H, w_start:w_start + W, :] = img
            idx += 1
    return grid


def visualize_conv1_filters(W1, save_path, max_filters=32):
    """
    可视化第一层卷积核。
    W1 形状： (F, C, HH, WW)
    """
    F, C, HH, WW = W1.shape
    num_show = min(F, max_filters)
    W_show = W1[:num_show].copy()

    W_min = W_show.min()
    W_max = W_show.max()
    if W_max > W_min:
        W_show = (W_show - W_min) / (W_max - W_min)
    else:
        W_show = np.zeros_like(W_show)

    grid = make_grid(W_show, pad=2)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid)
    plt.axis("off")
    plt.title(f"Conv1 filters (first {num_show})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved conv1 filters visualization to {save_path}")


def conv_forward_naive(X, W, b, conv_param):
    """
    简单的卷积前向（只用于可视化，不追求极致效率）。

    X: (N, C, H, W)
    W: (F, C, HH, WW)
    b: (F,)
    conv_param: dict, must contain 'stride' and 'pad'
    """
    stride = conv_param.get("stride", 1)
    pad = conv_param.get("pad", 0)

    N, C, H, W_in = X.shape
    F, _, HH, WW = W.shape

    X_padded = np.pad(
        X,
        pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)),
        mode="constant",
        constant_values=0,
    )

    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W_in + 2 * pad - WW) // stride

    out = np.zeros((N, F, H_out, W_out), dtype=X.dtype)

    for n in range(N):
        for f in range(F):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    w_start = j * stride
                    x_slice = X_padded[n, :, h_start:h_start + HH, w_start:w_start + WW]
                    out[n, f, i, j] = np.sum(x_slice * W[f]) + b[f]

    return out


def relu_forward(x):
    return np.maximum(0, x)


def visualize_feature_maps(X, W1, b1, save_dir, num_samples=5):
    """
    对若干 test 样本，跑一层 conv+ReLU，然后把 feature maps 可视化出来。

    X: (N, C, H, W)，已经是归一化后的 test 图像
    W1, b1: 第一层卷积核参数
    """
    N_available = X.shape[0]
    N = min(num_samples, N_available)
    X_sub = X[:N]

    conv_param = {"stride": 1, "pad": 1}
    feat = conv_forward_naive(X_sub, W1, b1, conv_param)
    feat = relu_forward(feat)  # (N, F, H_out, W_out)

    N, F, H_out, W_out = feat.shape
    num_channels_show = min(F, 8)  # 每个样本展示 8 个 feature map

    for i in range(N):
        fmaps = feat[i, :num_channels_show]  # (C_show, H_out, W_out)

        f_min = fmaps.min()
        f_max = fmaps.max()
        if f_max > f_min:
            fmaps_norm = (fmaps - f_min) / (f_max - f_min)
        else:
            fmaps_norm = np.zeros_like(fmaps)

        # 变成 (N, 1, H, W) 以复用 make_grid
        fmaps_norm = fmaps_norm[:, None, :, :]  # (C_show, 1, H, W)
        grid = make_grid(fmaps_norm, pad=1)

        plt.figure(figsize=(6, 6))
        plt.imshow(grid[:, :, 0], cmap="gray")
        plt.axis("off")
        plt.title(f"Feature maps of sample #{i}")
        plt.tight_layout()

        fname = os.path.join(save_dir, f"cnn_feature_maps_sample_{i}.png")
        plt.savefig(fname, dpi=200)
        plt.close()
        print(f"Saved feature maps for sample {i} to {fname}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CNN filters and feature maps on CIFAR-10."
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
    parser.add_argument(
        "--num-filters",
        type=int,
        default=32,
        help="最多可视化多少个 conv1 filter",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="可视化多少个样本的特征图",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # 1. 加载数据（只要 test）
    X_test = load_cifar10_for_features(args.data_dir)
    print(f"Loaded CIFAR-10 test data (raw): {X_test.shape}")

    # 统一成 (N, 3, 32, 32)
    X_test = ensure_nchw(X_test)
    print(f"X_test after ensure_nchw: {X_test.shape}")

    # 2. 构建模型并加载参数
    model = build_model()
    load_model_params(model, args.model_path)

    # 3. 取出第一层参数
    W1 = model.params["W1"]
    b1 = model.params["b1"]

    # 4. 可视化卷积核
    filters_save_path = os.path.join(args.results_dir, "cnn_conv1_filters.png")
    visualize_conv1_filters(W1, filters_save_path, max_filters=args.num_filters)

    # 5. 可视化特征图
    visualize_feature_maps(
        X_test,
        W1,
        b1,
        save_dir=args.results_dir,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
