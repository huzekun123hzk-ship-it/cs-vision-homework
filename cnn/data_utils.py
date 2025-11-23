# cnn/data_utils.py
import os
import pickle
import numpy as np


def load_cifar_batch(filename):
    """加载 CIFAR-10 中的单个 batch 文件。"""
    with open(filename, "rb") as f:
        datadict = pickle.load(f, encoding="latin1")
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(-1, 3, 32, 32).astype("float32")
        Y = np.array(Y, dtype=np.int64)
        return X, Y


def load_cifar10(root):
    """
    加载 CIFAR-10 整个数据集。

    返回:
        X_train, y_train, X_test, y_test
    """
    xs, ys = [], []
    for b in range(1, 6):
        f = os.path.join(root, f"data_batch_{b}")
        X, Y = load_cifar_batch(f)
        xs.append(X)
        ys.append(Y)
    X_train = np.concatenate(xs, axis=0)
    y_train = np.concatenate(ys, axis=0)

    X_test, y_test = load_cifar_batch(os.path.join(root, "test_batch"))
    return X_train, y_train, X_test, y_test


def preprocess_cifar10_data(
    cifar10_dir="./data/cifar-10-batches-py",
    num_training=49000,
    num_validation=1000,
    num_test=10000,
    subtract_mean=True,
    normalize=True,
):
    """
    加载并预处理 CIFAR-10：

    1. 划分 train / val / test
    2. 减去训练集均值
    3. 可选整体标准化
    """
    X_train, y_train, X_test, y_test = load_cifar10(cifar10_dir)

    # 划分
    mask = range(num_training)
    X_train_sub = X_train[mask]
    y_train_sub = y_train[mask]

    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = range(num_test)
    X_test_sub = X_test[mask]
    y_test_sub = y_test[mask]

    # 减均值
    mean_image = None
    if subtract_mean:
        mean_image = np.mean(X_train_sub, axis=0, keepdims=True)
        X_train_sub = X_train_sub - mean_image
        X_val = X_val - mean_image
        X_test_sub = X_test_sub - mean_image

    # 整体标准化（可选）
    if normalize:
        std = np.std(X_train_sub.reshape(num_training, -1), axis=0)
        std[std < 1e-8] = 1.0
        X_train_sub = X_train_sub / std.reshape(1, 3, 32, 32)
        X_val = X_val / std.reshape(1, 3, 32, 32)
        X_test_sub = X_test_sub / std.reshape(1, 3, 32, 32)

    data = {
        "X_train": X_train_sub,
        "y_train": y_train_sub,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test_sub,
        "y_test": y_test_sub,
        "mean_image": mean_image,
    }

    return data
