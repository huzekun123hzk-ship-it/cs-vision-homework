# rnn/experiments/cifar10_utils.py
"""
CIFAR-10 数据集加载与预处理工具（RNN 版）

注意：这里的 preprocess 不再添加 bias 维度！
因为 RNNClassifier 自带 b_h / b_q，不需要 bias trick。
"""

from __future__ import annotations
import pickle
import numpy as np
from pathlib import Path


def unpickle(file_path: Path) -> dict:
    with open(file_path, "rb") as f:
        return pickle.load(f, encoding="bytes")


def load_cifar10(
    data_dir,
    train_samples: int = 49000,
    val_samples: int = 1000,
    test_samples: int = 1000,
):
    data_dir = Path(data_dir).expanduser()

    X_list, y_list = [], []
    for i in range(1, 6):
        p = data_dir / f"data_batch_{i}"
        if not p.exists():
            raise FileNotFoundError(f"Missing CIFAR-10 batch file: {p}")
        d = unpickle(p)
        X_list.append(d[b"data"])     # (10000,3072)
        y_list.append(d[b"labels"])   # list length 10000

    X_train_all = np.vstack(X_list)              # (50000,3072)
    y_train_all = np.hstack(y_list).astype(np.int64)

    test_p = data_dir / "test_batch"
    if not test_p.exists():
        raise FileNotFoundError(f"Missing CIFAR-10 test file: {test_p}")
    test_d = unpickle(test_p)
    X_test_all = test_d[b"data"]                 # (10000,3072)
    y_test_all = np.array(test_d[b"labels"], dtype=np.int64)

    # split
    X_val = X_train_all[train_samples:train_samples + val_samples]
    y_val = y_train_all[train_samples:train_samples + val_samples]

    X_train = X_train_all[:train_samples]
    y_train = y_train_all[:train_samples]

    X_test = X_test_all[:test_samples]
    y_test = y_test_all[:test_samples]

    # reshape to (N,32,32,3) for potential visualization
    def reshape_img(X):
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return X

    return reshape_img(X_train), y_train, reshape_img(X_val), y_val, reshape_img(X_test), y_test


def preprocess_data(X_train, X_val, X_test):
    """
    Standardize using train mean/std (per pixel).
    Inputs: (N,32,32,3)
    Outputs: flattened arrays (N,3072) float32
    """
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)
    X_val   = X_val.reshape(X_val.shape[0], -1).astype(np.float32)
    X_test  = X_test.reshape(X_test.shape[0], -1).astype(np.float32)

    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True) + 1e-7

    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    return X_train, X_val, X_test


def get_cifar10_class_names():
    return [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]
