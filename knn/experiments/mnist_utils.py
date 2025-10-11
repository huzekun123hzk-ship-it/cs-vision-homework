# knn/experiments/mnist_utils.py

"""
一个最小化的 MNIST 下载器/加载器 

本脚本会从镜像源下载原始的 IDX 格式的 MNIST 文件，并将其缓存
在 data/mnist/ 目录下。提供函数来将训练集和测试集加载为 NumPy 数组。
"""
import gzip
import os
import struct
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np

# 公开的镜像源 (按顺序尝试)
MNIST_URLS = {
    "train_images": [
        "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    ],
    "train_labels": [
        "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    ],
    "test_images": [
        "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    ],
    "test_labels": [
        "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
        "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
    ],
}

# 用于健全性检查的幻数 (Magic Number)
IMAGE_MAGIC = 2051
LABEL_MAGIC = 2049


def _download(urls: list[str], dest_path: Path) -> None:
    """从给定的 URL 列表中下载文件。"""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            print(f"正在下载 {url} -> {dest_path}")
            urllib.request.urlretrieve(url, dest_path)
            print("✓ 下载成功。")
            return
        except Exception as e:
            print(f"从 {url} 下载失败: {e}")
    raise RuntimeError(f"所有镜像源都无法下载 {dest_path.name}")


def _read_idx_images(gz_path: Path) -> np.ndarray:
    """从 IDX 格式的 Gzip 压缩包中读取图像数据。"""
    with gzip.open(gz_path, "rb") as f:
        # >IIII 表示使用大端序读取4个无符号整数
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != IMAGE_MAGIC:
            raise ValueError("无效的图像文件幻数")
        # 读取所有图像数据到一个缓冲区
        buf = f.read(rows * cols * num)
        # 从缓冲区创建 NumPy 数组
        data = np.frombuffer(buf, dtype=np.uint8)
        # 将一维数组重塑为 (n_images, height, width)
        return data.reshape(num, rows, cols)


def _read_idx_labels(gz_path: Path) -> np.ndarray:
    """从 IDX 格式的 Gzip 压缩包中读取标签数据。"""
    with gzip.open(gz_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != LABEL_MAGIC:
            raise ValueError("无效的标签文件幻数")
        buf = f.read(num)
        data = np.frombuffer(buf, dtype=np.uint8)
        return data


def load_mnist(root: str = "data/mnist") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    下载 (如果需要) 并加载 MNIST 训练集/测试集为 NumPy 数组。

    返回
    -------
    X_train : (60000, 28, 28) uint8 格式的训练图像
    y_train : (60000,) uint8 格式的训练标签
    X_test : (10000, 28, 28) uint8 格式的测试图像
    y_test : (10000,) uint8 格式的测试标签
    """
    root_path = Path(root)
    paths = {
        "train_images": root_path / "train-images-idx3-ubyte.gz",
        "train_labels": root_path / "train-labels-idx1-ubyte.gz",
        "test_images": root_path / "t10k-images-idx3-ubyte.gz",
        "test_labels": root_path / "t10k-labels-idx1-ubyte.gz",
    }

    # 检查每个文件是否存在，如果不存在则下载
    for key, p in paths.items():
        if not p.exists():
            _download(MNIST_URLS[key], p)

    print("正在解析 MNIST 数据文件...")
    X_train = _read_idx_images(paths["train_images"])
    y_train = _read_idx_labels(paths["train_labels"])
    X_test = _read_idx_images(paths["test_images"])
    y_test = _read_idx_labels(paths["test_labels"])
    print("✓ 解析完成。")
    return X_train, y_train, X_test, y_test


def mnist_to_flat_float32(X: np.ndarray) -> np.ndarray:
    """将 (N, 28, 28) uint8 数组转换为 (N, 784) float32 数组，值在 [0, 1] 区间。"""
    Xf = X.astype(np.float32) / 255.0
    return Xf.reshape(Xf.shape[0], -1)