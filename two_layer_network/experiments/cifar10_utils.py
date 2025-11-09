"""
CIFAR-10 数据集工具

提供：
- 数据加载
- 数据预处理
- 数据增强
- 类别名称
"""

import pickle
import numpy as np
from pathlib import Path


def load_cifar10_batch(file_path):
    """加载单个CIFAR-10批次"""
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    data = batch[b'data']
    labels = batch[b'labels']
    return data, labels


def load_cifar10(data_dir, train_samples=40000, val_samples=5000, test_samples=5000):
    """
    加载CIFAR-10数据集并划分
    
    参数：
        data_dir: 数据目录路径
        train_samples: 训练集样本数
        val_samples: 验证集样本数
        test_samples: 测试集样本数
    
    返回：
        X_train, y_train: 训练集
        X_val, y_val: 验证集
        X_test, y_test: 测试集
    """
    data_dir = Path(data_dir)
    
    # 加载所有训练批次
    X_train_list = []
    y_train_list = []
    
    for i in range(1, 6):
        batch_file = data_dir / f'data_batch_{i}'
        X_batch, y_batch = load_cifar10_batch(batch_file)
        X_train_list.append(X_batch)
        y_train_list.append(y_batch)
    
    X_train_full = np.vstack(X_train_list)
    y_train_full = np.hstack(y_train_list).astype(int)
    
    # 划分训练集和验证集
    X_train = X_train_full[:train_samples]
    y_train = y_train_full[:train_samples]
    X_val = X_train_full[train_samples:train_samples + val_samples]
    y_val = y_train_full[train_samples:train_samples + val_samples]
    
    # 加载测试集
    test_file = data_dir / 'test_batch'
    X_test, y_test = load_cifar10_batch(test_file)
    X_test = X_test[:test_samples]
    y_test = np.array(y_test[:test_samples])
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess_data(X_train, X_val, X_test):
    """
    数据预处理：归一化、中心化、添加偏置
    
    参数：
        X_train, X_val, X_test: 原始数据
    
    返回：
        预处理后的数据
    """
    # 转换为浮点数并归一化到[0, 1]
    X_train = X_train.astype(np.float64) / 255.0
    X_val = X_val.astype(np.float64) / 255.0
    X_test = X_test.astype(np.float64) / 255.0
    
    # 计算训练集均值
    mean_image = np.mean(X_train, axis=0)
    
    # 中心化
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    
    return X_train, X_val, X_test


def get_cifar10_class_names():
    """返回CIFAR-10类别名称"""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']
