"""
CIFAR-10数据集加载和预处理工具
"""

import numpy as np
import pickle
from pathlib import Path


def unpickle(file):
    """
    解压CIFAR-10数据文件
    
    Args:
        file: pickle文件路径
        
    Returns:
        dict: 包含'data'和'labels'的字典
    """
    with open(file, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    return data_dict


def load_cifar10(data_dir = 'data/cifar-10-batches-py',
                 train_samples=49000, val_samples=1000, test_samples=1000):
    """
    加载CIFAR-10数据集
    
    Args:
        data_dir: CIFAR-10数据目录
        train_samples: 训练集样本数
        val_samples: 验证集样本数  
        test_samples: 测试集样本数
        
    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    data_dir = Path(data_dir).expanduser()
    
    # 加载训练集
    X_train_list = []
    y_train_list = []
    
    for i in range(1, 6):
        batch_file = data_dir / f'data_batch_{i}'
        batch_dict = unpickle(batch_file)
        X_train_list.append(batch_dict[b'data'])
        y_train_list.append(batch_dict[b'labels'])
    
    X_train_all = np.vstack(X_train_list)
    y_train_all = np.hstack(y_train_list)
    
    # 分割训练集和验证集
    X_train = X_train_all[:train_samples]
    y_train = y_train_all[:train_samples]
    X_val = X_train_all[train_samples:train_samples+val_samples]
    y_val = y_train_all[train_samples:train_samples+val_samples]
    
    # 加载测试集
    test_dict = unpickle(data_dir / 'test_batch')
    X_test = test_dict[b'data'][:test_samples]
    y_test = np.array(test_dict[b'labels'][:test_samples])
    
    # 重塑为图像格式 [N, 32, 32, 3]
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_val = X_val.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocess_data(X_train, X_val, X_test):
    """
    预处理数据：展平、中心化、归一化、添加偏置
    
    Args:
        X_train: 训练集图像 [N, H, W, C]
        X_val: 验证集图像
        X_test: 测试集图像
        
    Returns:
        (X_train_processed, X_val_processed, X_test_processed)
        每个形状为 [N, D+1]，D=H*W*C，+1是偏置项
    """
    # 展平图像
    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype(np.float64)
    X_val_flat = X_val.reshape(X_val.shape[0], -1).astype(np.float64)
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype(np.float64)
    
    # 1. 计算训练集均值并中心化
    mean_image = np.mean(X_train_flat, axis=0)
    X_train_flat -= mean_image
    X_val_flat -= mean_image
    X_test_flat -= mean_image
    
    # 2. 归一化：除以标准差
    std = np.std(X_train_flat, axis=0) + 1e-8  # 避免除零
    X_train_flat /= std
    X_val_flat /= std
    X_test_flat /= std
    
    # 3. 添加偏置项
    X_train_flat = np.hstack([X_train_flat, np.ones((X_train_flat.shape[0], 1))])
    X_val_flat = np.hstack([X_val_flat, np.ones((X_val_flat.shape[0], 1))])
    X_test_flat = np.hstack([X_test_flat, np.ones((X_test_flat.shape[0], 1))])
    
    return X_train_flat, X_val_flat, X_test_flat


def get_cifar10_class_names():
    """
    获取CIFAR-10类别名称
    
    Returns:
        list: 10个类别名称
    """
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]


# CIFAR-10类别名称（向后兼容）
CIFAR10_CLASSES = get_cifar10_class_names()
