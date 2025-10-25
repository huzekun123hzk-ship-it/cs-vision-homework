"""
CIFAR-10数据集加载和预处理工具
"""

import numpy as np
import pickle
from pathlib import Path
from skimage.feature import hog  # <-- 新增
from skimage import color      # <-- 新增


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
    [HOG版-已修复] 预处理数据：
    1. (新) 提取 HOG 特征
    2. (旧) 中心化
    3. (旧) 归一化
    4. (旧) 添加偏置
    """
    
    print("  - [HOG版] 开始提取 HOG 特征...")
    
    # HOG 超参数
    ppc = 8
    cpb = 2
    
    # 定义一个辅助函数来处理单张图像
    def get_hog_features(image):
        image_gray = color.rgb2gray(image)
        features = hog(image_gray, 
                       pixels_per_cell=(ppc, ppc),
                       cells_per_block=(cpb, cpb),
                       orientations=9, 
                       block_norm='L2-Hys', 
                       visualize=False,
                       transform_sqrt=True)
        return features

    # 3. 对所有数据集并行提取特征
    X_train_hog = np.array([get_hog_features(img) for img in X_train])
    
    # ✨ [FIX] 有条件地提取 HOG, 兼容空数组
    hog_dim = X_train_hog.shape[1] # e.g., 324
    
    X_val_hog = np.array([get_hog_features(img) for img in X_val]) \
                if X_val.size > 0 else np.empty((0, hog_dim))
                
    X_test_hog = np.array([get_hog_features(img) for img in X_test]) \
                 if X_test.size > 0 else np.empty((0, hog_dim))
    
    print(f"  - [HOG版] 特征提取完成! 新的特征维度: {hog_dim}")
    
    # --- 后续流程 ---

    X_train_flat = X_train_hog.astype(np.float64)
    X_val_flat = X_val_hog.astype(np.float64)
    X_test_flat = X_test_hog.astype(np.float64)

    # 1. 计算训练集均值并中心化 (只对训练集)
    mean_feat = np.mean(X_train_flat, axis=0)
    X_train_flat -= mean_feat
    
    # 2. 归一化：除以标准差 (只对训练集)
    std_feat = np.std(X_train_flat, axis=0) + 1e-8 
    X_train_flat /= std_feat
    
    # 3. 添加偏置项 (只对训练集)
    X_train_processed = np.hstack([X_train_flat, np.ones((X_train_flat.shape[0], 1))])
    
    bias_dim = X_train_processed.shape[1] # e.g., 325

    # --- ✨ [FIX] 有条件地处理 Val 和 Test ---
    
    if X_val_flat.size > 0:
        X_val_flat -= mean_feat
        X_val_flat /= std_feat
        X_val_processed = np.hstack([X_val_flat, np.ones((X_val_flat.shape[0], 1))])
    else:
        X_val_processed = np.empty((0, bias_dim), dtype=np.float64)

    if X_test_flat.size > 0:
        X_test_flat -= mean_feat
        X_test_flat /= std_feat
        X_test_processed = np.hstack([X_test_flat, np.ones((X_test_flat.shape[0], 1))])
    else:
        X_test_processed = np.empty((0, bias_dim), dtype=np.float64)

    return X_train_processed, X_val_processed, X_test_processed

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
