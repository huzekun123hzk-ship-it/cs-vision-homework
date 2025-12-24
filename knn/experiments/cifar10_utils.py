# knn/experiments/cifar10_utils.py

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# --- 全局常量 ---
CIFAR10_CLASSES = [
    '飞机', '汽车', '鸟类', '猫', '鹿',
    '狗', '青蛙', '马', '船', '卡车'
]

def load_cifar10_batch(file_path):
    """加载单个 CIFAR-10 batch 文件。"""
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='latin1')
        images = data_dict['data']
        labels = data_dict['labels']
        return images, labels

def load_cifar10(data_dir):
    """从指定目录加载完整的 CIFAR-10 数据集。"""
    X_train_list = []
    y_train_list = []
    for i in range(1, 6):
        file_path = os.path.join(data_dir, f'data_batch_{i}')
        images, labels = load_cifar10_batch(file_path)
        X_train_list.append(images)
        y_train_list.append(labels)
    
    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)

    test_file_path = os.path.join(data_dir, 'test_batch')
    X_test, y_test = load_cifar10_batch(test_file_path)
    y_test = np.array(y_test)
    
    return X_train, y_train, X_test, y_test

def visualize_samples(X, y, class_names, num_samples=10):
    """
    可视化 CIFAR-10 数据集中的样本图像，并将其保存到文件。

    Args:
        X (np.array): 图像数据，形状为 (N, 3072)。
        y (np.array): 标签数据，形状为 (N,)。
        class_names (list): 类别名称列表。
        num_samples (int): 要展示的样本数量。
    """
    # 在函数内部直接设置字体，确保生效
    try:
        plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        print(f"警告: 设置中文字体失败，预览图中的中文可能无法显示。错误: {e}")

    # 根据样本数量动态调整画布大小
    if num_samples == 1:
        plt.figure(figsize=(4, 4))
    else:
        plt.figure(figsize=(num_samples * 1.5, 3))

    for i in range(num_samples):
        idx = np.random.randint(0, X.shape[0])
        image_array = X[idx].reshape(3, 32, 32).transpose(1, 2, 0)

        # 如果是归一化后的数据，需要转换回 uint8
        if X.dtype == np.float32 or X.dtype == np.float64:
             image_display = (image_array * 255).astype('uint8')
        else:
             image_display = image_array

        if num_samples > 1:
            plt.subplot(1, num_samples, i + 1)

        plt.imshow(image_display)
        plt.title(f"{class_names[y[idx]]}")
        plt.axis('off')

    plt.suptitle("训练样本预览", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 将图像保存到文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "training_samples_preview.png")

    plt.savefig(save_path)
    plt.close() # 关闭图形，释放内存
    print(f"✓ 训练样本预览图已保存至: {save_path}")