# knn/knn_model.py

import numpy as np
from scipy import stats

class KNNImageClassifier:
    """基于 K-近邻 (K-NN) 算法的图像分类器。"""

    def __init__(self, k=3):
        """
        初始化 K-NN 分类器。

        Args:
            k (int): K-近邻算法中的 K 值（邻居数量）。
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        “训练”模型。对于 K-NN，这仅仅是存储训练数据。

        Args:
            X (np.array): 训练图像数据，形状为 (N, D)，N 为样本数，D 为特征维度。
            y (np.array): 训练标签数据，形状为 (N,)。
        """
        self.X_train = X
        self.y_train = y
        print("模型已“训练”，数据已存储。")

    def predict(self, X_test, num_loops=0):
        """
        对测试数据进行预测。

        该方法实现了不同循环层级的距离计算方式，以展示
        向量化计算的巨大性能优势。

        Args:
            X_test (np.array): 测试图像数据，形状为 (M, D)，M 为测试样本数。
            num_loops (int): 控制距离计算的实现方式。
                           0: 完全向量化，无显式 for 循环 (最高效)。
                           1: 半向量化，一层 for 循环 (遍历测试样本)。
                           2: 无向量化，两层 for 循环 (遍历测试和训练样本，最低效)。

        Returns:
            np.array: 对每个测试样本的预测标签，形状为 (M,)。
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("模型尚未“训练”，请先调用 fit 方法。")

        if num_loops == 0:
            # 完全向量化的实现 (无显式 for 循环)
            dists = self.compute_distances_no_loops(X_test)
        elif num_loops == 1:
            # 半向量化的实现 (一层 for 循环)
            dists = self.compute_distances_one_loop(X_test)
        elif num_loops == 2:
            # 无向量化的实现 (两层 for 循环)
            dists = self.compute_distances_two_loops(X_test)
        else:
            raise ValueError(f"无效的 num_loops 值 {num_loops}，应为 0, 1, 或 2。")

        return self.predict_labels(dists)

    def compute_distances_two_loops(self, X_test):
        """两层循环计算距离 (非常低效，仅用于教学对比)。"""
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                # 计算 L2 距离
                dists[i, j] = np.sqrt(np.sum((X_test[i] - self.X_train[j])**2))
        return dists

    def compute_distances_one_loop(self, X_test):
        """一层循环计算距离 (比两层循环高效)。"""
        num_test = X_test.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            # 利用 Numpy 的广播机制
            dists[i, :] = np.sqrt(np.sum((self.X_train - X_test[i, :])**2, axis=1))
        return dists

    def compute_distances_no_loops(self, X_test):
        """完全向量化计算距离 (最高效)。"""
        # 利用广播机制和矩阵运算: (a-b)^2 = a^2 - 2ab + b^2
        test_sum_sq = np.sum(X_test**2, axis=1, keepdims=True)
        train_sum_sq = np.sum(self.X_train**2, axis=1, keepdims=True).T
        inner_product = np.dot(X_test, self.X_train.T)
        
        # dists_sq 存储的是距离的平方
        dists_sq = test_sum_sq - 2 * inner_product + train_sum_sq
        return np.sqrt(dists_sq)

    def predict_labels(self, dists):
        """根据距离矩阵，为每个测试点预测标签。"""
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # 找到距离最近的 k 个训练样本的索引
            k_closest_indices = np.argsort(dists[i, :])[:self.k]
            # 获取这些邻居的标签
            k_closest_labels = self.y_train[k_closest_indices]
            # 通过投票法（找众数）确定最终类别
            # stats.mode 返回众数和其出现次数，我们只需要众数
            mode_result = stats.mode(k_closest_labels)
            y_pred[i] = mode_result.mode
        return y_pred