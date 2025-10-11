# knn/knn_classifier.py

"""
使用纯 NumPy 实现的 K-近邻 (KNN) 分类器。

本模块提供:
- KNNClassifier: 一个基于内存、使用闵可夫斯基距离的分类器。
- train_test_split_indices: 一个用于可复现地拆分索引的辅助函数。

设计说明:
- 距离计算是向量化的；当 p=2 (欧氏距离) 时，使用了点积技巧进行优化。
- 对于 p!=2 的大规模输入，预测过程会通过对查询批次和特征块进行分块处理，
  以保证内存安全，避免分配 (M, N, D) 形状的巨大临时数组。
- 邻居选择使用 np.argpartition 进行部分排序，以提升性能。
- 投票平局的打破规则是确定性的：多数票 -> 距离和 -> 标签值。
"""
from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


class KNNClassifier:
    """一个使用纯 NumPy 实现的 K-近邻分类器。

    该分类器存储训练数据集，并为每个查询样本计算到所有训练样本的距离，
    检索 K 个最近邻的索引，并通过带有确定性平局打破规则的多数投票来预测标签。

    参数
    ----------
    p : float, default=2.0
        闵可夫斯基距离的阶数。p=2 对应欧氏距离，p=1 对应曼哈顿距离。
        必须为正数。
    """

    def __init__(self, p: float = 2.0) -> None:
        if p <= 0:
            raise ValueError("参数 p 对于闵可夫斯基距离必须为正数")
        self.p: float = p
        # 训练特征和标签在 fit 方法中被赋值
        self._X: Optional[np.ndarray] = None
        self._y: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """
        存储训练数据，此过程不学习任何参数。

        KNN 是一种基于内存的方法。本函数验证输入数组的形状，并保留
        对所提供数组的引用（调用者应避免在外部修改它们）。

        参数
        ----------
        X : np.ndarray, 形状为 (n_samples, n_features)
            训练特征矩阵。
        y : np.ndarray, 形状为 (n_samples,)
            训练标签 (整数或字符串)。

        返回
        -------
        self : KNNClassifier
            经过“拟合”的分类器实例。
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError("输入 X 必须是形状为 (n_samples, n_features) 的二维数组")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("输入 y 必须是与 X 长度相同的一维数组")
        self._X = X
        self._y = y
        return self

    def predict(
        self,
        X: np.ndarray,
        k: int = 5,
        batch_size: Optional[int] = None,
        feature_block_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        为一批查询样本预测标签。

        参数
        ----------
        X : np.ndarray, 形状为 (n_queries, n_features)
            需要分类的查询样本。其特征维度必须与训练数据匹配。
        k : int, default=5
            使用的邻居数量。必须在 [1, n_train] 范围内。
        batch_size : Optional[int]
            当 p!=2 时，为限制内存使用，查询将按此大小分批处理。
            如果为 None，将使用自动启发式规则。
        feature_block_size : Optional[int]
            当 p!=2 时，为限制内存使用，距离将按此大小的特征块累加计算。
            如果为 None，将使用自动启发式规则。

        返回
        -------
        preds : np.ndarray, 形状为 (n_queries,)
            每个查询样本的预测标签。
        """
        if self._X is None or self._y is None:
            raise RuntimeError("模型尚未拟合，请先调用 fit 方法")
        if k <= 0 or k > self._X.shape[0]:
            raise ValueError("参数 k 必须在 [1, n_train] 范围内")

        Xq = np.asarray(X)
        if Xq.ndim != 2 or Xq.shape[1] != self._X.shape[1]:
            raise ValueError("输入 X 必须是二维的，且特征维度需与训练数据匹配")

        num_queries = Xq.shape[0]
        num_train = self._X.shape[0]
        num_features = self._X.shape[1]

        if self.p == 2:
            # 使用点积技巧一次性计算欧氏距离 (M,N)
            distances = self._pairwise_minkowski(Xq, self._X, self.p)
            # 使用 argpartition 找到前 k 小的距离的索引
            neighbor_indices = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
            topk_labels = self._y[neighbor_indices]
            row_indices = np.arange(num_queries)[:, None]
            topk_distances = distances[row_indices, neighbor_indices]
            preds = np.array(
                [self._vote_with_tiebreak(topk_labels[i], topk_distances[i]) for i in range(num_queries)],
                dtype=topk_labels.dtype,
            )
            return preds

        if batch_size is None:
            batch_size = 32 if num_train >= 2000 else 128
        if feature_block_size is None:
            feature_block_size = 128 if num_features >= 1024 else min(128, num_features)

        preds_out = np.empty((num_queries,), dtype=self._y.dtype)

        for start in range(0, num_queries, batch_size):
            end = min(start + batch_size, num_queries)
            A = Xq[start:end]  # 当前批次的查询样本 (b, d)
            # 在特征块上累加计算当前批次的距离
            dmat = np.zeros((A.shape[0], num_train), dtype=np.float32)
            for f0 in range(0, num_features, feature_block_size):
                f1 = min(f0 + feature_block_size, num_features)
                Ab = A[:, f0:f1]
                Bb = self._X[:, f0:f1]
                # 使用广播机制计算块距离并累加
                diff_block = np.abs(Ab[:, None, :] - Bb[None, :, :])
                if self.p == 1:
                    dmat += np.sum(diff_block, axis=2)
                else:
                    dmat += np.sum(diff_block ** self.p, axis=2)
            if self.p != 1:
                # 完成闵可夫斯基距离计算：在求和后取 1/p 次方
                np.power(dmat, 1.0 / self.p, out=dmat)

            # 为当前批次选择 top-k 邻居
            neigh_idx = np.argpartition(dmat, kth=k - 1, axis=1)[:, :k]
            batch_labels = self._y[neigh_idx]
            row_idx = np.arange(A.shape[0])[:, None]
            batch_dists = dmat[row_idx, neigh_idx]
            batch_preds = np.array(
                [self._vote_with_tiebreak(batch_labels[i], batch_dists[i]) for i in range(A.shape[0])],
                dtype=batch_labels.dtype,
            )
            preds_out[start:end] = batch_preds

        return preds_out

    @staticmethod
    def _pairwise_minkowski(A: np.ndarray, B: np.ndarray, p: float) -> np.ndarray:
        """
        计算两个矩阵之间的成对闵可夫斯基距离。

        结果矩阵的 (i, j) 位置对应 A[i] 和 B[j] 之间的距离。
        对于 p=2，使用一个优化的公式 ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        来避免分配 (m,n,d) 形状的中间数组。

        参数
        ----------
        A : np.ndarray, 形状为 (m, d)
            包含 m 个查询点的矩阵。
        B : np.ndarray, 形状为 (n, d)
            包含 n 个参考点的矩阵。
        p : float
            闵可夫斯基距离的阶数。

        返回
        -------
        dists : np.ndarray, 形状为 (m, n)
            A 和 B 之间的成对距离。
        """
        if p == 2:
            # 使用 (a-b)^2 的展开式，通过 BLAS 高效计算距离的平方
            A2 = np.sum(A * A, axis=1, keepdims=True)  # (m,1)
            B2 = np.sum(B * B, axis=1, keepdims=True).T  # (1,n)
            G = A @ B.T  # (m,n)
            sq = A2 + B2 - 2.0 * G
            np.maximum(sq, 0.0, out=sq) # 确保距离的平方不为负
            return np.sqrt(sq, dtype=A.dtype)

        # 通用的 p 值路径 (未优化)
        diff = np.abs(A[:, None, :] - B[None, :, :])
        if p == 1:
            d = np.sum(diff, axis=2)
        else:
            d = np.sum(diff ** p, axis=2) ** (1.0 / p)
        return d

    @staticmethod
    def _vote_with_tiebreak(labels: np.ndarray, dists: np.ndarray):
        """
        带有确定性平局打破规则的多数投票。

        策略:
        1) 选择票数最多的标签。
        2) 如果票数并列，选择对应邻居距离总和更小的那个标签。
        3) 如果距离总和仍然并列，选择数值上更小的那个标签。
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        max_count = np.max(counts)
        candidates = unique_labels[counts == max_count]
        if candidates.size == 1:
            return candidates[0]

        best_label = None
        best_sum = None
        for lab in candidates:
            mask = labels == lab
            s = float(np.sum(dists[mask]))
            if best_sum is None or s < best_sum or (s == best_sum and lab < best_label):
                best_sum = s
                best_label = lab
        return best_label


def train_test_split_indices(
    n_samples: int,
    test_ratio: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将整数索引 [0, n_samples) 拆分为训练和测试子集。

    参数
    ----------
    n_samples : int
        要拆分的样本总数。
    test_ratio : float, default=0.3
        分配给测试集的样本比例 (将四舍五入为整数)。
    rng : np.random.Generator, optional
        用于可复现性的随机数生成器。如果为 None，则使用默认生成器。

    返回
    -------
    train_idx : np.ndarray
        分配给训练集的索引。
    test_idx : np.ndarray
        分配给测试集的索引。
    """
    if rng is None:
        rng = np.random.default_rng()
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_test = int(round(n_samples * test_ratio))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return train_idx, test_idx