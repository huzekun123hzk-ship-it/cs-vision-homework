# cnn/model.py
import numpy as np

from .layers import (
    conv_forward_fast,
    conv_backward_fast,
    relu_forward,
    relu_backward,
    max_pool_forward_fast,
    max_pool_backward_fast,
    affine_forward,
    affine_backward,
    softmax_loss,
)


class Cifar10SimpleConvNet(object):
    """
    一个简化版的卷积神经网络，结构为：
        conv - relu - 2x2 max pool - affine - relu - affine - softmax

    参数:
        input_dim: 输入图像维度 (C, H, W)
        num_filters: 卷积层滤波器个数
        filter_size: 卷积核尺寸 (filter_size x filter_size)
        hidden_dim: 全连接隐藏层神经元个数
        num_classes: 分类类别数（CIFAR-10 为 10）
        weight_scale: 权重初始化标准差（高斯分布）
        reg: L2 正则化系数
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=5,
        hidden_dim=512,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        self.params = {}
        self.reg = reg

        C, H, W = input_dim

        # 卷积层: W1 (F, C, HH, WW), b1 (F,)
        F = num_filters
        HH = WW = filter_size
        self.params["W1"] = weight_scale * np.random.randn(F, C, HH, WW)
        self.params["b1"] = np.zeros(F)

        # conv 输出尺寸（same padding, stride=1）
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}
        stride = conv_param["stride"]
        pad = conv_param["pad"]
        H_conv = 1 + (H + 2 * pad - HH) // stride
        W_conv = 1 + (W + 2 * pad - WW) // stride

        # 2x2 池化后尺寸
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}
        H_pool = (H_conv - pool_param["pool_height"]) // pool_param["stride"] + 1
        W_pool = (W_conv - pool_param["pool_width"]) // pool_param["stride"] + 1

        # Affine 层输入维度
        affine_input_dim = F * H_pool * W_pool

        # 全连接层: W2, b2
        self.params["W2"] = weight_scale * np.random.randn(
            affine_input_dim, hidden_dim
        )
        self.params["b2"] = np.zeros(hidden_dim)

        # 输出层: W3, b3
        self.params["W3"] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params["b3"] = np.zeros(num_classes)

    def loss(self, X, y=None):
        """
        计算网络的前向和反向传播。

        输入:
            - X: 输入数据, 形状 (N, C, H, W)
            - y: 标签, 形状 (N,). 若为 None，则只做前向传播并返回 scores

        返回:
            - 若 y is None: scores, 形状 (N, num_classes)
            - 若 y 不为 None: (loss, grads dict)
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # ---- 前向传播 ----
        # conv - relu - pool
        conv_param = {"stride": 1, "pad": (W1.shape[2] - 1) // 2}
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        out1, cache_conv = conv_forward_fast(X, W1, b1, conv_param)
        out2, cache_relu1 = relu_forward(out1)
        out3, cache_pool = max_pool_forward_fast(out2, pool_param)

        # affine - relu
        out4, cache_affine1 = affine_forward(out3, W2, b2)
        out5, cache_relu2 = relu_forward(out4)

        # affine
        scores, cache_affine2 = affine_forward(out5, W3, b3)

        if y is None:
            return scores

        # ---- 计算损失 ----
        data_loss, dscores = softmax_loss(scores, y)
        reg_loss = 0.5 * self.reg * (
            np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3)
        )
        loss = data_loss + reg_loss

        # ---- 反向传播 ----
        grads = {}

        # affine2 反向
        dx5, dW3, db3 = affine_backward(dscores, cache_affine2)
        dW3 += self.reg * W3

        # relu2 反向
        dx4 = relu_backward(dx5, cache_relu2)

        # affine1 反向
        dx3, dW2, db2 = affine_backward(dx4, cache_affine1)
        dW2 += self.reg * W2

        # pool 反向
        dx2 = max_pool_backward_fast(dx3, cache_pool)

        # relu1 反向
        dx1 = relu_backward(dx2, cache_relu1)

        # conv 反向
        dx, dW1, db1 = conv_backward_fast(dx1, cache_conv)
        dW1 += self.reg * W1

        grads["W1"], grads["b1"] = dW1, db1
        grads["W2"], grads["b2"] = dW2, db2
        grads["W3"], grads["b3"] = dW3, db3

        return loss, grads
