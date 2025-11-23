# cnn/layers.py
import numpy as np


# ======== 基础工具：im2col / col2im，用于加速卷积和池化 ========

def im2col_indices(x, field_height, field_width, padding=0, stride=1):
    """
    将 4D 输入张量 (N, C, H, W) 展开成 2D 矩阵，用于卷积 / 池化的矩阵乘法实现。

    返回:
        cols: 形状为 (C * field_height * field_width, N * H_out * W_out)
    """
    N, C, H, W = x.shape
    assert (H + 2 * padding - field_height) % stride == 0, "Invalid height"
    assert (W + 2 * padding - field_width) % stride == 0, "Invalid width"

    H_out = (H + 2 * padding - field_height) // stride + 1
    W_out = (W + 2 * padding - field_width) // stride + 1

    # pad 输入
    x_padded = np.pad(
        x,
        ((0, 0), (0, 0), (padding, padding), (padding, padding)),
        mode="constant"
    )

    # 计算每个感受野的索引
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(H_out), W_out)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(W_out), H_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    cols = x_padded[:, k, i, j]  # (N, C*HH*WW, H_out*W_out)
    cols = cols.transpose(1, 2, 0).reshape(C * field_height * field_width, -1)
    return cols


def col2im_indices(cols, x_shape, field_height, field_width, padding=0, stride=1):
    """
    将 im2col 得到的 2D 矩阵还原回 4D 张量 (N, C, H, W)。
    """
    N, C, H, W = x_shape
    H_out = (H + 2 * padding - field_height) // stride + 1
    W_out = (W + 2 * padding - field_width) // stride + 1

    x_padded = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=cols.dtype)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(H_out), W_out)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(W_out), H_out)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    cols_reshaped = cols.reshape(C * field_height * field_width, H_out * W_out, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)

    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


# ======== 卷积层 ========

def conv_forward_fast(x, w, b, conv_param):
    """
    快速卷积前向传播，使用 im2col + 矩阵乘法实现。

    输入:
        - x: 输入数据, 形状 (N, C, H, W)
        - w: 卷积核权重, 形状 (F, C, HH, WW)
        - b: 偏置, 形状 (F,)
        - conv_param: dict, 包含:
            - 'stride': 步幅
            - 'pad': 填充像素数

    返回:
        - out: 输出特征图, 形状 (N, F, H_out, W_out)
        - cache: 反向传播用的中间变量
    """
    stride = conv_param.get("stride", 1)
    pad = conv_param.get("pad", 0)

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    H_out = (H + 2 * pad - HH) // stride + 1
    W_out = (W + 2 * pad - WW) // stride + 1

    x_cols = im2col_indices(x, HH, WW, padding=pad, stride=stride)
    w_row = w.reshape(F, -1)  # (F, C*HH*WW)

    out = w_row @ x_cols + b.reshape(-1, 1)  # (F, N*H_out*W_out)
    out = out.reshape(F, H_out, W_out, N).transpose(3, 0, 1, 2)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache


def conv_backward_fast(dout, cache):
    """
    快速卷积反向传播，对应 conv_forward_fast。
    """
    x, w, b, conv_param, x_cols = cache
    stride = conv_param.get("stride", 1)
    pad = conv_param.get("pad", 0)

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    H_out = (H + 2 * pad - HH) // stride + 1
    W_out = (W + 2 * pad - WW) // stride + 1

    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(F, -1)

    db = np.sum(dout_reshaped, axis=1)
    dw = dout_reshaped @ x_cols.T
    dw = dw.reshape(w.shape)

    w_row = w.reshape(F, -1)
    dx_cols = w_row.T @ dout_reshaped
    dx = col2im_indices(dx_cols, x.shape, HH, WW, padding=pad, stride=stride)

    return dx, dw, db


# ======== ReLU ========

def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    x = cache
    dx = dout * (x > 0)
    return dx


# ======== 最大池化层 ========

def max_pool_forward_fast(x, pool_param):
    """
    最大池化前向传播，使用 im2col 实现。

    pool_param:
        - 'pool_height'
        - 'pool_width'
        - 'stride'
    """
    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    H_out = (H - pool_height) // stride + 1
    W_out = (W - pool_width) // stride + 1

    x_reshaped = x.reshape(N * C, 1, H, W)
    x_cols = im2col_indices(
        x_reshaped, pool_height, pool_width, padding=0, stride=stride
    )  # (pool_height*pool_width, N*C*H_out*W_out)

    # 每列里取最大值
    max_idx = np.argmax(x_cols, axis=0)
    out = x_cols[max_idx, np.arange(max_idx.size)]
    out = out.reshape(H_out, W_out, N, C).transpose(2, 3, 0, 1)

    cache = (x, pool_param, x_cols, max_idx)
    return out, cache


def max_pool_backward_fast(dout, cache):
    x, pool_param, x_cols, max_idx = cache
    N, C, H, W = x.shape
    pool_height = pool_param["pool_height"]
    pool_width = pool_param["pool_width"]
    stride = pool_param["stride"]

    H_out = (H - pool_height) // stride + 1
    W_out = (W - pool_width) // stride + 1

    dout_flat = dout.transpose(2, 3, 0, 1).ravel()

    dx_cols = np.zeros_like(x_cols)
    dx_cols[max_idx, np.arange(max_idx.size)] = dout_flat

    dx_reshaped = col2im_indices(
        dx_cols, (N * C, 1, H, W), pool_height, pool_width, padding=0, stride=stride
    )
    dx = dx_reshaped.reshape(N, C, H, W)
    return dx


# ======== 仿射层（全连接） ========

def affine_forward(x, w, b):
    """
    仿射前向: out = xW + b

    x: (N, d1, d2, ...) 或 (N, D)
    w: (D, M)
    b: (M,)
    """
    N = x.shape[0]
    x_row = x.reshape(N, -1)
    out = x_row @ w + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    N = x.shape[0]
    x_row = x.reshape(N, -1)

    dw = x_row.T @ dout
    db = np.sum(dout, axis=0)
    dx_row = dout @ w.T
    dx = dx_row.reshape(x.shape)

    return dx, dw, db


# ======== Softmax Loss ========

def softmax_loss(x, y):
    """
    计算 softmax 交叉熵损失和梯度。

    x: (N, C) scores
    y: (N,) labels
    """
    shifted = x - np.max(x, axis=1, keepdims=True)  # 防止溢出
    exp_scores = np.exp(shifted)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N

    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
