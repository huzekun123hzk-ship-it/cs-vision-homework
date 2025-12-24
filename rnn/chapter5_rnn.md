## 第五章：循环神经网络 (Recurrent Neural Network, RNN)

## 第一部分：RNN理论基础

&emsp;&emsp;在第四章中，我们看到 **CNN** 之所以强大，是因为它通过“**局部感受野 + 参数共享**”保留并利用了图像的空间结构。  
&emsp;&emsp;但当数据变成 **序列（sequence）** 时，问题的核心不再是“空间结构”，而是“**时间结构 / 顺序依赖**”。例如：文本是词序列 $(w_1, w_2, \dots, w_T)$；语音是随时间变化的帧序列；视频是帧序列（动作需要上下文）；时间序列数据（传感器/金融）也往往满足“当前与过去相关”。

&emsp;&emsp;本章要介绍的 **循环神经网络（Recurrent Neural Network, RNN）**，就是为了解决“**变长输入 + 时序依赖**”而提出的经典结构。


### 1.1 序列建模的根本挑战：变长输入 + 上下文依赖

&emsp;&emsp;序列任务通常同时有两类“天然难点”。

&emsp;&emsp;第一类是 **长度不固定（Variable Length）**。输入是一段序列：
$$
X = (x_1, x_2, \dots, x_T),
$$
其中 $T$ 会因样本而异。我们希望模型无需“重新设计输入层”，就能直接处理不同长度的序列。

&emsp;&emsp;第二类是 **时序依赖（Temporal Dependency）**。当前时刻的含义往往依赖历史上下文：比如 “good” 与 “not good” 的区别来自前面的 “not”；视频里单帧可能看不出动作，但一段连续帧能确定“起跳→腾空→落地”。

&emsp;&emsp;如果我们像 FCN 那样把序列拼接成一个超长向量再做分类，会立刻遇到两个问题：一方面 $T$ 一变，输入维度就变；另一方面每个位置都学一套参数会让参数量随 $T$ 增长，泛化也会变差。  
&emsp;&emsp;RNN 的思路非常直接：**用一套共享参数，沿时间步重复使用；再用一个“隐藏状态”携带历史信息。**

<p align="center"><b>图 1：序列任务的“上下文依赖”示意（后续输出依赖历史）</b></p>
<p align="center">
  <img src="./assets/5-1.png" alt="序列上下文依赖示意图" width="85%">
</p>


### 1.2 RNN 的核心思想：隐藏状态（Hidden State）作为“记忆”

&emsp;&emsp;RNN 在每个时间步 $t$ 读入输入 $x_t$，同时维护一个隐藏状态 $h_t$（可以理解为“记忆向量”）。最经典的 Vanilla RNN 更新公式为：
$$
h_t = \phi\big(W_{xh}x_t + W_{hh}h_{t-1} + b_h\big)
$$

&emsp;&emsp;如果任务需要每步输出（例如逐帧分类、序列标注），我们再接一个输出层：
$$
o_t = W_{hy}h_t + b_y, \qquad y_t = g(o_t)
$$

&emsp;&emsp;这里 $x_t$ 是第 $t$ 步输入（词向量、帧特征、传感器读数等），$h_t$ 是第 $t$ 步隐藏状态（携带历史信息），$y_t$ 是第 $t$ 步输出（是否需要取决于任务）。$\phi(\cdot)$ 是非线性（常用 $\tanh$；也可用 ReLU），$g(\cdot)$ 是输出激活（分类常用 softmax；回归常用恒等映射）。

&emsp;&emsp;你可以把 RNN 的设计浓缩成两句话：**状态传递**（$h_{t-1}\rightarrow h_t$）让信息沿时间流动；**参数共享**（同一套 $W_{xh},W_{hh},b_h$ 在所有时间步复用）让模型天然支持变长序列。

<p align="center"><b>图 2：RNN 单步结构（x_t 与 h_{t-1} 共同决定 h_t）</b></p>
<p align="center">
  <img src="./assets/5-2.png" alt="RNN cell：x_t 与 h_{t-1} 输入到同一单元，输出 h_t（与可选的 y_t）" width="72%">
</p>


### 1.3 时间展开（Unroll）：把“循环”看成一个深网络

&emsp;&emsp;虽然名字叫“循环网络”，但在训练与推导时，我们通常把它在时间轴上展开（unroll）成一个深网络：
$$
(x_1 \rightarrow h_1) \rightarrow (x_2 \rightarrow h_2) \rightarrow \cdots \rightarrow (x_T \rightarrow h_T)
$$

&emsp;&emsp;展开后的直观意义是：网络的“深度”会随序列长度 $T$ 增长，因此更容易出现长链梯度问题；但与此同时，每一层结构相同并共享参数，因此**参数量不随 $T$ 增长**，也更容易在不同位置复用同一种规律。

<p align="center"><b>图 3：RNN 在时间维的展开（Unroll）示意图</b></p>
<p align="center">
  <img src="./assets/5-3.png" alt="RNN 时间展开：每个时间步共享同一套参数" width="88%">
</p>


### 1.4 维度与参数形式：把符号写清楚（工程最常用的一页）

&emsp;&emsp;为避免实现时维度混乱，我们明确如下设定：输入维度为 $D$，隐藏维度为 $H$，输出维度为 $K$（例如 $K$ 类分类）：
$$
x_t \in \mathbb{R}^{D}, \qquad h_t \in \mathbb{R}^{H}, \qquad y_t \in \mathbb{R}^{K}.
$$

&emsp;&emsp;对应参数矩阵形状为：
$$
W_{xh} \in \mathbb{R}^{H \times D}, \quad
W_{hh} \in \mathbb{R}^{H \times H}, \quad
b_h \in \mathbb{R}^{H},
$$
$$
W_{hy} \in \mathbb{R}^{K \times H}, \quad
b_y \in \mathbb{R}^{K}.
$$

&emsp;&emsp;此外，RNN 需要初始状态 $h_0$。最常见做法是 **零初始化**：$h_0=\mathbf{0}$；有时也会把 $h_0$ 作为可学习参数，但在入门实现中并不是必须。


### 1.5 输出模式：many-to-one / many-to-many / seq2seq

&emsp;&emsp;RNN 的“输入输出对齐方式”决定了它如何使用 $h_t$ 与损失函数。最常见三种形态如下。

&emsp;&emsp;最常见的是 **many-to-one：整段序列 → 一个输出**，例如句子情感分类、视频动作分类。常用做法是用最后状态 $h_T$ 做分类：
$$
y = g(W_{hy}h_T + b_y).
$$

&emsp;&emsp;第二类是 **many-to-many（对齐）：每步输入 → 每步输出**，例如序列标注（词性标注）、逐帧分类。此时每个时间步都会输出 $y_t$。

&emsp;&emsp;第三类是 **seq2seq（不对齐）：输入序列 → 输出序列（长度可不同）**，例如机器翻译、图像描述（常见 Encoder-Decoder）。这一类通常会在更后面章节或扩展部分展开。

<p align="center"><b>图 4：RNN 常见输入输出形态（one-to-one / one-to-many / many-to-one / many-to-many）</b></p>
<p align="center">
  <img src="./assets/5-4.png" alt="RNN 常见任务形态示意图" width="92%">
</p>


### 1.6 训练方法：时间反向传播（BPTT）

&emsp;&emsp;RNN 的训练依然使用梯度下降，但反向传播需要沿时间展开回传，这称为 **BPTT（Backpropagation Through Time）时间反向传播**。

&emsp;&emsp;以 many-to-many 为例，如果每一步都有损失 $\mathcal{L}_t$，总损失通常是时间求和（或求均值）：
$$
\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t.
$$

&emsp;&emsp;BPTT 的关键点在于：反向传播从 $t=T$ 回到 $t=1$；由于参数共享，$W_{hh}$、$W_{xh}$ 的梯度会累积来自每个时间步的贡献。  
&emsp;&emsp;在工程实现中，这意味着 forward 必须缓存每步的中间量（如 $h_{t-1}$、线性项、激活结果），而 backward 则必须按 $t=T,T-1,\dots,1$ 的顺序回传并累积梯度。

<p align="center"><b>图 5：BPTT 沿时间链条从后往前回传梯度</b></p>
<p align="center">
  <img src="./assets/5-5.png" alt="BPTT：梯度从后往前回传，跨越多个时间步" width="86%">
</p>


### 1.7 经典难点：梯度消失与梯度爆炸

&emsp;&emsp;这是 RNN 最著名的问题。在 BPTT 过程中，梯度需要沿时间链条经过长达 $T$ 次的连乘。如果连乘项整体“小于 1”，梯度会迅速衰减到零（消失），导致模型很难学习长程依赖；如果连乘项整体“大于 1”，梯度会呈指数级增长（爆炸），导致训练不稳定甚至直接崩溃。

&emsp;&emsp;工程上的对策通常包括：**梯度裁剪（Gradient Clipping）**（通过限制梯度范数来防止爆炸）以及 **截断 BPTT（Truncated BPTT）**。

&emsp;&emsp;截断 BPTT 的思路是：把长序列切成长度为 $k$ 的片段分段训练。**隐藏状态在片段之间连续传递**，但**梯度在边界处被截断（stop gradient / detach）**，也就是把上一段末尾的隐藏状态当作常数，避免梯度跨段无限回传。

&emsp;&emsp;例如完整序列为：
$$
[t_0, t_1, t_2, \dots, t_{100}],
$$
若设截断窗口 $k=10$，则训练过程可以理解为反复执行：
$$
[t_{0}\!\rightarrow\!t_{10}],\ [t_{11}\!\rightarrow\!t_{20}],\ \dots,\ [t_{91}\!\rightarrow\!t_{100}],
$$
每段内部做一次 BPTT 更新参数，并把该段末尾隐藏状态（如 $h_{10},h_{20},\dots$）传给下一段作为初始状态，但不把梯度再传回上一段。

<p align="center"><b>图 6：梯度随时间回传可能指数衰减（消失）或指数增长（爆炸）</b></p>
<p align="center">
  <img src="./assets/5-6.png" alt="vanishing/exploding gradients 示意图" width="86%">
</p>


### 1.8 RNN 与 CNN 的关系：从“空间建模”到“时序建模”

&emsp;&emsp;在视觉任务中，RNN 往往并不是“直接吃像素”，而是与 CNN 配合使用：先用 CNN 提取每帧特征 $x_t$，再由 RNN 在时间维度整合上下文：
$$
x_t = \text{CNN}(\text{frame}_t), \qquad h_t = \text{RNN}(x_t, h_{t-1}).
$$

&emsp;&emsp;同理，在图像描述（captioning）中，CNN 先把图像编码成一个向量表示，再用 RNN/Decoder 逐词生成句子。你可以把两者分工理解为：CNN 擅长“**空间结构**”（纹理/形状/局部到整体），RNN 擅长“**顺序结构**”（上下文/先后关系）。

<p align="center"><b>图 7：CNN + RNN 组合（逐帧 CNN 特征 → RNN 时间整合）</b></p>
<p align="center">
  <img src="./assets/5-7.png" alt="CNN 提取每帧特征，RNN 融合时间上下文" width="92%">
</p>


### 1.9 小结：RNN 第一部分的核心结论

&emsp;&emsp;本节我们建立了 RNN 的基础理论框架：隐藏状态 $h_t$ 是“记忆”，负责把历史信息压缩并传递到当前；时间展开（Unroll）+ 参数共享让 RNN 能处理变长序列且参数量不随 $T$ 增长；BPTT 是 RNN 的反向传播方式，本质是沿时间链条回传并在共享参数处累积梯度；而梯度消失/爆炸是长链回传带来的经典困难，工程上常用梯度裁剪与截断 BPTT，结构上通常用 LSTM/GRU 改善长程依赖学习。  
&emsp;&emsp;在接下来的第二部分中，我们将从工程角度“手搓”一个最小可训练的 RNN：实现单步 forward/backward、序列版 BPTT、以及梯度裁剪，并通过一个简单序列任务验证它确实能学到时序依赖。





## 第二部分：代码实现详解（

&emsp;&emsp;本部分将深入 `rnn/` 目录下的四份核心代码，展示如何用纯 NumPy 完成 CIFAR-10 的 RNN 分类实验：

```text
rnn/
  ├── experiments/
  │    ├── cifar10_experiment.py      # 主实验脚本：训练 + 测试 + 保存结果
  │    ├── cifar10_utils.py           # CIFAR-10 加载与预处理工具
  │    └── overfit_debug.py           # 小样本过拟合调试脚本（sanity check）
  └── rnn_classifier.py               # Vanilla RNN 分类器（forward + BPTT + clip）
````


### 2.1 `cifar10_utils.py`：加载与预处理（从 batch 文件到标准化向量）

  `cifar10_utils.py` 的目标是：从 `cifar-10-batches-py/` 目录中读取 CIFAR-10 的 python 版 batch 文件，并返回 train/val/test 的图像张量与标签；随后提供一个标准预处理函数，把图像转成可直接喂给模型的数值矩阵。

#### 2.1.1 `load_cifar10()`：读取并划分 Train / Val / Test

&emsp;&emsp;CIFAR-10 的训练集被拆成 `data_batch_1 ~ data_batch_5`，测试集为 `test_batch`。代码用 `pickle` 读出每个 batch 的 `data` 与 `labels`，再做拼接：

* 训练全集：`X_train_all` 形状为 `(50000, 3072)`
* 测试全集：`X_test_all` 形状为 `(10000, 3072)`

随后按参数切片：

* 训练集：前 `train_samples` 张
* 验证集：紧接其后的 `val_samples` 张
* 测试集：测试集前 `test_samples` 张

#### 2.1.2 `reshape_img()`：把 `(N, 3072)` 还原成 `(N, 32, 32, 3)`

&emsp;&emsp;CIFAR-10 存储的 `data` 是扁平向量，但其通道顺序是 `(3, 32, 32)`。我们需要 reshape 并 transpose：

```python
X = X.reshape(-1, 3, 32, 32)     # (N, 3, 32, 32)
X = X.transpose(0, 2, 3, 1)      # (N, 32, 32, 3)
```

这样后续既能可视化，也能按“行/列”把图像序列化。

#### 2.1.3 `preprocess_data()`：Flatten + 中心化 + 归一化 + Bias Trick

&emsp;&emsp;预处理函数的输出是 `(N, 3073)`，其中最后一维为 bias=1（Bias Trick）：

1. Flatten：`(N, 32, 32, 3) -> (N, 3072)`
2. Mean Subtraction：减去训练集均值（对 train/val/test 统一使用训练均值）
3. Std Normalization：除以训练集标准差（加 `1e-7` 防止除零）
4. Bias Trick：拼接一列全 1 得到 `(N, 3073)`

> 注意：RNN 自身带 `b_h` 与 `b_q`，因此在进入 RNN 前我们会把 bias 这一维去掉（见 2.2 节）。


### 2.2 `cifar10_experiment.py`：关键步骤——把图像重塑为序列 `(N, T, D)`

&emsp;&emsp;RNN 的输入不是二维图像，而是序列张量 (X\in\mathbb{R}^{N\times T\times D})。本章采用 **row-by-row** 序列化策略：

* 一张图像：`32 × 32 × 3`
* 时间步：`T = 32`（每行一个时间步）
* 每步特征：`D = 32 × 3 = 96`

因此重塑规则是：

$$ (N, 3072) \longrightarrow (N, 32, 96) $$

对应代码中的关键函数 `reshape_for_rnn()`：

```python
# 1) 去掉 preprocess_data 添加的 bias 维度
X_nobias = X_flat[:, :-1]         # (N, 3072)

# 2) 直接 reshape 成 (N, 32, 96)
X_seq = X_nobias.reshape(N, 32, 96)
```

#### 2.2.1 为什么这种 reshape 合理？

&emsp;&emsp;这等价于：把图像当作“从上到下读的一段序列”。RNN 的隐藏状态就像“读完上一行后留下的记忆”，它可以把上方信息带到下方，从而形成一种弱形式的空间建模。

#### 2.2.2 形状检查（防止 silent bug）

&emsp;&emsp;在实验脚本中，我们打印了关键形状：

* Train: `(49000, 32, 96)`
* Val: `(1000, 32, 96)`
* Test: `(1000, 32, 96)`

这一步能快速确认序列化是否正确（尤其是 `T=32` 与 `D=96` 是否对齐）。



### 2.3 `rnn_classifier.py`：Vanilla RNN 前向 + BPTT 反向（核心）

  `RNNClassifier` 完整实现了一个 many-to-one 的 Vanilla RNN 分类器。核心由四部分组成：

1. 参数初始化（特别是 (W_{hh}) 的初始化）
2. `_forward()`：时间展开前向传播
3. `_compute_loss_and_gradient()`：softmax loss + BPTT + clipping
4. `train()`：mini-batch SGD + early stopping + 学习率衰减

#### 2.3.1 参数初始化：为什么要用 Xavier / Glorot

  RNN 对初始化非常敏感。若初始权重太大，tanh 容易饱和导致梯度消失；若太小则学习慢。你采用了 Xavier 风格初始化（用输入维度缩放方差）：

* (W_{xh}\sim \mathcal{N}(0,1)/\sqrt{D})
* (W_{hh}\sim \mathcal{N}(0,1)/\sqrt{H})
* (W_{hq}\sim \mathcal{N}(0,1)/\sqrt{H})

这能在训练初期保持激活的尺度稳定，让优化更顺畅。


#### 2.3.2 `_forward()`：时间展开与隐藏状态缓存

&emsp;&emsp;对输入 (X\in\mathbb{R}^{N\times T\times D})，前向传播按时间循环，并缓存每一个时刻的隐藏状态：

```python
h_states = np.zeros((N, T+1, H))  # 额外存 h0=0

for t in range(T):
    x_t = X[:, t, :]                 # (N, D)
    h_prev = h_states[:, t, :]       # (N, H)
    linear = x_t @ W_xh + h_prev @ W_hh + b_h
    h_next = np.tanh(linear)
    h_states[:, t+1, :] = h_next

h_last = h_states[:, T, :]          # (N, H)
scores = h_last @ W_hq + b_q         # (N, K)
```

&emsp;&emsp;这里的关键点是 **h_states 必须缓存全部时间步**，因为 BPTT 反向传播要用到每个 (h_t) 的值。


#### 2.3.3 Softmax 交叉熵损失 + L2 正则

&emsp;&emsp;输出 scores 经过 softmax 得到概率：

$$ p_{i,j} = \frac{e^{s_{i,j}}}{\sum_k e^{s_{i,k}}} $$

&emsp;&emsp;损失：

$$ L = \frac{1}{N} \sum_i -\log p_{i,y_i} + \frac{\lambda}{2} \left( \|W_{xh}\|^2 + \|W_{hh}\|^2 + \|W_{hq}\|^2 \right) $$

实现中使用 `scores - max(scores)` 做数值稳定。

---

#### 2.3.4 BPTT：从输出层一路回传到所有时间步

&emsp;&emsp;输出层梯度是经典 softmax 结论：

```python
dscores = probs
dscores[range(N), y] -= 1
dscores /= N

dW_hq = h_last.T @ dscores
db_q  = np.sum(dscores, axis=0)

dh = dscores @ W_hq.T   # 传播回最后时刻隐藏层
```

然后从 (t=T-1) 到 (0) 反向循环（BPTT）：
$
* tanh 导数：$((1-h_t^2))$
* 参数共享，因此每个时间步梯度需要累加

```python
for t in reversed(range(T)):
    h_curr = h_states[:, t+1, :]
    h_prev = h_states[:, t, :]
    x_t    = X[:, t, :]

    dtanh = (1 - h_curr**2) * dh
    dW_xh += x_t.T @ dtanh
    dW_hh += h_prev.T @ dtanh
    db_h  += np.sum(dtanh, axis=0)

    dh = dtanh @ W_hh.T
```

最后加上正则项梯度：

```python
dW_xh += reg * W_xh
dW_hh += reg * W_hh
dW_hq += reg * W_hq
```


#### 2.3.5 梯度裁剪（Gradient Clipping）

&emsp;&emsp;RNN 很容易出现梯度爆炸。为保证训练稳定，你对所有梯度做了逐元素裁剪：

```python
np.clip(grad, -5, 5, out=grad)
```

这能有效避免某次更新把参数“炸飞”，是 Vanilla RNN 能跑起来的关键工程技巧之一。



### 2.4 `train()`：mini-batch SGD + early stopping + 学习率衰减

&emsp;&emsp;训练循环做了四件事：

1. 每个 epoch 打乱训练集
2. 逐 batch 计算 loss 与 grads
3. SGD 更新参数
4. 每隔 `print_every` 评估 train/val acc，并用 early stopping 保存最佳参数

early stopping 的逻辑是：

* 若 val acc 变好：保存 best 参数，计数清零
* 否则累计无提升轮数
* 无提升达到 patience：停止训练并恢复 best 参数

学习率衰减采用“按周期乘 0.95”的策略，避免后期震荡。


### 2.5 `overfit_debug.py`：为什么一定要做“小数据过拟合”验证

&emsp;&emsp;手写网络最重要的 sanity check 是：在一个小数据集上能否过拟合到很高训练精度。如果连小数据都学不上去，通常说明 forward/backward 或梯度实现存在错误。

&emsp;&emsp;你的 debug 结果显示训练集准确率可以接近 1（如 0.99+），说明实现大概率正确；而 val/test 明显低，则更多是泛化问题或序列化策略带来的表达能力上限，而不是代码 bug。


## 第三部分：实验结果与分析（图表 + 结论）

&emsp;&emsp;本部分基于 `cifar10_experiment.py` 的完整实验输出，展示训练曲线、预测可视化与最终测试精度，并与第三章（FCN）做横向对比。



### 3.1 实验设置

#### 3.1.1 数据划分与预处理

* CIFAR-10 数据集
* 划分：

  * 训练集：49000
  * 验证集：1000
  * 测试集：1000
* 预处理：flatten → 训练集均值中心化 → 标准差归一化（bias trick 后在进入 RNN 前去掉）

#### 3.1.2 图像序列化方式（Row-by-Row）

* 序列长度：(T=32)
* 每步维度：(D=96)
* 输入张量：$(X\in\mathbb{R}^{N\times 32\times 96})$

#### 3.1.3 模型与超参数（与代码一致）

* input_dim = 96
* hidden_dim = 256
* output_dim = 10
* reg_strength = 1e-4
* learning_rate = 2e-3
* batch_size = 128
* num_epochs = 200（early stopping）
* patience = 40
* lr_decay_rate = 0.95（周期性衰减）


### 3.2 训练过程：Early Stopping 与收敛表现

&emsp;&emsp;训练日志显示验证集准确率在达到峰值后进入平台期，最终触发 early stopping：

```text
Early stopping! No improvement for 40 epochs. Best ValAcc=0.4800
Training finished. Restored best model. Best ValAcc=0.4800
```

&emsp;&emsp;这说明：模型确实学到了有效模式，但在验证集上很快出现“收益递减”，继续训练无法带来明显提升，因此 early stopping 能有效防止无意义的长时间训练与潜在过拟合。


### 3.3 最终测试集表现（核心结果）

&emsp;&emsp;使用验证集表现最佳的参数在测试集评估，得到：


$$ \text{Test Accuracy} = \mathbf{49.30\%} $$

对应终端输出：

```text
🏆 Test Accuracy: 0.4930 (49.30%)
```


### 3.4 训练曲线：Loss 与 Accuracy

<p align="center"><b>图 1：RNN 训练曲线（Loss / Train Acc / Val Acc）</b></p>
<p align="center">
  <img src="./experiments/cifar10_results/training_curves.png"
       alt="RNN Training Curves" width="92%">
</p>

&emsp;&emsp;从曲线可以观察到：

1. **Loss 持续下降**：说明优化过程有效，梯度实现正确且训练稳定。
2. **Train Acc 明显高于 Val Acc**：存在一定过拟合趋势，但 early stopping 在验证集不再提升时及时截断训练。
3. **Val Acc 平台期明显**：提示当前结构（Vanilla RNN + row-by-row）存在性能上限，继续堆 epoch 不会显著提升泛化。



### 3.5 预测可视化：正确与错误样本

<p align="center"><b>图 2：预测可视化（绿色=正确，红色=错误）</b></p>
<p align="center">
  <img src="./experiments/cifar10_results/prediction_viz.png"
       alt="Prediction Visualization" width="92%">
</p>

&emsp;&emsp;从预测样例可以看到：车辆类（如 ship、truck）往往更容易识别；而动物类（cat/dog、deer/horse）更容易混淆。这一现象与第三章 FCN 的混淆矩阵结论一致：**动物类别姿态变化大、背景复杂且类间纹理相似**，对简单模型更不友好。



### 3.6 与第三章 FCN 的对比：RNN 能学到什么？学不到什么？

&emsp;&emsp;我们把第三章两层 FCN 与本章 RNN 做一个直接对比（都使用原始像素，不引入卷积归纳偏置）：

| 模型              | 输入表示                  | 测试准确率      |
| --------------- | --------------------- | ---------- |
| 两层 FCN（第三章）     | flatten 像素 (3072)     | **49.84%** |
| Vanilla RNN（本章） | row-by-row 序列 (32×96) | **49.30%** |

&emsp;&emsp;可以看到两者非常接近：RNN 并没有显著超过 FCN。原因也很直观：

1. **RNN 的归纳偏置更偏“序列”，不适合二维局部结构**

   * FCN 完全忽略空间结构；
   * RNN 虽然引入了“行方向的记忆”，但仍缺少 CNN 那种“局部卷积核 + 平移共享”的强视觉归纳偏置。

2. **row-by-row 序列化会弱化二维邻域关系**

   * 同一行内的像素是同时输入的一大坨 96 维向量；
   * 图像的“局部 3×3 邻域”在这种表示中没有被显式强调；
   * RNN 更擅长建模跨时间步（跨行）的依赖，而不是二维 patch 的局部组合。

3. **Vanilla RNN 容易遇到梯度衰减**

   * 虽然 (T=32) 不算长，但 tanh + 循环权重仍可能造成有效梯度逐步变小；
   * 本章用 Xavier 初始化与梯度裁剪缓解爆炸，但“更深层次的长程建模能力”仍有限。

&emsp;&emsp;因此，本章实验更像是在验证一个观点：**“即使不用卷积，只把图像当序列，RNN 也能学到一些模式并达到接近 FCN 的水平；但要显著超过 FCN，CNN 的结构性优势仍然更强。”**



### 3.7 实验总结与下一步

1. **RNN 在 CIFAR-10 上可训练、可收敛**

   * Xavier 初始化 + BPTT + 梯度裁剪保证了训练稳定性；
   * early stopping 能有效控制无效训练与过拟合风险。

2. **最终准确率达到 49.30%**

   * 与第三章两层 FCN（49.84%）非常接近；
   * 说明“序列视角”能学到一部分图像判别信息，但不具备 CNN 的优势。

3. **局限性明确：RNN 缺少适配图像的归纳偏置**

   * 若想进一步提升性能，需要引入更适合视觉任务的结构（CNN、ResNet）；
   * 或者在序列视角下引入更强序列模型（LSTM/GRU、Attention/Transformer），以及更合理的序列化方式（如 patch 序列、双向扫描等）。

  至此，我们完成了从“线性模型 → 两层 FCN → CNN → RNN（序列视角）”的完整探索链条，也为后续更强的结构（如 LSTM/GRU、Attention）打下了工程与理论基础。

