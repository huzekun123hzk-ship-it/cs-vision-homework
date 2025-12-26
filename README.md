# 《用纯Python手搓经典计算机视觉算法》开源教材项目

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目总览

&emsp;&emsp;欢迎来到《用纯Python手搓经典计算机视觉算法》！这是一本开源教材项目，旨在通过纯`Python`和`NumPy`实现五个经典的计算机视觉模型，从简单到复杂递进，帮助读者深入理解算法的核心原理。

&emsp;&emsp;本项目是 **23级实验班计算机视觉（1）大作业** 的成果。与传统的课程项目不同，本项目的核心创新点在于**与大语言模型（LLM）共创**。我们将 LLM 视为虚拟的"资深算法工程师"和"私人导师"，在它的指导下，完成了从项目结构设计、代码实现、调试优化到文档撰写的整个软件工程流程。

### 章节规划

本项目计划实现以下五个模型，难度递进：

1. **第一章：[图像分类器 - K-近邻 (K-NN)](./knn/chapter_1_knn.md)** 
2. **第二章：[线性分类器 - Softmax 分类器](./softmax/chapter_2_softmax.md)** 
3. **第三章：[两层全连接神经网络](./two_layer_network/chapter_3.md)** 
4. **第四章：[卷积神经网络 (CNN) - 简化版](./cnn/chapter_4_cnn.md)** 
5. **第五章：[循环神经网络 (RNN) - 基础版](./rnn/chapter5_rnn.md)** 

---

## 第一章：K-近邻 (K-NN) 分类器成果概览

&emsp;&emsp;作为本书的开篇章节，我们不仅实现了 K-NN 算法，更围绕它构建了一套完整的实验和分析流程。

* **理论深度**：我们提供了从现实类比、核心要素、数学定义到算法流程的详尽讲解。
* **代码质量**：`KNNClassifier` 的实现不仅是纯 NumPy 手搓，还包含了高效的**向量化距离计算**、内存安全的分块处理和严谨的**投票平局打破机制**。
* **实验广度**：我们设计了 **5** 个独立的实验，在 CIFAR-10、MNIST、Digits 等多个数据集上对算法进行了全面验证，并深入分析了"维度灾难"等现象。

| 实验名称 | 数据集 | 核心功能 | 主要发现 |
| :--- | :--- | :--- | :--- |
| **CIFAR-10 实验** | 彩色图像 (32x32) | 最优K值搜索 | 最佳准确率仅 **35.61%**，揭示了像素距离在复杂图像上的局限性。 |
| **Digits 实验** | 手写数字 (8x8) | 混淆矩阵分析 | 准确率高达 **98.89%**，证明算法在低维、结构化数据上的优异表现。 |
| **MNIST 实验** | 手写数字 (28x28) | 大规模数据验证 | 准确率 **96.91%**，验证了算法在更大数据集上的稳定性。 |

> 👉 **想要深入了解 K-NN 的所有细节？请阅读 [第一章的完整内容](./knn/chapter_1_knn.md)。**

---

## 第二章：Softmax 分类器成果概览

&emsp;&emsp;在第二章中，我们实现了 Softmax 线性分类器，并深入探讨了特征工程的重要性。

* **理论与实现**：我们详细推导了 Softmax 函数、交叉熵损失和梯度，并通过纯 NumPy 实现了**向量化**、**数值稳定**且包含**偏置技巧**的 `SoftmaxClassifier`。
* **特征工程对比**：核心亮点在于对比了**原始像素**和 **HOG 特征** 对**同一线性模型**性能的巨大影响。
* **深度分析**：通过超参数搜索、训练曲线、权重可视化和**混淆矩阵**，我们揭示了线性模型的局限性以及 HOG 特征的优势与不足。

| 实验名称 | 特征 | 核心功能 | 主要发现 |
| :--- | :--- | :--- | :--- |
| **CIFAR-10 像素基线** | 原始像素 (3073维) | 建立基准 | 最佳测试准确率 **38.70%**，仅比 K-NN 略好，验证了线性模型在原始像素上的瓶颈。 |
| **CIFAR-10 HOG 特征** | HOG 特征 (325维) | 特征工程 | 准确率飙升至 **53.60%** (+14.9%)，证明了优质特征远比模型本身更重要。混淆矩阵显示模型在"交通工具"上表现好，但在"动物"类别（如猫狗）上混淆严重。 |
| **超参数搜索 (HOG)** | HOG 特征 | 寻找最优参数 | 验证了 `lr=0.005`, `reg=0.0005` 是当时 HOG 模型的最佳参数组合。 |

> 👉 **想要了解 Softmax 和特征工程的威力？请阅读 [第二章的完整内容](./softmax/chapter_2_softmax.md)。**

---

## 第三章：两层全连接神经网络成果概览

&emsp;&emsp;在第三章中，我们从线性模型迈向非线性模型，从零开始"手搓"了一个完整的两层全连接神经网络。

* **理论与实现**：我们详细推导了包含 **ReLU 激活函数**的**反向传播**过程，并实现了一个功能完备的 `TwoLayerNetwork` 类，该类**显式**管理 $W_1, b_1, W_2, b_2$ 参数。
* **先进技术**：我们的实现集成了**He 权重初始化**（针对 ReLU）、**L2 正则化**、**Dropout**（使用 Inverted Dropout 技巧）以及**学习率衰减**和**早停**等现代训练策略。
* **深度分析**：通过在 CIFAR-10 原始像素上进行实验，我们对模型进行了全面的可视化分析（包括训练曲线、混淆矩阵、每类准确率、错误样本，以及**第一层隐藏权重 `W1` 的可视化**）。

| 实验名称 | 特征 | 核心功能 | 主要发现 |
| :--- | :--- | :--- | :--- |
| **CIFAR-10 像素基线** | 原始像素 (3072维) | 验证非线性模型 | 最佳测试准确率 **49.84%**。**显著优于** Softmax 像素基线 (38.70%)，证明了**非线性隐藏层**能自动学习像素特征。`W1` 可视化也显示模型学到了边缘/颜色等基础模式。 |
| **(与第二章对比)** | - | 对比特征工程 | 49.84% 的准确率**低于** Softmax + HOG (53.60%)。这揭示了一个深刻见解：**一个简单的模型配合优质的手工特征，有时可以胜过一个无法利用空间结构的简单神经网络**。 |
| **错误分析** | 原始像素 (3072维) | 分析模型瓶颈 | 混淆矩阵和错误样本分析显示，模型依然在"动物"类别（如 `cat`, `dog`, `bird`）上表现很差，这暴露了**全连接层（FCN）无法有效处理图像空间结构**的根本局限性。 |

> 👉 **想要探究两层网络如何工作以及它的局限性？请阅读 [第三章的完整内容](./two_layer_network/chapter_3.md)。**

---

## 第四章：卷积神经网络 (CNN) 成果概览

&emsp;&emsp;在第四章中，我们终于上了"正牌"图像模型——卷积神经网络。相较于第三章的全连接网络，本章的 CNN 显式利用了图像的**空间结构**，在 CIFAR-10 上取得了明显更好的表现。

* **模型结构**：严格采用教材中的**简化版 CNN**：

$$
\begin{gathered}
\text{Conv}(3\times3, 32) \to \text{ReLU} \to \text{MaxPool}(2\times2) \to
\text{Conv}(3\times3, 64) \to \text{ReLU} \\
\downarrow \\
\text{MaxPool}(2\times2) \to
\text{FC}(100) \to \text{ReLU} \to \text{FC}(10) \to \text{Softmax}
\end{gathered}
$$

* **纯 NumPy 手搓实现**：
  - 自己实现 `conv_forward_fast` / `max_pool_forward_fast` 等卷积与池化算子；
  - 自定义 `Cifar10SimpleConvNet` 类管理参数字典 `W1, b1, W2, b2, W3, b3`；
  - 使用带 momentum 的 SGD 训练器，支持学习率衰减和早停；
  - 最佳模型在 CIFAR-10 测试集上达到约 **70.52%** 的分类准确率。

* **可视化与分析**：
  - 通过 `cnn.visualize_confusion_cnn` 生成**混淆矩阵**，分析各类别准确率。结果显示：
    - `car, truck, plane, horse` 等刚性物体类别表现较好；
    - `cat, dog, bird, deer` 等细粒度"动物类"仍然存在较明显混淆。
  - 通过 `cnn.visualize_features_cnn` 可视化：
    - 第一层卷积核学到的多种方向的**边缘/纹理检测器**；
    - 若干测试样本在 Conv1 后的**特征图**，可以直观看到网络如何逐步"抹掉背景、突出主体"。

* **与前三章对比的关键结论**：
  - 同样使用**原始像素**作为输入：
    - K-NN：≈ 35%；
    - Softmax：≈ 39%；
    - 两层 FCN：≈ 50%；
    - **简化版 CNN：≈ 70.52%**。
  - 这说明：在图像任务中，**结构化的模型（卷积 + 池化 + 局部感受野 + 权重共享）本身就是一种"强特征工程"**，相比简单地在扁平化像素上做全连接，CNN 能自动学习到更高级、更具空间意义的表征。

> 👉 **想要看卷积实现细节、训练曲线和卷积核/特征图截图？请阅读 [第四章的完整内容](./cnn/chapter_4_cnn.md)。**

---

## 第五章：循环神经网络 (RNN) 成果概览 ✨

&emsp;&emsp;在第五章中，我们实现了三种 RNN 变体（Vanilla RNN、LSTM、GRU），并通过图像序列化实验揭示了它们在短序列任务上的性能差异和梯度稳定性特征。

* **序列化策略（Row-by-Row）**：
  - 把每张 CIFAR-10 图像视为长度为 $T=32$ 的序列（逐行扫描）；
  - 每个时间步输入维度为 $D=32\times 3=96$；
  - 输入张量形状：$X\in\mathbb{R}^{N\times 32\times 96}$。

* **纯 NumPy 手搓实现三种模型**：
  - **Vanilla RNN**：经典 tanh 循环结构，使用 $W_{hh} = I + 0.01\mathcal{N}(0,1)$ 稳定初始化；
  - **LSTM**：四个门控机制（遗忘门、输入门、输出门、候选值）+ cell state；
  - **GRU**：简化的门控结构（重置门、更新门）。
  - 所有模型均包含完整的 BPTT 反向传播、全局梯度裁剪、early stopping 和学习率衰减。

* **实验结果与关键发现**：

| 模型 | 测试准确率 | 梯度范数变化 | 训练状态 | 核心特征 |
| :--- | :--- | :--- | :--- | :--- |
| **Vanilla RNN** | **49.30%** 🏆 | 2.45 → 13.89 ⚠️ | Early Stop (Epoch 182) | **表现最好**！稳定初始化让短序列训练成功，但有梯度爆炸倾向。 |
| **LSTM** | **49.20%** 🥈 | 0.87 → 2.71 ✅ | 完整 200 epochs | **梯度最稳定**，泛化能力强（train-val gap 仅 3%）。 |
| **GRU** | **41.10%** 🥉 | 1.17 → 2.11 ✅ | 完整 200 epochs | 梯度稳定，但性能较差（超参数可能未优化）。 |

* **深度分析与反直觉发现**：
  
  **发现一：为什么 Vanilla RNN 表现最好？**
  - 序列长度（$T=32$）不够长，无法体现 LSTM/GRU 处理长程依赖的优势；
  - 稳定的 $W_{hh}$ 初始化策略是关键；
  - LSTM/GRU 参数量更多（4倍），在小数据集上优化更困难。

  **发现二：门控机制确实提升梯度稳定性**
  - Vanilla RNN 的原始梯度范数持续增长（2.45 → 13.89），虽然用了梯度裁剪但仍有爆炸倾向；
  - LSTM/GRU 的梯度范数始终稳定在低值（<3），证明门控机制的理论优势。

  **发现三：模型选择需考虑任务特性**
  - 图像的 row-by-row 序列化并非 RNN 的理想应用；
  - 在真正的长序列任务（语言建模、机器翻译）上，LSTM/GRU 会显著优于 Vanilla RNN。

* **与前四章的性能对比（原始像素）**：
  - K-NN：≈ 35%
  - Softmax：≈ 39%
  - 两层 FCN：≈ 50%
  - **Vanilla RNN：≈ 49%**（与 FCN 持平）
  - 简化版 CNN：≈ 71%

  结论：**序列建模视角能学到一定判别信息，但无法匹敌 CNN 的空间归纳偏置**。

> 👉 **想要看 RNN/LSTM/GRU 的 BPTT 推导、训练曲线和梯度稳定性分析？请阅读 [第五章的完整内容](./rnn/chapter5_rnn.md)。**

---

## 🛠️ 环境配置与运行

 ### 系统要求
 * **操作系统**：Linux 环境 (推荐 WSL2、Docker 或虚拟机)
 * **Python 版本**：3.8+
+* **内存要求**：运行第 5 章 RNN（尤其是数据处理/序列化阶段）建议保证 **可用运行内存 ≥ 2GB**。
+  若可用内存 < 2GB，可能出现内存溢出（OOM）报错。可尝试：关闭其他程序、增加 WSL/Docker 分配内存、或降低 batch size。

### 安装与运行
1. **克隆仓库**
   ```bash
   git clone https://github.com/huzekun123hzk-ship-it/cs-vision-homework.git
   cd cs-vision-homework
   ```

2. **创建并激活虚拟环境**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

   *注意：为了在 Linux 环境下正确显示图表中的中文，您可能需要安装中文字体，例如：`sudo apt-get install -y fonts-wqy-microhei`*

4. **运行实验**

   * 各章节的实验脚本分别位于 `knn/experiments/`、`softmax/experiments/`、`two_layer_network/experiments/`、`cnn/` 与 `rnn/experiments/` 目录下。
   
   * **示例 1：运行 K-NN Digits 实验 (K=3)**
     ```bash
     python3 -m knn.experiments.digits_experiment --k 3
     ```
   
   * **示例 2：运行 Softmax HOG 特征实验**
     ```bash
     python3 -m softmax.experiments.cifar10_experiment_hog
     ```
   
   * **示例 3：运行两层全连接网络实验**
     ```bash
     python3 -m two_layer_network.experiments.cifar10_experiment
     ```
   
   * **示例 4：运行简化版 CNN 训练 (CIFAR-10)**
     ```bash
     python -m cnn.experiment_cifar10_cnn \
       --num-epochs 20 \
       --batch-size 128 \
       --learning-rate 1e-2 \
       --update sgd_momentum
     ```
   
   * **示例 5：CNN 混淆矩阵可视化**
     ```bash
     python -m cnn.visualize_confusion_cnn \
       --data-dir ./data/cifar-10-batches-py \
       --model-path ./cnn/experiments/results/cnn_cifar10_best.npz \
       --results-dir ./cnn/experiments/results
     ```
   
   * **示例 6：CNN 卷积核 & 特征图可视化**
     ```bash
     python -m cnn.visualize_features_cnn \
       --data-dir ./data/cifar-10-batches-py \
       --model-path ./cnn/experiments/results/cnn_cifar10_best.npz \
       --results-dir ./cnn/experiments/results
     ```
   
   * **示例 7：运行 Vanilla RNN CIFAR-10 实验**
     ```bash
     cd rnn/experiments
     python cifar10_experiment.py
     ```
   
   * **示例 8：运行 LSTM CIFAR-10 实验**
     ```bash
     cd rnn/experiments
     python lstm_experiment.py
     ```
   
   * **示例 9：运行 GRU CIFAR-10 实验**
     ```bash
     cd rnn/experiments
     python gru_experiment.py
     ```
   
   * **示例 10：RNN/LSTM/GRU 性能对比实验**
     ```bash
     cd rnn/experiments
     python compare_rnn_variants.py
     ```
   
   * **示例 11：RNN 小样本过拟合调试（sanity check）**
     ```bash
     cd rnn/experiments
     python overfit_debug.py
     ```

---

## 🤖 与 LLM 的协作记录

&emsp;&emsp;本项目全程在 LLM 的指导下进行。我们详细记录了在**项目结构设计、代码重构、Bug修复、实验分析、特征工程探索**等关键环节与 AI 的协作过程。

> 👉 **查看 K-NN 章节的 [LLM 协作日志](./llm_interactions/knn_chapter_logs.md)。**

> 👉 **查看 Softmax 章节的 [LLM 协作日志](./llm_interactions/softmax_chapter_logs.md)。**

> 👉 **查看两层网络章节的 [LLM 协作日志](./llm_interactions/two_layer_network_logs.md)。**

> 👉 **查看 CNN 章节的 [LLM 协作日志](./llm_interactions/cnn_chapter_logs.md)。**

> 👉 **查看 RNN 章节的 [LLM 协作日志](./llm_interactions/rnn_chapter_logs.md)。**

---

## 📊 全书性能对比总结

&emsp;&emsp;在相同的 CIFAR-10 数据集上，五种模型使用**原始像素**作为输入的性能对比：

| 章节 | 模型 | 测试准确率 | 参数量 | 核心特征 |
| :--- | :--- | :---: | :---: | :--- |
| **第一章** | K-NN | 35.61% | 0 | 非参数模型，受维度灾难影响 |
| **第二章** | Softmax | 38.70% | ~30K | 线性模型，无法捕获非线性模式 |
| **第三章** | 两层 FCN | 49.84% | ~400K | 非线性但无空间归纳偏置 |
| **第四章** | **简化版 CNN** | **70.52%** | ~200K | **空间归纳偏置，适合视觉任务** |
| **第五章** | Vanilla RNN | 49.30% | ~255K | 序列建模，适合时序数据 |
| **第五章** | LSTM | 49.20% | ~1.05M | 门控机制，梯度最稳定 |
| **第五章** | GRU | 41.10% | ~768K | 简化门控，需调优 |


**关键结论**：
1. **特征比模型更重要**：Softmax + HOG (53.60%) > 两层 FCN + 原始像素 (49.84%)
2. **归纳偏置至关重要**：CNN 的卷积结构天然适配图像，显著优于其他模型
3. **RNN 的价值在序列**：在图像任务上表现平平，但在真正的序列任务（NLP、时序）上会展现优势
4. **门控机制的价值**：LSTM/GRU 虽然性能相近，但梯度稳定性显著优于 Vanilla RNN


## 贡献指南

&emsp;&emsp;我们欢迎任何形式的贡献，无论是报告问题、提交代码还是提出改进建议！请遵循标准的 Fork & Pull Request 流程。

## 许可证

&emsp;&emsp;本项目采用 MIT 许可证。详情请查看 [LICENSE](./LICENSE) 文件。

## 安全策略

我们重视本项目的安全性。如果您发现了任何安全漏洞，请负责任地向我们报告。

> 👉 **查看完整的 [安全策略 (SECURITY.md)](./SECURITY.md)。**
