# 《用纯Python手搓经典计算机视觉算法》开源教材项目

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 项目总览

&emsp;&emsp;欢迎来到《用纯Python手搓经典计算机视觉算法》！这是一本开源教材项目，旨在通过纯`Python`和`NumPy`实现五个经典的计算机视觉模型，从简单到复杂递进，帮助读者深入理解算法的核心原理。

&emsp;&emsp;本项目是 **23级实验班计算机视觉（1）大作业** 的成果。与传统的课程项目不同，本项目的核心创新点在于**与大语言模型（LLM）共创**。我们将 LLM 视为虚拟的“资深算法工程师”和“私人导师”，在它的指导下，完成了从项目结构设计、代码实现、调试优化到文档撰写的整个软件工程流程。

### 章节规划

本项目计划实现以下五个模型，难度递进：

1.  **第一章：[图像分类器 - K-近邻 (K-NN)](./knn/chapter_1_knn_theory.md)** ✅ **已完成**
2.  **第二章：线性分类器 - Softmax 分类器** 📝 *待完成*
3.  **第三章：两层全连接神经网络** 📝 *待完成*
4.  **第四章：卷积神经网络 (CNN) - 简化版** 📝 *待完成*
5.  **第五章：循环神经网络 (RNN) - 基础版** 📝 *待完成*

---

## 🚀 第一章：K-近邻 (K-NN) 分类器成果概览

&emsp;&emsp;作为本书的开篇章节，我们不仅实现了 K-NN 算法，更围绕它构建了一套完整的实验和分析流程。

* **理论深度**：我们提供了从现实类比、核心要素、数学定义到算法流程的详尽讲解。
* **代码质量**：`KNNClassifier` 的实现不仅是纯 NumPy 手搓，还包含了高效的**向量化距离计算**、内存安全的分块处理和严谨的**投票平局打破机制**。
* **实验广度**：我们设计了 **5** 个独立的实验，在 CIFAR-10、MNIST、Digits 等多个数据集上对算法进行了全面验证，并深入分析了“维度灾难”等现象。

| 实验名称 | 数据集 | 核心功能 | 主要发现 |
| :--- | :--- | :--- | :--- |
| **CIFAR-10 实验** | 彩色图像 (32x32) | 最优K值搜索 | 最佳准确率仅 **35.61%**，揭示了像素距离在复杂图像上的局限性。 |
| **Digits 实验** | 手写数字 (8x8) | 混淆矩阵分析 | 准确率高达 **98.89%**，证明算法在低维、结构化数据上的优异表现。 |
| **MNIST 实验** | 手写数字 (28x28) | 大规模数据验证 | 准确率 **96.91%**，验证了算法在更大数据集上的稳定性。 |
| **Toy Dataset 实验**| 2D 虚拟数据 | 决策边界可视化 | 直观展示了 K 值从 1 到 15 时，决策边界从“过拟合”到“平滑”的变化过程。 |
| **自定义图像实验**| 花卉照片 | 通用性测试 | 成功应用于任意自定义图像文件夹，展示了算法的通用性。 |

> 👉 **想要深入了解 K-NN 的所有细节？请阅读 [第一章的完整内容](./knn/chapter_1_knn_theory.md)。**

---

## 🛠️ 环境配置与运行

### 系统要求
* **操作系统**：Linux 环境 (推荐 WSL2、Docker 或虚拟机)
* **Python 版本**：3.8+

### 安装与运行
1.  **克隆仓库**
    ```bash
    git clone [https://github.com/huzekun123hzk-ship-it/cs-vision-homework.git](https://github.com/huzekun123hzk-ship-it/cs-vision-homework.git)
    cd cs-vision-homework
    ```

2.  **创建并激活虚拟环境**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```
    *注意：为了在 Linux 环境下正确显示图表中的中文，您可能需要安装中文字体，例如：`sudo apt-get install -y fonts-wqy-microhei`*

4.  **运行实验**
    * 所有实验脚本都位于 `knn/experiments/` 目录下，并支持丰富的命令行参数。详细的运行指南请参考 [第一章文档](./knn/chapter_1_knn_theory.md) 的“实验总览”部分。
    * **示例：运行 Digits 手写数字分类实验**
        ```bash
        python3 -m knn.experiments.digits_experiment --k 3
        ```

## 🤖 与 LLM 的协作记录

&emsp;&emsp;本项目全程在 LLM 的指导下进行。我们详细记录了在**项目结构设计、代码重构、Bug修复、实验分析**等关键环节与 AI 的协作过程。

> 👉 **查看完整的 [LLM 协作日志](./llm_interactions/knn_chapter_logs.md)。**

## 🤝 贡献指南

&emsp;&emsp;我们欢迎任何形式的贡献，无论是报告问题、提交代码还是提出改进建议！请遵循标准的 Fork & Pull Request 流程。

## 📜 许可证

&emsp;&emsp;本项目采用 MIT 许可证。详情请查看 [LICENSE](./LICENSE) 文件。

## 🔐 安全策略

我们重视本项目的安全性。如果您发现了任何安全漏洞，请负责任地向我们报告。

> 👉 **查看完整的 [安全策略 (SECURITY.md)](./SECURITY.md)。**