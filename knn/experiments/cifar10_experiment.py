# knn/experiments/cifar10_experiment.py

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# --- 导入自定义模块 ---
# 将上级目录 (knn/) 添加到 Python 的模块搜索路径中
# 以便能够找到位于 knn/ 目录下的模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from knn_classifier import KNNClassifier
from .cifar10_utils import load_cifar10, visualize_samples, CIFAR10_CLASSES

def setup_matplotlib_font():
    """设置 Matplotlib 的中文字体，以正确显示图表中的中文。"""
    try:
        plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]
        plt.rcParams["axes.unicode_minus"] = False
        print("✓ 中文字体 'WenQuanYi Micro Hei' 设置成功。")
    except Exception as e:
        print(f"警告: 设置中文字体失败。错误: {e}")

def main():
    """主函数，执行在 CIFAR-10 上的 K-NN 分类实验。"""
    
    setup_matplotlib_font()
    
    # --- 1. 设置路径并加载数据 ---
    # 获取当前实验脚本所在的目录
    current_experiment_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建到项目根目录下的 data 文件夹的路径
    data_dir = os.path.join(current_experiment_dir, '../../data/cifar-10-batches-py')

    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 '{data_dir}' 不存在。")
        return

    print("正在加载 CIFAR-10 数据集...")
    X_train_full, y_train_full, X_test_full, y_test_full = load_cifar10(data_dir)
    print("✓ 数据加载完成。")
    print(f"  - 完整训练集: {X_train_full.shape[0]} 个样本")
    print(f"  - 完整测试集: {X_test_full.shape[0]} 个样本")

    # --- 2. 数据预处理与采样 ---
    num_train_samples = 50000
    num_test_samples = 10000

    X_train = X_train_full[:num_train_samples]
    y_train = y_train_full[:num_train_samples]
    X_test = X_test_full[:num_test_samples]
    y_test = y_test_full[:num_test_samples]

    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0
    
    print(f"✓ 本次实验使用训练集大小: {X_train.shape[0]}")
    print(f"✓ 本次实验使用测试集大小: {X_test.shape[0]}")

    # --- 3. 寻找最佳 K 值并记录结果 ---
    k_candidates = [1, 3, 5, 8, 10, 12, 15] # 减少k的数量以加快实验
    
    # 将结果保存在专属的 cifar10_results/ 文件夹下
    results_dir = os.path.join(current_experiment_dir, "cifar10_results")
    os.makedirs(results_dir, exist_ok=True)
    results_filename = os.path.join(results_dir, "cifar10_experiment_results.txt")
    
    print(f"\n正在通过测试不同K值寻找最优解... (结果将保存至 {results_dir})")
    print("="*40)
    
    best_k = -1
    best_accuracy = -1.0
    
    # 使用 L2 (欧氏) 距离进行实验
    knn = KNNClassifier(p=2)
    knn.fit(X_train, y_train)
    
    with open(results_filename, "w", encoding="utf-8") as f:
        f.write("K-NN 在 CIFAR-10 上的实验结果\n")
        f.write(f"训练集大小: {num_train_samples}, 测试集大小: {num_test_samples}, 距离: L2 (Euclidean)\n")
        f.write("="*40 + "\n")
        
        for k in k_candidates:
            print(f"正在测试 k = {k}...")
            # k 值在 predict 方法中传入
            y_pred = knn.predict(X_test, k=k)
            
            num_correct = np.sum(y_pred == y_test)
            accuracy = float(num_correct) / num_test_samples
            
            log_message = f"当 K = {k:2d} 时, 准确率为: {accuracy:.4f}"
            print(log_message)
            f.write(log_message + "\n")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k
        
        final_summary = f"\n实验完成。最佳 K 值为: {best_k}, 对应准确率为: {best_accuracy:.4f}"
        print("="*40)
        print(final_summary)
        f.write("="*40 + "\n")
        f.write(final_summary + "\n")
    
    print(f"✓ 所有实验结果已保存。")

    # --- 4. 使用最佳模型可视化部分预测结果 ---
    print(f"\n正在使用最佳 K 值 (k={best_k}) 可视化部分预测结果...")
    y_pred_final = knn.predict(X_test, k=best_k)

    num_display = 20
    display_indices = np.random.choice(num_test_samples, num_display, replace=False)
    
    X_display = X_test[display_indices]
    y_display_true = y_test[display_indices]
    y_display_pred = y_pred_final[display_indices]
    
    plt.figure(figsize=(15, 10))
    for i in range(num_display):
        image_array = X_display[i].reshape(3, 32, 32).transpose(1, 2, 0)
        
        true_label = CIFAR10_CLASSES[y_display_true[i]]
        pred_label = CIFAR10_CLASSES[int(y_display_pred[i])]
        
        plt.subplot(4, 5, i + 1)
        plt.imshow(image_array)
        
        title_color = 'green' if true_label == pred_label else 'red'
        plt.title(f"真实: {true_label}\n预测: {pred_label}", color=title_color)
        plt.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) 
    plt.suptitle("CIFAR-10 部分样本预测结果展示", fontsize=16, y=0.98) 
    
    visualization_path = os.path.join(results_dir, "cifar10_prediction_visualization.png")
    plt.savefig(visualization_path, bbox_inches='tight', dpi=300) 
    plt.close()
    print(f"✓ 预测结果可视化图片已保存至: {visualization_path}")

if __name__ == '__main__':
    main()