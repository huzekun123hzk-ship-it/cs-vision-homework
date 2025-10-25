"""
CIFAR-10 Softmax分类器实验 (HOG 特征版)

完整的实验流程：
1. 加载CIFAR-10数据集
2. 数据预处理 (提取HOG特征)
3. 训练Softmax分类器
4. 评估模型性能
5. 生成可视化结果 (包括混淆矩阵)
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from softmax_classifier import SoftmaxClassifier
from cifar10_utils_hog import load_cifar10, preprocess_data, get_cifar10_class_names

# 新增导入, 用于混淆矩阵
from sklearn.metrics import confusion_matrix
import seaborn as sns 
# ============================================================================
# 实验配置 - 在这里修改所有超参数
# ============================================================================
CONFIG = {
    # 数据集配置
    'data': {
        'train_samples': 49000,
        'val_samples': 1000,
        'test_samples': 1000,
    },
    
    # 模型配置
    'model': {
        'reg_strength': 5e-4,  # L2正则化强度
    },
    
    # 训练配置
    'training': {
        'learning_rate': 0.005,      # 初始学习率
        'num_epochs': 1000,          # 最大训练轮数
        'batch_size': 128,           # 批次大小
        'patience': 200,             # 早停耐心值
        'print_every': 20,           # 打印间隔
        'lr_decay_epochs': 150,      # 学习率衰减间隔
        'lr_decay_rate': 0.99,       # 学习率衰减率
    },
    
    # 可视化配置
    'visualization': {
        'num_pred_samples': 20,      # 预测可视化样本数
    }
}
# ============================================================================


def visualize_weights(W, class_names, save_path):
    """可视化学习到的权重模板"""
    # (注意: HOG 特征权重 (325维) 无法被 reshape 为 (32, 32, 3))
    # (此函数在 HOG 实验中不应被调用)
    print("⚠️ 警告: visualize_weights 无法用于 HOG 特征。跳过...")
    return 


def plot_training_curves(history, save_path):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = history['epochs']
    
    # 损失曲线 - 转换字典为列表
    loss_epochs = sorted(history['loss_history'].keys())
    loss_values = [history['loss_history'][e] for e in loss_epochs]
    
    ax1.plot(loss_values, 'b-', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, len(loss_values))
    
    # 准确率曲线
    ax2.plot(epochs, history['train_acc_history'], 'b-', 
             label='Training Accuracy', linewidth=2, alpha=0.8)
    ax2.plot(epochs, history['val_acc_history'], 'r-', 
             label='Validation Accuracy', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(epochs))
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_predictions(X_raw, y_test, classifier, X_processed, class_names, save_path, num_samples=20):
    """可视化模型预测结果"""
    indices = np.random.choice(len(X_raw), num_samples, replace=False)
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        i = indices[idx]
        
        # 使用 X_raw (原始像素) 来显示图像
        img = X_raw[i] # X_raw 已经是 (32, 32, 3)
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = np.clip(img, 0, 1)
        
        # 使用 X_processed (HOG特征) 来进行预测
        pred = classifier.predict(X_processed[i:i+1])[0]
        true = y_test[i]
        
        ax.imshow(img)
        color = 'green' if pred == true else 'red'
        
        ax.set_title(f'True: {class_names[true]}\nPred: {class_names[pred]}',
               color=color, fontsize=10, fontweight='bold')

        ax.axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(pad=0.5, h_pad=1.0) # 调整布局防止标题重叠
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 预测可视化已保存: {save_path.name}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    绘制并保存混淆矩阵热力图
    """
    print(f"生成混淆矩阵...")
    
    # 1. 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 2. 归一化 (按行, 显示召回率/真实为A的, 有多少被预测为B)
    # 加 1e-6 避免除零
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    plt.figure(figsize=(10, 8))
    
    # 3. 使用 Seaborn 绘制热力图
    sns.heatmap(cm_normalized, 
                annot=True,     # 在格子里显示数字
                fmt='.2f',      # 数字格式 (两位小数)
                cmap='Blues',   # 颜色
                xticklabels=class_names,
                yticklabels=class_names)
    
    plt.ylabel('True Label (真实类别)', fontsize=13)
    plt.xlabel('Predicted Label (预测类别)', fontsize=13)
    plt.title('Confusion Matrix (Normalized by True Label)', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ 混淆矩阵已保存: {save_path.name}")


def print_config():
    """打印当前配置"""
    print("" + "=" * 70)
    print("实验配置:")
    print("=" * 70)
    
    print("📊 数据集:")
    for key, value in CONFIG['data'].items():
        print(f"  • {key}: {value}")
    
    print("🧠 模型:")
    for key, value in CONFIG['model'].items():
        print(f"  • {key}: {value}")
    
    print("🎯 训练:")
    for key, value in CONFIG['training'].items():
        print(f"  • {key}: {value}")
    
    print("🎨 可视化:")
    for key, value in CONFIG['visualization'].items():
        print(f"  • {key}: {value}")
    
    print("=" * 70)


def main():
    """主实验流程"""
    print("=" * 70)
    print(" " * 15 + "CIFAR-10 Softmax分类器实验 (HOG 特征版)")
    print("=" * 70)
    
    # 打印配置
    print_config()
    
    # 设置结果目录
    results_dir = Path(__file__).parent / 'cifar10_hog_results'
    results_dir.mkdir(exist_ok=True)
    
    # ==================== 1. 加载数据 ====================
    print("[1/6] 加载CIFAR-10数据集...")
    print("-" * 70)
    
    # 保留原始图像数据, 用于可视化
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = load_cifar10(
        data_dir='../../data/cifar-10-batches-py',
        train_samples=CONFIG['data']['train_samples'],
        val_samples=CONFIG['data']['val_samples'],
        test_samples=CONFIG['data']['test_samples']
    )
    print(f"✓ 训练集: {len(X_train_raw)} 样本")
    print(f"✓ 验证集: {len(X_val_raw)} 样本")
    print(f"✓ 测试集: {len(X_test_raw)} 样本")
    print(f"✓ 图像尺寸: {X_train_raw.shape[1:]}")
    
    # ==================== 2. 预处理 ====================
    print("[2/6] 数据预处理...")
    print("-" * 70)
    
    # X_train, X_val, X_test 现在是 HOG 特征
    X_train, X_val, X_test = preprocess_data(X_train_raw, X_val_raw, X_test_raw)
    
    # 打印信息来自 HOG 版 utils
    print(f"✓ 展平后特征维度: {X_train.shape[1]} (HOG特征 + 1偏置)") 
    print(f"✓ 数据已中心化（减去训练集均值）")
    print(f"✓ 已添加偏置项")
    
    # ==================== 3. 创建分类器 ====================
    print("[3/6] 创建Softmax分类器...")
    print("-" * 70)
    classifier = SoftmaxClassifier(
        num_features=X_train.shape[1], # HOG 特征维度 (e.g., 325)
        num_classes=10,
        reg_strength=CONFIG['model']['reg_strength']
    )
    print(f"✓ 输入特征数: {classifier.num_features}")
    print(f"✓ 输出类别数: {classifier.num_classes}")
    print(f"✓ 权重参数量: {classifier.W.size:,}")
    print(f"✓ 正则化强度: {classifier.reg_strength}")
    
    # ==================== 4. 训练 ====================
    print("[4/6] 开始训练...")
    print("=" * 70)
    
    history = classifier.train(
        X_train, y_train,
        X_val, y_val,
        learning_rate=CONFIG['training']['learning_rate'],
        num_epochs=CONFIG['training']['num_epochs'],
        batch_size=CONFIG['training']['batch_size'],
        patience=CONFIG['training']['patience'],
        verbose=True,
        print_every=CONFIG['training']['print_every']
    )
    
    # ==================== 5. 评估 ====================
    print("[5/6] 评估模型性能...")
    print("=" * 70)
    train_acc = classifier.evaluate(X_train, y_train)
    val_acc = classifier.evaluate(X_val, y_val)
    test_acc = classifier.evaluate(X_test, y_test)
    
    print(f"训练集准确率: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("=" * 70)
    
    # ==================== 6. 保存结果 ====================
    print("[6/6] 保存实验结果...")
    print("-" * 70)
    
    # 6.1 保存模型权重
    model_path = results_dir / 'softmax_classifier.npy'
    classifier.save_model(str(model_path))
    print(f"✓ 模型权重已保存: {model_path.name}")
    
    # 6.2 保存文本结果（包含配置）
    results_file = results_dir / 'cifar10_experiment_results.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write(" " * 15 + "CIFAR-10 Softmax分类器实验 (HOG 特征)\n")
        f.write("=" * 70 + "\n\n")
        
        # 保存配置
        f.write("实验配置:\n")
        f.write("-" * 70 + "\n")
        f.write("📊 数据集:\n")
        for key, value in CONFIG['data'].items():
            f.write(f"   • {key}: {value}\n")
        f.write("🧠 模型:\n")
        for key, value in CONFIG['model'].items():
            f.write(f"   • {key}: {value}\n")
        f.write("🎯 训练:\n")
        for key, value in CONFIG['training'].items():
            f.write(f"   • {key}: {value}\n")
        f.write("\n" + "-" * 70 + "\n\n")
        
        f.write("数据集信息:\n")
        f.write(f"   训练集: {len(X_train)} 样本\n")
        f.write(f"   验证集: {len(X_val)} 样本\n")
        f.write(f"   测试集: {len(X_test)} 样本\n")
        f.write(f"   特征维度: {X_train.shape[1]} (HOG + 偏置)\n\n")
        
        f.write("最终性能:\n")
        f.write(f"   训练准确率: {train_acc:.4f} ({train_acc*100:.2f}%)\n")
        f.write(f"   验证准确率: {val_acc:.4f} ({val_acc*100:.2f}%)\n")
        f.write(f"   测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)\n\n")
        
        f.write("训练历史:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Epoch':<10}{'Loss':<15}{'Train Acc':<15}{'Val Acc':<15}\n")
        f.write("-" * 70 + "\n")
        for i, epoch in enumerate(history['epochs']):
            f.write(f"{epoch:<10}{history['loss_history'][epoch]:<15.4f}"
                  f"{history['train_acc_history'][i]:<15.4f}"
                  f"{history['val_acc_history'][i]:<15.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"✓ 文本结果已保存: {results_file.name}")
    
    # 6.3 生成可视化
    print("生成可视化...")
    print("-" * 70)
    
    curves_path = results_dir / 'training_curves.png'
    plot_training_curves(history, curves_path)
    
    # HOG 实验不生成权重可视化
    class_names = get_cifar10_class_names()
    
    pred_path = results_dir / 'cifar10_prediction_visualization.png'
    
    visualize_predictions(
        X_test_raw, y_test, classifier, X_test, # 传入原始图像和HOG特征
        class_names, pred_path,
        num_samples=CONFIG['visualization']['num_pred_samples']
    )
    
    # --- 新增：生成混淆矩阵 ---
    print("正在获取测试集所有预测, 用于生成混淆矩阵...")
    y_pred_test = classifier.predict(X_test)
    
    # 定义路径并调用新函数
    cm_path = results_dir / 'confusion_matrix.png'
    plot_confusion_matrix(y_test, y_pred_test, class_names, cm_path)
    # -------------------------
    
    # ==================== 完成 ====================
    print("" + "=" * 70)
    print("✅ 实验完成！")
    print("=" * 70)
    print(f"📊 最终结果:")
    print(f"   测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"📁 结果文件位置:")
    print(f"   {results_dir}/")
    print(f"   ├── cifar10_experiment_results.txt")
    print(f"   ├── training_curves.png")
    print(f"   ├── cifar10_prediction_visualization.png")
    print(f"   ├── confusion_matrix.png") # <--- 已添加
    print(f"   └── softmax_classifier.npy")
    print("=" * 70)


if __name__ == '__main__':
    main()