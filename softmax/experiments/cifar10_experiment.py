"""
CIFAR-10 Softmax分类器实验

完整的实验流程：
1. 加载CIFAR-10数据集
2. 数据预处理
3. 训练Softmax分类器
4. 评估模型性能
5. 生成可视化结果
"""

import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from softmax_classifier import SoftmaxClassifier
from cifar10_utils import load_cifar10, preprocess_data, get_cifar10_class_names


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
        'num_epochs': 1000,           # 最大训练轮数
        'batch_size': 128,           # 批次大小
        'patience': 200,             # 早停耐心值
        'print_every': 20,           # 打印间隔
        'lr_decay_epochs': 150,       # 学习率衰减间隔
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
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()
    
    for i in range(10):
        w = W[:3072, i].reshape(32, 32, 3)
        w_normalized = (w - w.min()) / (w.max() - w.min())
        
        axes[i].imshow(w_normalized)
        axes[i].set_title(class_names[i], fontsize=11, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle('Learned Weight Templates for Each Class', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 权重可视化已保存: {save_path.name}")


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

def visualize_predictions(X_test, y_test, classifier, class_names, save_path, num_samples=20):
    """可视化模型预测结果"""
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()
        
    for idx, ax in enumerate(axes):
        i = indices[idx]
        img = X_test[i, :3072].reshape(32, 32, 3)
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = np.clip(img, 0, 1)
        
        pred = classifier.predict(X_test[i:i+1])[0]
        true = y_test[i]
        
        ax.imshow(img)
        color = 'green' if pred == true else 'red'
        
        ax.set_title(f'True: {class_names[true]}Pred: {class_names[pred]}',
            color=color, fontsize=10, fontweight='bold')

        ax.axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 预测可视化已保存: {save_path.name}")


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
    print(" " * 15 + "CIFAR-10 Softmax分类器实验")
    print("=" * 70)
    
    # 打印配置
    print_config()
    
    # 设置结果目录
    results_dir = Path(__file__).parent / 'cifar10_results'
    results_dir.mkdir(exist_ok=True)
    
    # ==================== 1. 加载数据 ====================
    print("[1/6] 加载CIFAR-10数据集...")
    print("-" * 70)
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(
        data_dir='../../data/cifar-10-batches-py',
        train_samples=CONFIG['data']['train_samples'],
        val_samples=CONFIG['data']['val_samples'],
        test_samples=CONFIG['data']['test_samples']
    )
    print(f"✓ 训练集: {len(X_train)} 样本")
    print(f"✓ 验证集: {len(X_val)} 样本")
    print(f"✓ 测试集: {len(X_test)} 样本")
    print(f"✓ 图像尺寸: {X_train.shape[1:]}")
    
    # ==================== 2. 预处理 ====================
    print("[2/6] 数据预处理...")
    print("-" * 70)
    X_train, X_val, X_test = preprocess_data(X_train, X_val, X_test)
    print(f"✓ 展平后特征维度: {X_train.shape[1]} (3072像素 + 1偏置)")
    print(f"✓ 数据已中心化（减去训练集均值）")
    print(f"✓ 已添加偏置项")
    
    # ==================== 3. 创建分类器 ====================
    print("[3/6] 创建Softmax分类器...")
    print("-" * 70)
    classifier = SoftmaxClassifier(
        num_features=X_train.shape[1],
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
        f.write("=" * 70 + "")
        f.write("CIFAR-10 Softmax分类器实验结果")
        f.write("=" * 70 + "")
        
        # 保存配置
        f.write("实验配置:")
        f.write("-" * 70 + "")
        f.write("数据集:")
        for key, value in CONFIG['data'].items():
            f.write(f"  {key}: {value}")
        f.write("模型:")
        for key, value in CONFIG['model'].items():
            f.write(f"  {key}: {value}")
        f.write("训练:")
        for key, value in CONFIG['training'].items():
            f.write(f"  {key}: {value}")
        f.write("" + "-" * 70 + "")
        
        f.write("数据集信息:")
        f.write(f"  训练集: {len(X_train)} 样本")
        f.write(f"  验证集: {len(X_val)} 样本")
        f.write(f"  测试集: {len(X_test)} 样本")
        f.write(f"  特征维度: {X_train.shape[1]}")
        
        f.write("最终性能:")
        f.write(f"  训练准确率: {train_acc:.4f} ({train_acc*100:.2f}%)")
        f.write(f"  验证准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
        f.write(f"  测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        f.write("训练历史:")
        f.write("-" * 70 + "")
        f.write(f"{'Epoch':<10}{'Loss':<15}{'Train Acc':<15}{'Val Acc':<15}")
        f.write("-" * 70 + "")
        for i, epoch in enumerate(history['epochs']):
            f.write(f"{epoch:<10}{history['loss_history'][epoch]:<15.4f}"
                   f"{history['train_acc_history'][i]:<15.4f}"
                   f"{history['val_acc_history'][i]:<15.4f}")
        
        f.write("" + "=" * 70 + "")
    
    print(f"✓ 文本结果已保存: {results_file.name}")
    
    # 6.3 生成可视化
    print("生成可视化...")
    print("-" * 70)
    
    curves_path = results_dir / 'training_curves.png'
    plot_training_curves(history, curves_path)
    
    weights_path = results_dir / 'weight_visualization.png'
    class_names = get_cifar10_class_names()
    visualize_weights(classifier.W, class_names, weights_path)
    
    pred_path = results_dir / 'cifar10_prediction_visualization.png'
    visualize_predictions(
        X_test, y_test, classifier, class_names, pred_path,
        num_samples=CONFIG['visualization']['num_pred_samples']
    )
    
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
    print(f"   ├── weight_visualization.png")
    print(f"   ├── cifar10_prediction_visualization.png")
    print(f"   └── softmax_classifier.npy")
    print("=" * 70)


if __name__ == '__main__':
    main()
