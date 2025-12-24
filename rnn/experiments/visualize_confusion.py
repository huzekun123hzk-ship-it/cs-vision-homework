# rnn/experiments/visualize_confusion.py
"""
RNN 混淆矩阵可视化脚本

生成：
1. 混淆矩阵（计数版）
2. 混淆矩阵（归一化版）
3. 每类准确率柱状图
"""

from __future__ import annotations
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))

from rnn_classifier import RNNClassifier
from cifar10_utils import load_cifar10, preprocess_data, get_cifar10_class_names


def reshape_for_rnn(X_flat: np.ndarray) -> np.ndarray:
    """(N, 3072) -> (N, 32, 96)"""
    N = X_flat.shape[0]
    return X_flat.reshape(N, 32, 96).astype(np.float32)


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir: Path):
    """生成混淆矩阵可视化"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-6)
    
    # 1. 计数版混淆矩阵
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('RNN Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    path1 = save_dir / 'rnn_confusion_matrix_counts.png'
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"✓ Saved: {path1.name}")
    
    # 2. 归一化版混淆矩阵
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('RNN Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    path2 = save_dir / 'rnn_confusion_matrix_normalized.png'
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"✓ Saved: {path2.name}")
    
    return cm, cm_normalized


def plot_per_class_accuracy(cm, class_names, save_dir: Path):
    """生成每类准确率柱状图"""
    # 计算每类准确率（对角线 / 行和）
    per_class_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-6)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2ecc71' if acc >= 0.5 else '#e74c3c' for acc in per_class_acc]
    bars = ax.bar(class_names, per_class_acc, color=colors, edgecolor='black', linewidth=1.2)
    
    # 添加数值标签
    for bar, acc in zip(bars, per_class_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('RNN Per-Class Accuracy on CIFAR-10', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.axhline(y=np.mean(per_class_acc), color='blue', linestyle='--', 
               label=f'Mean: {np.mean(per_class_acc):.1%}')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    path = save_dir / 'rnn_per_class_accuracy.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"✓ Saved: {path.name}")
    
    return per_class_acc


def main():
    print("=" * 60)
    print("RNN Confusion Matrix Visualization")
    print("=" * 60)
    
    results_dir = current_dir / "cifar10_results"
    model_path = results_dir / "rnn_model.npy"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please run cifar10_experiment.py first!")
        return
    
    # 1. 加载数据
    print("\n[1/4] Loading CIFAR-10 data...")
    data_dir = (current_dir.parent.parent / "data" / "cifar-10-batches-py").resolve()
    
    # 使用更多测试样本以获得更准确的混淆矩阵
    _, _, _, _, X_test_raw, y_test = load_cifar10(
        data_dir=data_dir,
        train_samples=49000,
        val_samples=1000,
        test_samples=10000,  # 使用全部测试集
    )
    
    X_test_flat, _, _ = preprocess_data(X_test_raw, X_test_raw[:1], X_test_raw[:1])
    X_test = reshape_for_rnn(X_test_flat)
    print(f"   Test set: {X_test.shape}")
    
    # 2. 加载模型
    print("\n[2/4] Loading trained model...")
    model = RNNClassifier(input_dim=96, hidden_dim=256, output_dim=10)
    model.load_model(str(model_path))
    print("   Model loaded successfully!")
    
    # 3. 预测
    print("\n[3/4] Generating predictions...")
    y_pred = model.predict(X_test, batch_size=500)
    accuracy = np.mean(y_pred == y_test)
    print(f"   Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 4. 生成可视化
    print("\n[4/4] Generating visualizations...")
    class_names = get_cifar10_class_names()
    
    cm, cm_norm = plot_confusion_matrix(y_test, y_pred, class_names, results_dir)
    per_class_acc = plot_per_class_accuracy(cm, class_names, results_dir)
    
    # 打印分析
    print("\n" + "=" * 60)
    print("Per-Class Accuracy Analysis:")
    print("=" * 60)
    for name, acc in zip(class_names, per_class_acc):
        status = "✓" if acc >= 0.5 else "✗"
        print(f"  {status} {name:12s}: {acc:.1%}")
    print("-" * 60)
    print(f"  Mean Accuracy: {np.mean(per_class_acc):.1%}")
    print(f"  Best Class:  {class_names[np.argmax(per_class_acc)]} ({per_class_acc.max():.1%})")
    print(f"  Worst Class: {class_names[np.argmin(per_class_acc)]} ({per_class_acc.min():.1%})")
    print("=" * 60)
    
    print("\n✅ Done! Check results in:", results_dir)


if __name__ == "__main__":
    main()