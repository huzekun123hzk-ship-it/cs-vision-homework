#!/usr/bin/env python3
"""
两层全连接神经网络 - CIFAR-10 分类实验

完整的实验流程：
1. 加载CIFAR-10数据集
2. 数据预处理
3. 超参数搜索（可选）
4. 训练两层神经网络（带学习率衰减）
5. 评估模型性能
6. 生成专业可视化结果

改进点：
- He权重初始化，收敛更快
- 学习率衰减调度
- Dropout正则化
- 详细的性能分析
- 专业的结果呈现
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
import time
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns

from two_layer_network.two_layer_network import TwoLayerNetwork
from .cifar10_utils import load_cifar10, preprocess_data, get_cifar10_class_names


# ============================================================================
# 实验配置 - 在这里修改所有超参数
# ============================================================================
CONFIG = {
    # 数据集配置
    'data': {
        'train_samples': 40000,
        'val_samples': 5000,
        'test_samples': 5000,
    },
    
    # 模型配置
    'model': {
        'weight_scale': None,         # None使用He初始化
        'dropout': 0.5,               # Dropout比率
    },
    
    # 训练配置
    'training': {
        'learning_rate_init': 1e-2,   # 初始学习率
        'num_epochs': 400,            # 训练轮数
        'batch_size': 256,            # 批次大小
        'patience': 80,               # 早停耐心值
        'print_every': 20,            # 打印间隔
        'lr_decay_epochs': 50,        # 学习率衰减间隔
        'lr_decay_factor': 0.95,      # 学习率衰减因子
    },
    
    # 超参数搜索配置
    'hyperparam_search': {
        'do_search': False,           # 是否执行搜索
        'hidden_sizes': [100, 150, 200, 250],
        'regularizations': [1e-3, 5e-3, 1e-2, 5e-2],
        'learning_rates': [5e-3, 1e-2, 2e-2, 5e-2],
        'search_epochs': 200,         # 每个组合训练轮数
    },
    
    # 默认超参数（不搜索时使用）
    'model_params': {
        'hidden_size': 200,
        'reg': 5e-3,
    },
}
# ============================================================================


def print_config():
    """打印实验配置"""
    print("\n" + "=" * 80)
    print(" " * 25 + "CIFAR-10 实验配置")
    print("=" * 80)
    
    print("\n📊 数据集配置:")
    for key, value in CONFIG['data'].items():
        print(f"   • {key:25s}: {value:,}")
    
    print("\n🧠 模型配置:")
    for key, value in CONFIG['model'].items():
        print(f"   • {key:25s}: {value}")
    
    print("\n🎯 训练配置:")
    for key, value in CONFIG['training'].items():
        print(f"   • {key:25s}: {value}")
    
    if CONFIG['hyperparam_search']['do_search']:
        print("\n🔍 超参数搜索:")
        for key, value in CONFIG['hyperparam_search'].items():
            if key != 'do_search':
                print(f"   • {key:25s}: {value}")
    else:
        print("\n🔍 使用默认模型参数:")
        for key, value in CONFIG['model_params'].items():
            print(f"   • {key:25s}: {value}")
    
    print("\n" + "=" * 80 + "\n")

def cross_validation(X_train, y_train, X_val, y_val):
    """
    超参数搜索 - 使用验证集评估
    
    参数：
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
    
    返回：
        results: 所有超参数组合的结果
        best_params: 最优超参数
        best_val_acc: 最优验证准确率
    """
    cfg = CONFIG['hyperparam_search']
    
    hidden_sizes = cfg['hidden_sizes']
    regularizations = cfg['regularizations']
    learning_rates = cfg['learning_rates']
    num_epochs = cfg['search_epochs']
    
    results = {}
    best_val_acc = 0
    best_params = None
    
    print("\n" + "=" * 80)
    print("🔍 开始超参数搜索")
    print("=" * 80)
    print(f"隐藏层大小: {hidden_sizes}")
    print(f"正则化系数: {regularizations}")
    print(f"学习率: {learning_rates}")
    print(f"每个组合训练轮数: {num_epochs}")
    print(f"总搜索组合数: {len(hidden_sizes) * len(regularizations) * len(learning_rates)}")
    print("=" * 80)
    
    total = len(hidden_sizes) * len(regularizations) * len(learning_rates)
    current = 0
    
    for h in hidden_sizes:
        for reg in regularizations:
            for lr in learning_rates:
                current += 1
                
                # 创建网络
                net = TwoLayerNetwork(
                    input_size=X_train.shape[1],
                    hidden_size=h,
                    num_classes=10,
                    weight_scale=CONFIG['model']['weight_scale'],
                    reg=reg,
                    dropout=CONFIG['model']['dropout']
                )
                
                # 训练
                print(f"\n[{current}/{total}] H={h:3d}, λ={reg:.0e}, lr={lr:.0e}", end='')
                
                best_epoch_acc = 0
                for epoch in range(num_epochs):
                    # 小批量训练
                    indices = np.random.choice(len(X_train), 
                                              min(256, len(X_train)), 
                                              replace=False)
                    net.train_step(X_train[indices], y_train[indices], lr)
                    
                    if (epoch + 1) % max(1, num_epochs // 5) == 0:
                        val_acc = net.evaluate(X_val, y_val)
                        print(f" → Epoch {epoch+1}: val_acc={val_acc:.4f}", end='')
                        best_epoch_acc = max(best_epoch_acc, val_acc)
                
                results[(h, reg, lr)] = best_epoch_acc
                
                if best_epoch_acc > best_val_acc:
                    best_val_acc = best_epoch_acc
                    best_params = {
                        'hidden_size': h,
                        'reg': reg,
                        'learning_rate': lr
                    }
                    print(" ✓ 新最优！")
    
    print("\n" + "=" * 80)
    print(f"✅ 超参数搜索完成！")
    print(f"最优参数: {best_params}")
    print(f"最优验证准确率: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print("=" * 80)
    
    return results, best_params, best_val_acc
def train_with_decay(net, X_train, y_train, X_val, y_val, best_params=None):
    """
    带学习率衰减的训练
    
    参数：
        net: 神经网络模型
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        best_params: 最优超参数（包含learning_rate）
    
    返回：
        history: 训练历史
        best_epoch: 最优轮数
        best_model_params: 最优模型参数
    """
    cfg = CONFIG['training']
    
    if best_params is not None:
        lr_init = best_params.get('learning_rate', cfg['learning_rate_init'])
    else:
        lr_init = cfg['learning_rate_init']
    
    num_epochs = cfg['num_epochs']
    batch_size = cfg['batch_size']
    patience = cfg['patience']
    print_every = cfg['print_every']
    lr_decay_epochs = cfg['lr_decay_epochs']
    lr_decay_factor = cfg['lr_decay_factor']
    
    history = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rates': [],
    }
    
    best_val_acc = 0
    best_epoch = 0
    best_model_params = None
    epochs_no_improve = 0
    lr = lr_init
    
    num_batches = len(X_train) // batch_size
    
    print("\n" + "=" * 80)
    print("🎯 开始训练")
    print("=" * 80)
    print(f"初始学习率: {lr_init:.0e}")
    print(f"总轮数: {num_epochs}")
    print(f"批次大小: {batch_size}")
    print(f"早停耐心值: {patience}")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # 学习率衰减
        if epoch > 0 and epoch % lr_decay_epochs == 0:
            lr *= lr_decay_factor
        
        # 训练一个epoch
        epoch_loss = 0
        indices = np.random.permutation(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            loss = net.train_step(X_batch, y_batch, lr)
            epoch_loss += loss
        
        epoch_loss /= num_batches
        
        # 计算训练和验证精度
        train_acc = net.evaluate(X_train, y_train)
        val_acc = net.evaluate(X_val, y_val)
        
        # 记录历史
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(lr)
        
        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_params = {k: v.copy() for k, v in net.params.items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # 打印进度
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"LR: {lr:.0e}")
        
        # 早停
        if epochs_no_improve >= patience:
            print(f"\n⏹️  早停！在第 {epoch+1} 轮达到最优验证准确率: {best_val_acc:.4f}")
            break
    
    elapsed_time = time.time() - start_time
    
    print("=" * 80)
    print(f"✅ 训练完成！")
    print(f"    总耗时: {elapsed_time:.2f} 秒")
    print(f"    最优轮数: {best_epoch}")
    print(f"    最优验证准确率: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print("=" * 80)
    
    # 恢复最优模型参数
    if best_model_params is not None:
        net.params = best_model_params
    
    return history, best_epoch, best_model_params, elapsed_time


def plot_training_curves(history, results_dir):
    """
    绘制训练曲线（损失、准确率、学习率）
    
    参数：
        history: 训练历史字典
        results_dir: 结果保存目录
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = history['epochs']
    
    # 1. 损失曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2.5, 
                 label='Training Loss', marker='o', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training Loss', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(fontsize=11)
    
    # 2. 准确率曲线
    axes[1].plot(epochs, history['train_acc'], 'b-', linewidth=2.5,
                 label='Training Accuracy', marker='o', markersize=4)
    axes[1].plot(epochs, history['val_acc'], 'r-', linewidth=2.5,
                 label='Validation Accuracy', marker='s', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(fontsize=11, loc='lower right')
    axes[1].set_ylim([0, 1])
    
    # 3. 学习率变化
    axes[2].plot(epochs, history['learning_rates'], 'g-', linewidth=2.5, marker='D', markersize=4)
    axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('Learning Rate', fontsize=12, fontweight='bold')
    axes[2].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_yscale('log')
    
    plt.suptitle('Training Curves - Two Layer Network', fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    save_path = results_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 训练曲线已保存: {save_path.name}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, results_dir):
    """
    绘制混淆矩阵
    
    参数：
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        results_dir: 结果保存目录
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绘制原始混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Raw Count)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    # 绘制归一化混淆矩阵
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens', ax=ax2,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    
    plt.suptitle('Confusion Matrix Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    save_path = results_dir / 'confusion_matrix.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 混淆矩阵已保存: {save_path.name}")
    plt.close()
    
    # 计算每类准确率
    class_accuracies = cm_normalized.diagonal()
    
    return class_accuracies


def plot_per_class_accuracy(class_accuracies, class_names, results_dir):
    """
    绘制每类准确率柱状图
    
    参数：
        class_accuracies: 每类准确率数组
        class_names: 类别名称列表
        results_dir: 结果保存目录
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.RdYlGn(class_accuracies)
    bars = ax.bar(range(len(class_names)), class_accuracies, color=colors, 
                  edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1%}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 添加平均线
    mean_acc = np.mean(class_accuracies)
    ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_acc:.1%}')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    save_path = results_dir / 'per_class_accuracy.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 每类准确率已保存: {save_path.name}")
    plt.close()


def visualize_predictions(X_test, y_test, net, class_names, results_dir, num_samples=20):
    """
    可视化预测结果（正确和错误）
    
    参数：
        X_test: 测试数据
        y_test: 测试标签
        net: 训练好的网络
        class_names: 类别名称列表
        results_dir: 结果保存目录
        num_samples: 可视化样本数
    """
    predictions = net.predict(X_test)
    
    # 分离正确和错误样本
    correct_mask = predictions == y_test
    incorrect_mask = ~correct_mask
    
    correct_indices = np.where(correct_mask)[0]
    incorrect_indices = np.where(incorrect_mask)[0]
    
    # 采样
    num_correct = min(10, len(correct_indices))
    num_incorrect = min(10, len(incorrect_indices))
    
    correct_samples = np.random.choice(correct_indices, num_correct, replace=False)
    incorrect_samples = np.random.choice(incorrect_indices, num_incorrect, replace=False)
    
    fig, axes = plt.subplots(2, 10, figsize=(16, 4))
    
    # 绘制正确预测
    for idx, sample_idx in enumerate(correct_samples):
        ax = axes[0, idx]
        img = X_test[sample_idx].reshape(32, 32, 3)
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        true_label = class_names[y_test[sample_idx]]
        ax.set_title(f'{true_label}', color='green', fontweight='bold', fontsize=9)
        ax.axis('off')
    
    # 绘制错误预测
    for idx, sample_idx in enumerate(incorrect_samples):
        ax = axes[1, idx]
        img = X_test[sample_idx].reshape(32, 32, 3)
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        true_label = class_names[y_test[sample_idx]]
        pred_label = class_names[predictions[sample_idx]]
        title = f'True: {true_label}\nPred: {pred_label}'
        ax.set_title(title, color='red', fontweight='bold', fontsize=8)
        ax.axis('off')
    
    axes[0, 0].text(-0.5, 0.5, 'Correct Predictions', transform=axes[0, 0].transAxes,
                     fontsize=11, fontweight='bold', rotation=90, va='center')
    axes[1, 0].text(-0.5, 0.5, 'Wrong Predictions', transform=axes[1, 0].transAxes,
                     fontsize=11, fontweight='bold', rotation=90, va='center')
    
    plt.suptitle('Sample Predictions (  Green=Correct, Red=Wrong)', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    save_path = results_dir / 'predictions_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 预测可视化已保存: {save_path.name}")
    plt.close()


def visualize_learned_features(net, class_names, results_dir):
    """
    可视化学习到的隐藏层特征权重
    
    参数：
        net: 训练好的网络
        class_names: 类别名称列表
        results_dir: 结果保存目录
    """
    W1 = net.params['W1']
    
    # 选择最有代表性的32个神经元（基于权重范数）
    weight_norms = np.linalg.norm(W1, axis=0)
    top_indices = np.argsort(weight_norms)[-32:][::-1]
    
    fig, axes = plt.subplots(4, 8, figsize=(14, 7))
    axes = axes.flatten()
    
    for idx, neuron_idx in enumerate(top_indices):
        w = W1[:, neuron_idx].reshape(32, 32, 3)
        
        # 归一化到[0, 1]
        w_normalized = (w - w.min()) / (w.max() - w.min() + 1e-5)
        w_normalized = np.clip(w_normalized, 0, 1)
        
        axes[idx].imshow(w_normalized)
        axes[idx].set_title(f'Neuron {neuron_idx}', fontsize=9, fontweight='bold')
        axes[idx].axis('off')
    
    plt.suptitle('Learned Hidden Layer Features (Top 32 by Norm)', 
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    save_path = results_dir / 'learned_features.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 学习特征可视化已保存: {save_path.name}")
    plt.close()


def save_results(net, history, train_acc, val_acc, test_acc, 
                 class_accuracies, class_names, elapsed_time, results_dir):
    """
    保存所有实验结果
    
    参数：
        net: 训练好的网络
        history: 训练历史
        train_acc, val_acc, test_acc: 准确率
        class_accuracies: 每类准确率
        class_names: 类别名称
        elapsed_time: 训练耗时
        results_dir: 结果保存目录
    """
    # 1. 保存模型
    model_path = results_dir / 'best_model.pkl'
    net.save_model(model_path)
    print(f"✓ 模型已保存: {model_path.name}")
    
    # 2. 保存训练历史
    history_path = results_dir / 'training_history.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"✓ 训练历史已保存: {history_path.name}")
    
    # 3. 生成详细报告
    report_path = results_dir / 'experiment_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(" " * 20 + "CIFAR-10 实验报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 模型配置
        f.write("【模型配置】\n")
        f.write(f"  输入维度: {net.input_size}\n")
        f.write(f"  隐藏层维度: {net.hidden_size}\n")
        f.write(f"  输出维度: {net.num_classes}\n")
        f.write(f"  正则化系数: {net.reg}\n")
        f.write(f"  Dropout: {net.dropout}\n")
        f.write(f"  总参数量: {net.input_size * net.hidden_size + net.hidden_size * net.num_classes:,}\n\n")
        
        # 训练配置
        f.write("【训练配置】\n")
        f.write(f"  训练样本数: {CONFIG['data']['train_samples']:,}\n")
        f.write(f"  验证样本数: {CONFIG['data']['val_samples']:,}\n")
        f.write(f"  测试样本数: {CONFIG['data']['test_samples']:,}\n")
        f.write(f"  初始学习率: {CONFIG['training']['learning_rate_init']}\n")
        f.write(f"  批次大小: {CONFIG['training']['batch_size']}\n")
        f.write(f"  总训练轮数: {len(history['epochs'])}\n")
        f.write(f"  训练耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)\n\n")
        
        # 性能指标
        f.write("【性能指标】\n")
        f.write(f"  训练集准确率: {train_acc:.4f} ({train_acc*100:.2f}%)\n")
        f.write(f"  验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)\n")
        f.write(f"  测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)\n\n")
        
        # 每类准确率
        f.write("【每类准确率】\n")
        for class_name, acc in zip(class_names, class_accuracies):
            f.write(f"  {class_name:12s}: {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"  平均准确率: {np.mean(class_accuracies):.4f} ({np.mean(class_accuracies)*100:.2f}%)\n\n")
        
        # 最优和最差类别
        best_class_idx = np.argmax(class_accuracies)
        worst_class_idx = np.argmin(class_accuracies)
        f.write(f"  最优类别: {class_names[best_class_idx]} ({class_accuracies[best_class_idx]*100:.2f}%)\n")
        f.write(f"  最差类别: {class_names[worst_class_idx]} ({class_accuracies[worst_class_idx]*100:.2f}%)\n\n")
        
        # 训练历史摘要
        f.write("【训练历史】\n")
        f.write(f"  最终训练损失: {history['train_loss'][-1]:.4f}\n")
        f.write(f"  最小训练损失: {min(history['train_loss']):.4f}\n")
        f.write(f"  最高验证准确率: {max(history['val_acc']):.4f} ({max(history['val_acc'])*100:.2f}%)\n")
        f.write(f"  最终学习率: {history['learning_rates'][-1]:.0e}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("实验完成时间: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ 实验报告已保存: {report_path.name}")
    
    # 4. 打印摘要到控制台
    print("\n" + "=" * 80)
    print(" " * 28 + "实验结果摘要")
    print("=" * 80)
    print(f"训练集准确率: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"训练耗时: {elapsed_time:.2f} 秒")
    print(f"参数数量: {net.input_size * net.hidden_size + net.hidden_size * net.num_classes:,}")
    print("=" * 80 + "\n")


def main():
    """主实验流程"""
    
    # 打印配置
    print_config()
    
    # 创建结果目录
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    print(f"📁 结果将保存到: {results_dir.resolve()}\n")
    
    # ========================================================================
    # 1. 加载数据
    # ========================================================================
    print("=" * 80)
    print("📊 加载 CIFAR-10 数据集")
    print("=" * 80)
    
    data_dir = Path('data/cifar-10-batches-py')
    
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请确保CIFAR-10数据集在正确位置")
        return
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(
        data_dir,
        train_samples=CONFIG['data']['train_samples'],
        val_samples=CONFIG['data']['val_samples'],
        test_samples=CONFIG['data']['test_samples']
    )
    
    print(f"✓ 训练集: {X_train.shape}")
    print(f"✓ 验证集: {X_val.shape}")
    print(f"✓ 测试集: {X_test.shape}")
    
    # ========================================================================
    # 2. 数据预处理
    # ========================================================================
    print("\n" + "=" * 80)
    print("🔧 数据预处理（归一化 + 中心化 + 偏置）")
    print("=" * 80)
    
    X_train, X_val, X_test = preprocess_data(X_train, X_val, X_test)
    
    print(f"✓ 预处理后训练集: {X_train.shape}")
    print(f"✓ 预处理后验证集: {X_val.shape}")
    print(f"✓ 预处理后测试集: {X_test.shape}")
    
    class_names = get_cifar10_class_names()
    
    # ========================================================================
    # 3. 超参数搜索（可选）
    # ========================================================================
    best_params = None
    
    if CONFIG['hyperparam_search']['do_search']:
        results, best_params, best_val_acc = cross_validation(
            X_train, y_train, X_val, y_val
        )
        
        # 使用搜索到的最优参数
        hidden_size = best_params['hidden_size']
        reg = best_params['reg']
    else:
        # 使用默认参数
        hidden_size = CONFIG['model_params']['hidden_size']
        reg = CONFIG['model_params']['reg']
        best_params = {
            'hidden_size': hidden_size,
            'reg': reg,
            'learning_rate': CONFIG['training']['learning_rate_init']
        }
    
    # ========================================================================
    # 4. 创建并训练最终模型
    # ========================================================================
    print("\n" + "=" * 80)
    print("🧠 创建最终模型")
    print("=" * 80)
    
    net = TwoLayerNetwork(
        input_size=X_train.shape[1],
        hidden_size=hidden_size,
        num_classes=10,
        weight_scale=CONFIG['model']['weight_scale'],
        reg=reg,
        dropout=CONFIG['model']['dropout']
    )
    
    print(f"✓ 输入维度: {net.input_size}")
    print(f"✓ 隐藏层维度: {net.hidden_size}")
    print(f"✓ 输出维度: {net.num_classes}")
    print(f"✓ 正则化系数: {net.reg}")
    print(f"✓ Dropout: {net.dropout}")
    print(f"✓ 总参数量: {net.input_size * net.hidden_size + net.hidden_size * net.num_classes:,}")
    
    # 训练
    history, best_epoch, best_model_params, elapsed_time = train_with_decay(
        net, X_train, y_train, X_val, y_val, best_params
    )
    
    # ========================================================================
    # 5. 评估模型
    # ========================================================================
    print("\n" + "=" * 80)
    print("📈 评估模型性能")
    print("=" * 80)
    
    train_acc = net.evaluate(X_train, y_train)
    val_acc = net.evaluate(X_val, y_val)
    test_acc = net.evaluate(X_test, y_test)
    
    print(f"✓ 训练集准确率: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"✓ 验证集准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"✓ 测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # ========================================================================
    # 6. 生成可视化
    # ========================================================================
    print("\n" + "=" * 80)
    print("🎨 生成可视化结果")
    print("=" * 80)
    
    # 训练曲线
    plot_training_curves(history, results_dir)
    
    # 混淆矩阵
    y_test_pred = net.predict(X_test)
    class_accuracies = plot_confusion_matrix(y_test, y_test_pred, class_names, results_dir)
    
    # 每类准确率
    plot_per_class_accuracy(class_accuracies, class_names, results_dir)
    
    # 预测可视化
    visualize_predictions(X_test, y_test, net, class_names, results_dir)
    
    # 学习特征可视化
    visualize_learned_features(net, class_names, results_dir)
    
    # ========================================================================
    # 7. 保存结果
    # ========================================================================
    print("\n" + "=" * 80)
    print("💾 保存实验结果")
    print("=" * 80)
    
    save_results(net, history, train_acc, val_acc, test_acc,
                 class_accuracies, class_names, elapsed_time, results_dir)
    
    print("\n🎉 实验全部完成！所有结果已保存到 results/ 目录")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()