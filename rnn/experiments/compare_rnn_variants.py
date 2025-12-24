# rnn/experiments/compare_rnn_variants.py
"""
RNN变体对比实验：Vanilla RNN vs GRU vs LSTM

全面对比三种架构在CIFAR-10上的表现
"""

from __future__ import annotations
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))

from rnn_classifier import RNNClassifier
from gru_classifier import GRUClassifier
from lstm_classifier import LSTMClassifier
from cifar10_utils import load_cifar10, preprocess_data


def reshape_for_rnn(X_flat: np.ndarray) -> np.ndarray:
    """(N, 3072) -> (N, 32, 96)"""
    return X_flat.reshape(X_flat.shape[0], 32, 96).astype(np.float32)


def load_history(path: Path):
    """加载训练历史"""
    if not path.exists():
        return None
    return np.load(str(path), allow_pickle=True).item()


def plot_comparison(histories: dict, save_dir: Path):
    """
    生成对比图表
    histories: {"Vanilla RNN": history1, "GRU": history2, "LSTM": history3}
    """
    colors = {
        "Vanilla RNN": "#e74c3c",
        "GRU": "#3498db",
        "LSTM": "#2ecc71",
    }
    
    # 1. Loss对比
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, hist in histories.items():
        if hist is None:
            continue
        loss_epochs = sorted(hist["loss_history"].keys())
        losses = [hist["loss_history"][e] for e in loss_epochs]
        ax.plot(loss_epochs, losses, label=name, linewidth=2, color=colors.get(name, 'gray'))
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / 'comparison_loss.png', dpi=150)
    plt.close(fig)
    print("✓ Saved: comparison_loss.png")
    
    # 2. Validation Accuracy对比
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, hist in histories.items():
        if hist is None:
            continue
        epochs = hist["epochs"]
        val_acc = hist["val_acc_history"]
        ax.plot(epochs, val_acc, label=name, linewidth=2.5, 
                marker='o', markersize=4, color=colors.get(name, 'gray'))
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy', fontsize=12)
    ax.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.3, 0.6])
    fig.tight_layout()
    fig.savefig(save_dir / 'comparison_val_acc.png', dpi=150)
    plt.close(fig)
    print("✓ Saved: comparison_val_acc.png")
    
    # 3. Gradient Norm对比
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, hist in histories.items():
        if hist is None:
            continue
        grad_norms = hist.get("grad_norm_history", [])
        if len(grad_norms) > 0:
            ax.plot(grad_norms, label=name, linewidth=1.5, alpha=0.7, color=colors.get(name, 'gray'))
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Norm Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / 'comparison_grad_norm.png', dpi=150)
    plt.close(fig)
    print("✓ Saved: comparison_grad_norm.png")


def plot_final_results(results: dict, save_dir: Path):
    """
    绘制最终测试准确率柱状图
    results: {"Vanilla RNN": 0.493, "GRU": 0.51, "LSTM": 0.52}
    """
    names = list(results.keys())
    accs = [results[n] * 100 for n in names]
    colors_list = ["#e74c3c", "#3498db", "#2ecc71"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, accs, color=colors_list, edgecolor='black', linewidth=1.5, width=0.6)
    
    # 添加数值标签
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('CIFAR-10 Test Accuracy: RNN Variants Comparison', 
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, 60])
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_dir / 'comparison_final_results.png', dpi=150)
    plt.close(fig)
    print("✓ Saved: comparison_final_results.png")


def generate_report(results: dict, histories: dict, save_path: Path):
    """生成文字报告"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RNN Variants Comparison Report - CIFAR-10\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. Final Test Accuracy\n")
        f.write("-" * 70 + "\n")
        for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
            f.write(f"   {name:15s}: {acc:.4f} ({acc*100:.2f}%)\n")
        
        f.write("\n2. Training Summary\n")
        f.write("-" * 70 + "\n")
        for name, hist in histories.items():
            if hist is None:
                f.write(f"\n{name}: No data available\n")
                continue
            
            best_val = max(hist["val_acc_history"])
            final_val = hist["val_acc_history"][-1] if len(hist["val_acc_history"]) > 0 else 0
            total_epochs = len(hist["loss_history"])
            
            f.write(f"\n{name}:\n")
            f.write(f"   Total Epochs:       {total_epochs}\n")
            f.write(f"   Best Val Acc:       {best_val:.4f}\n")
            f.write(f"   Final Val Acc:      {final_val:.4f}\n")
            f.write(f"   Test Acc:           {results.get(name, 0):.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Key Findings:\n")
        f.write("-" * 70 + "\n")
        
        best_model = max(results.items(), key=lambda x: x[1])[0]
        worst_model = min(results.items(), key=lambda x: x[1])[0]
        
        f.write(f"- Best Model:  {best_model} ({results[best_model]*100:.2f}%)\n")
        f.write(f"- Worst Model: {worst_model} ({results[worst_model]*100:.2f}%)\n")
        f.write(f"- Improvement: {(results[best_model] - results[worst_model])*100:.2f}%\n")
        
        f.write("\n")
        f.write("Conclusions:\n")
        f.write("- GRU和LSTM通过门控机制缓解了梯度消失问题\n")
        f.write("- LSTM的cell state提供了更强的长期记忆能力\n")
        f.write("- 但在CIFAR-10这种图像任务上，改进幅度有限\n")
        f.write("- 原因：row-by-row序列化破坏了2D空间结构\n")
        f.write("=" * 70 + "\n")
    
    print(f"✓ Saved: {save_path.name}")


def main():
    print("\n" + "=" * 70)
    print("📊 RNN Variants Comparison Experiment")
    print("=" * 70 + "\n")
    
    results_dir = current_dir / "comparison_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集已有结果
    print("[1/3] Collecting experiment results...")
    
    rnn_hist = load_history(current_dir / "cifar10_results" / "history.npy")
    gru_hist = load_history(current_dir / "gru_results" / "gru_history.npy")
    lstm_hist = load_history(current_dir / "lstm_results" / "lstm_history.npy")
    
    histories = {
        "Vanilla RNN": rnn_hist,
        "GRU": gru_hist,
        "LSTM": lstm_hist,
    }
    
    # 检查哪些实验已完成
    available = {name: hist is not None for name, hist in histories.items()}
    print(f"   Vanilla RNN: {'✓' if available['Vanilla RNN'] else '✗'}")
    print(f"   GRU:         {'✓' if available['GRU'] else '✗'}")
    print(f"   LSTM:        {'✓' if available['LSTM'] else '✗'}")
    
    if not any(available.values()):
        print("\n❌ No experiment results found!")
        print("Please run the following experiments first:")
        print("   - python -m rnn.experiments.cifar10_experiment")
        print("   - python -m rnn.experiments.gru_experiment")
        print("   - python -m rnn.experiments.lstm_experiment")
        return
    
    # 加载测试集准确率
    print("\n[2/3] Loading test accuracies...")
    results = {}
    
    # RNN
    rnn_report = current_dir / "cifar10_results" / "rnn_experiment_results.txt"
    if rnn_report.exists():
        with open(rnn_report, 'r') as f:
            for line in f:
                if "Test Accuracy:" in line:
                    acc = float(line.split(":")[1].strip())
                    results["Vanilla RNN"] = acc
                    print(f"   Vanilla RNN: {acc:.4f}")
                    break
    
    # GRU
    gru_report = current_dir / "gru_results" / "gru_experiment_results.txt"
    if gru_report.exists():
        with open(gru_report, 'r') as f:
            for line in f:
                if "Test Accuracy:" in line:
                    acc = float(line.split(":")[1].strip())
                    results["GRU"] = acc
                    print(f"   GRU:         {acc:.4f}")
                    break
    
    # LSTM
    lstm_report = current_dir / "lstm_results" / "lstm_experiment_results.txt"
    if lstm_report.exists():
        with open(lstm_report, 'r') as f:
            for line in f:
                if "Test Accuracy:" in line:
                    acc = float(line.split(":")[1].strip())
                    results["LSTM"] = acc
                    print(f"   LSTM:        {acc:.4f}")
                    break
    
    if len(results) == 0:
        print("\n❌ Could not load any test results!")
        return
    
    # 生成对比图表
    print("\n[3/3] Generating comparison visualizations...")
    plot_comparison(histories, results_dir)
    
    if len(results) > 0:
        plot_final_results(results, results_dir)
    
    # 生成报告
    report_path = results_dir / "comparison_report.txt"
    generate_report(results, histories, report_path)
    
    print("\n" + "=" * 70)
    print("✅ Comparison Complete!")
    print(f"📁 Results saved to: {results_dir}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()