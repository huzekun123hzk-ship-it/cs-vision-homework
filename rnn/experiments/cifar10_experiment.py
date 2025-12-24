# rnn/experiments/cifar10_experiment.py
"""
CIFAR-10 RNN 分类器实验主程序

流程：
1) 加载 CIFAR-10
2) preprocess（中心化+标准化）
3) reshape 为序列 (N, 32, 96)
4) 训练 Vanilla RNN
5) 保存曲线与可视化
"""

from __future__ import annotations
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# add rnn/ to sys.path to import rnn_classifier
current_dir = Path(__file__).resolve().parent          # rnn/experiments
sys.path.insert(0, str(current_dir.parent))            # rnn/

from rnn_classifier import RNNClassifier
from cifar10_utils import load_cifar10, preprocess_data, get_cifar10_class_names


CONFIG = {
    "data": {
        "train_samples": 49000,
        "val_samples": 1000,
        "test_samples": 1000,
        "data_dir": "../../data/cifar-10-batches-py",
    },
    "model": {
        "input_dim": 96,
        "hidden_dim": 256,
        "output_dim": 10,
        "reg_strength": 1e-4,
        "grad_clip": 5.0,
        "seed": 42,
    },
    "training": {
        "learning_rate": 2e-3,
        "num_epochs": 200,
        "batch_size": 128,
        "patience": 40,
        "print_every": 1,
        "lr_decay_step": 50,
        "lr_decay_rate": 0.95,
        "eval_subset": 5000,
    },
    "visualization": {
        "num_pred_samples": 20,
    }
}


def reshape_for_rnn(X_flat: np.ndarray) -> np.ndarray:
    """
    X_flat: (N,3072)
    return: (N,32,96)  row-by-row: T=32, D=32*3=96
    """
    N = X_flat.shape[0]
    assert X_flat.shape[1] == 3072
    return X_flat.reshape(N, 32, 96).astype(np.float32)


def plot_training_curves(history: dict, save_path: Path):
    epochs = history["epochs"]
    train_acc = history["train_acc_history"]
    val_acc = history["val_acc_history"]

    loss_epochs = sorted(history["loss_history"].keys())
    losses = [history["loss_history"][e] for e in loss_epochs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(loss_epochs, losses)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_acc, marker="o", label="Train Acc")
    ax2.plot(epochs, val_acc, marker="s", label="Val Acc")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"✓ Saved curves: {save_path}")


def visualize_predictions(X_seq, y_true, model, class_names, save_path: Path, num_samples: int = 20):
    rng = np.random.default_rng(0)
    N = X_seq.shape[0]
    idx = rng.choice(N, size=min(num_samples, N), replace=False)

    Xs = X_seq[idx]
    preds = model.predict(Xs, batch_size=1000)

    cols = 5
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i >= len(idx):
            ax.axis("off")
            continue

        img = Xs[i].reshape(32, 32, 3)
        # for display only: min-max normalize
        disp = (img - img.min()) / (img.max() - img.min() + 1e-8)

        ax.imshow(disp)
        t = class_names[int(y_true[idx[i]])]
        p = class_names[int(preds[i])]
        color = "green" if preds[i] == y_true[idx[i]] else "red"
        ax.set_title(f"T:{t}\nP:{p}", color=color, fontweight="bold")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"✓ Saved prediction viz: {save_path}")


def main():
    print("🚀 CIFAR-10 RNN Experiment")
    results_dir = current_dir / "cifar10_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) load
    data_dir = (current_dir.parent.parent / "data" / "cifar-10-batches-py").resolve()
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = load_cifar10(
        data_dir=data_dir,
        train_samples=CONFIG["data"]["train_samples"],
        val_samples=CONFIG["data"]["val_samples"],
        test_samples=CONFIG["data"]["test_samples"],
    )

    # 2) preprocess -> (N,3072)
    X_train_flat, X_val_flat, X_test_flat = preprocess_data(X_train_raw, X_val_raw, X_test_raw)

    # 3) reshape -> (N,32,96)
    X_train = reshape_for_rnn(X_train_flat)
    X_val = reshape_for_rnn(X_val_flat)
    X_test = reshape_for_rnn(X_test_flat)

    print(f"Train X: {X_train.shape}, y: {y_train.shape}")
    print(f"Val   X: {X_val.shape}, y: {y_val.shape}")
    print(f"Test  X: {X_test.shape}, y: {y_test.shape}")

    # 4) model
    model = RNNClassifier(
        input_dim=CONFIG["model"]["input_dim"],
        hidden_dim=CONFIG["model"]["hidden_dim"],
        output_dim=CONFIG["model"]["output_dim"],
        seed=CONFIG["model"]["seed"],
    )

    # 5) train
    history = model.train(
        X_train, y_train, X_val, y_val,
        learning_rate=CONFIG["training"]["learning_rate"],
        num_epochs=CONFIG["training"]["num_epochs"],
        batch_size=CONFIG["training"]["batch_size"],
        reg_strength=CONFIG["model"]["reg_strength"],
        grad_clip=CONFIG["model"]["grad_clip"],
        patience=CONFIG["training"]["patience"],
        print_every=CONFIG["training"]["print_every"],
        lr_decay_step=CONFIG["training"]["lr_decay_step"],
        lr_decay_rate=CONFIG["training"]["lr_decay_rate"],
        eval_subset=CONFIG["training"]["eval_subset"],
        verbose=True,
        seed=CONFIG["model"]["seed"],
    )

    # 6) evaluate
    test_acc = model.evaluate(X_test, y_test, batch_size=1000, num_samples=None, seed=0)
    print(f"🏆 Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # save
    model_path = results_dir / "rnn_model.npy"
    model.save_model(str(model_path))
    print(f"✓ Saved model: {model_path}")

    hist_path = results_dir / "history.npy"
    np.save(str(hist_path), history, allow_pickle=True)
    print(f"✓ Saved history: {hist_path}")

    curves_path = results_dir / "training_curves.png"
    plot_training_curves(history, curves_path)

    pred_path = results_dir / "prediction_viz.png"
    class_names = get_cifar10_class_names()
    visualize_predictions(X_test, y_test, model, class_names, pred_path,
                          num_samples=CONFIG["visualization"]["num_pred_samples"])

    # small text report
    report_path = results_dir / "rnn_experiment_results.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("CIFAR-10 Vanilla RNN Experiment\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
        f.write("CONFIG:\n")
        f.write(str(CONFIG) + "\n")
    print(f"✓ Saved report: {report_path}")

    print("✅ Done!")


if __name__ == "__main__":
    main()
