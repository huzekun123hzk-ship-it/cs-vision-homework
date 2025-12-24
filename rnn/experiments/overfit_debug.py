# rnn/experiments/overfit_debug.py
"""
Overfit Debug: 用很小的数据集验证实现正确性

思路：
- 取 200~500 张训练样本
- 多跑一些 epoch
- 如果 train acc 能明显升高（比如 >0.6 甚至更高），说明 forward/backward 基本没大问题
"""

from __future__ import annotations
import sys
import numpy as np
from pathlib import Path

current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))
sys.path.insert(0, str(current_dir))

from rnn_classifier import RNNClassifier
from cifar10_utils import load_cifar10, preprocess_data


def reshape_for_rnn(X_flat: np.ndarray) -> np.ndarray:
    N = X_flat.shape[0]
    return X_flat.reshape(N, 32, 96).astype(np.float32)


def main():
    data_dir = (current_dir.parent.parent / "data" / "cifar-10-batches-py").resolve()

    # small split
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = load_cifar10(
        data_dir=data_dir,
        train_samples=500,   # small train
        val_samples=200,
        test_samples=200,
    )

    X_train_flat, X_val_flat, X_test_flat = preprocess_data(X_train_raw, X_val_raw, X_test_raw)
    X_train = reshape_for_rnn(X_train_flat)
    X_val = reshape_for_rnn(X_val_flat)
    X_test = reshape_for_rnn(X_test_flat)

    model = RNNClassifier(input_dim=96, hidden_dim=128, output_dim=10, seed=0)

    history = model.train(
        X_train, y_train, X_val, y_val,
        learning_rate=5e-3,
        num_epochs=300,
        batch_size=50,
        reg_strength=0.0,      # overfit debug: turn off reg
        grad_clip=5.0,
        patience=200,
        print_every=5,
        lr_decay_step=100,
        lr_decay_rate=0.95,
        eval_subset=None,      # compute exact train acc (small anyway)
        verbose=True,
        seed=0,
    )

    train_acc = model.evaluate(X_train, y_train, batch_size=200, num_samples=None, seed=0)
    val_acc = model.evaluate(X_val, y_val, batch_size=200, num_samples=None, seed=0)
    test_acc = model.evaluate(X_test, y_test, batch_size=200, num_samples=None, seed=0)

    print("\n===== OVERFIT DEBUG RESULT =====")
    print(f"Train Acc: {train_acc:.4f}")
    print(f"Val   Acc: {val_acc:.4f}")
    print(f"Test  Acc: {test_acc:.4f}")
    print("================================\n")


if __name__ == "__main__":
    main()
