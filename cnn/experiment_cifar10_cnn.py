# cnn/experiment_cifar10_cnn.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from .model import Cifar10SimpleConvNet
from .trainer import Trainer
from .data_utils import preprocess_cifar10_data


def plot_training_curves(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Loss 曲线
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss_curves.png"))
    plt.close()

    # Acc 曲线
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Training / Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "acc_curves.png"))
    plt.close()


def evaluate_on_test(model, X_test, y_test, batch_size=100):
    N = X_test.shape[0]
    num_batches = int(np.ceil(N / batch_size))
    correct = 0
    for i in range(num_batches):
        X_batch = X_test[i * batch_size : (i + 1) * batch_size]
        y_batch = y_test[i * batch_size : (i + 1) * batch_size]
        scores = model.loss(X_batch)
        y_pred = np.argmax(scores, axis=1)
        correct += np.sum(y_pred == y_batch)
    return correct / N


def main():
    parser = argparse.ArgumentParser(description="NumPy CNN on CIFAR-10")
    parser.add_argument("--data-dir", type=str, default="./data/cifar-10-batches-py")
    parser.add_argument("--results-dir", type=str, default="./cnn/experiments/results")
    parser.add_argument("--num-train", type=int, default=49000)
    parser.add_argument("--num-val", type=int, default=1000)
    parser.add_argument("--num-test", type=int, default=10000)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--reg", type=float, default=1e-3)
    parser.add_argument("--num-filters", type=int, default=32)
    parser.add_argument("--filter-size", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--weight-scale", type=float, default=5e-3)
    parser.add_argument("--update", type=str, default="sgd_momentum",
                        choices=["sgd", "sgd_momentum"])
    parser.add_argument("--early-stop", type=int, default=10)

    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    save_path = os.path.join(args.results_dir, "cnn_cifar10_best.npz")

    # ----- 加载 & 预处理数据 -----
    data = preprocess_cifar10_data(
        cifar10_dir=args.data_dir,
        num_training=args.num_train,
        num_validation=args.num_val,
        num_test=args.num_test,
    )

    # ----- 构建模型 -----
    model = Cifar10SimpleConvNet(
        input_dim=(3, 32, 32),
        num_filters=args.num_filters,
        filter_size=args.filter_size,
        hidden_dim=args.hidden_dim,
        num_classes=10,
        weight_scale=args.weight_scale,
        reg=args.reg,
    )

    # ----- 训练 -----
    trainer = Trainer(
        model=model,
        data=data,
        update=args.update,
        learning_rate=args.learning_rate,
        learning_rate_decay=args.lr_decay,
        reg=args.reg,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        verbose=True,
        print_every=50,
        early_stopping_patience=args.early_stop,
        save_path=save_path,
    )

    history = trainer.train()
    plot_training_curves(history, args.results_dir)

    # ----- 在 test set 上评估 -----
    test_acc = evaluate_on_test(model, data["X_test"], data["y_test"])
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
