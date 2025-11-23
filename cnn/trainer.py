# cnn/trainer.py
import os
import time
import numpy as np


class Trainer(object):
    """
    一个简单的 NumPy 版训练器，支持:
        - mini-batch SGD
        - 学习率衰减
        - early stopping
        - 保存 best model 参数 (npz)
    """

    def __init__(
        self,
        model,
        data,
        update="sgd",
        learning_rate=1e-3,
        learning_rate_decay=0.95,
        reg=0.0,
        batch_size=128,
        num_epochs=20,
        verbose=True,
        print_every=100,
        early_stopping_patience=10,
        save_path=None,
    ):
        self.model = model
        self.X_train = data["X_train"]
        self.y_train = data["y_train"]
        self.X_val = data["X_val"]
        self.y_val = data["y_val"]

        self.update = update
        self.lr = learning_rate
        self.lr_decay = learning_rate_decay
        self.reg = reg
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.print_every = print_every
        self.early_stopping_patience = early_stopping_patience
        self.save_path = save_path

        # 记录历史
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

        # 动量缓存（如果用的话）
        self.velocity = {k: np.zeros_like(v) for k, v in model.params.items()}

    def _sgd_update(self, grads):
        for p_name in self.model.params:
            self.model.params[p_name] -= self.lr * grads[p_name]

    def _sgd_momentum_update(self, grads, mu=0.9):
        for p_name in self.model.params:
            v = self.velocity[p_name]
            v = mu * v - self.lr * grads[p_name]
            self.model.params[p_name] += v
            self.velocity[p_name] = v

    def _compute_accuracy(self, X, y, num_samples=1000, batch_size=100):
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples, replace=False)
            X = X[mask]
            y = y[mask]
            N = num_samples

        num_batches = int(np.ceil(N / batch_size))
        correct = 0
        total = 0
        for i in range(num_batches):
            X_batch = X[i * batch_size : (i + 1) * batch_size]
            y_batch = y[i * batch_size : (i + 1) * batch_size]
            scores = self.model.loss(X_batch)
            y_pred = np.argmax(scores, axis=1)
            correct += np.sum(y_pred == y_batch)
            total += y_batch.shape[0]
        return correct / total

    def train(self):
        num_train = self.X_train.shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        best_val_acc = 0.0
        best_params = None
        epochs_without_improve = 0

        if self.verbose:
            print(
                f"Starting training: {self.num_epochs} epochs, "
                f"{iterations_per_epoch} iters / epoch"
            )

        for epoch in range(self.num_epochs):
            # 一个 epoch 内随机打乱顺序
            idx = np.arange(num_train)
            np.random.shuffle(idx)
            X_train_shuff = self.X_train[idx]
            y_train_shuff = self.y_train[idx]

            epoch_loss = 0.0
            t0 = time.time()

            for it in range(iterations_per_epoch):
                X_batch = X_train_shuff[
                    it * self.batch_size : (it + 1) * self.batch_size
                ]
                y_batch = y_train_shuff[
                    it * self.batch_size : (it + 1) * self.batch_size
                ]

                loss, grads = self.model.loss(X_batch, y_batch)
                epoch_loss += loss

                if self.update == "sgd_momentum":
                    self._sgd_momentum_update(grads)
                else:
                    self._sgd_update(grads)

                if (
                    self.verbose
                    and self.print_every is not None
                    and (it + 1) % self.print_every == 0
                ):
                    print(
                        f"(epoch {epoch+1}/{self.num_epochs}, "
                        f"iter {it+1}/{iterations_per_epoch}) loss = {loss:.4f}"
                    )

            # 一个 epoch 结束，计算 train / val loss & acc
            epoch_loss /= iterations_per_epoch
            train_acc = self._compute_accuracy(self.X_train, self.y_train, num_samples=5000)
            val_loss, _ = self.model.loss(self.X_val, self.y_val)
            val_acc = self._compute_accuracy(self.X_val, self.y_val, num_samples=None)

            self.history["train_loss"].append(epoch_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(self.lr)

            if self.verbose:
                dt = time.time() - t0
                print(
                    f"Epoch [{epoch+1}/{self.num_epochs}] "
                    f"Train Loss: {epoch_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                    f"LR: {self.lr:.5f} | Time: {dt:.1f}s"
                )

            # 记录 best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_params = {k: v.copy() for k, v in self.model.params.items()}
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            # 学习率衰减
            self.lr *= self.lr_decay

            # early stopping
            if (
                self.early_stopping_patience is not None
                and epochs_without_improve >= self.early_stopping_patience
            ):
                if self.verbose:
                    print(
                        f"Early stopping at epoch {epoch+1}, "
                        f"best val acc = {best_val_acc:.4f}"
                    )
                break

        # 恢复 best model
        if best_params is not None:
            self.model.params = best_params

        # 保存参数
        if self.save_path is not None:
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            np.savez_compressed(self.save_path, **self.model.params)
            if self.verbose:
                print(f"Saved best model parameters to {self.save_path}")

        return self.history
