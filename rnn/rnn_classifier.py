# rnn/rnn_classifier.py
"""
Vanilla RNN (循环神经网络) 分类器 - CIFAR10 序列分类版

核心改进：
1) W_hh 使用 I + noise 初始化，稳定时间传播
2) 使用 Global Norm Gradient Clipping（比逐元素 clip 更合理）
3) 提供 train / predict / evaluate / save / load
"""

from __future__ import annotations
import numpy as np


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-12)


def clip_by_global_norm(arrs, max_norm: float = 5.0) -> float:
    """In-place global norm clipping. Returns original norm."""
    total = 0.0
    for a in arrs:
        total += float(np.sum(a * a))
    norm = float(np.sqrt(total))
    if max_norm is not None and max_norm > 0 and norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for a in arrs:
            a *= scale
    return norm


class RNNClassifier:
    """
    many-to-one RNN classifier:
      X: (N, T, D) -> RNN -> h_T -> scores: (N, K)
    """

    def __init__(
        self,
        input_dim: int = 96,
        hidden_dim: int = 256,
        output_dim: int = 10,
        seed: int = 42,
    ):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)

        rng = np.random.default_rng(seed)
        D, H, K = self.input_dim, self.hidden_dim, self.output_dim

        # Xavier for input/output projections
        self.W_xh = (rng.standard_normal((D, H)) / np.sqrt(D)).astype(np.float32)
        self.W_hq = (rng.standard_normal((H, K)) / np.sqrt(H)).astype(np.float32)

        # Key: stable recurrent init
        self.W_hh = (np.eye(H, dtype=np.float32) + 0.01 * rng.standard_normal((H, H)).astype(np.float32))

        self.b_h = np.zeros((H,), dtype=np.float32)
        self.b_q = np.zeros((K,), dtype=np.float32)

    def _forward(self, X: np.ndarray):
        """
        X: (N, T, D)
        Returns:
          h_states: (N, T+1, H) with h_states[:,0,:]=0
          scores: (N, K)
        """
        N, T, D = X.shape
        H = self.hidden_dim
        assert D == self.input_dim, f"Expected D={self.input_dim}, got {D}"

        h_states = np.zeros((N, T + 1, H), dtype=np.float32)

        for t in range(T):
            x_t = X[:, t, :]                 # (N,D)
            h_prev = h_states[:, t, :]       # (N,H)

            a = x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h  # (N,H)
            h_states[:, t + 1, :] = np.tanh(a)

        h_last = h_states[:, T, :]  # (N,H)
        scores = h_last @ self.W_hq + self.b_q  # (N,K)
        return h_states, scores

    def _loss_and_grads(
        self,
        X: np.ndarray,
        y: np.ndarray,
        reg_strength: float = 1e-4,
        grad_clip: float = 5.0,
    ):
        """
        Returns:
          loss: float
          grads: dict
          grad_norm: float (before clipping)
        """
        N, T, D = X.shape

        # forward
        h_states, scores = self._forward(X)
        probs = softmax(scores)

        # loss
        correct_logprobs = -np.log(probs[np.arange(N), y] + 1e-12)
        data_loss = float(np.mean(correct_logprobs))
        reg_loss = 0.5 * reg_strength * (
            float(np.sum(self.W_xh * self.W_xh)) +
            float(np.sum(self.W_hh * self.W_hh)) +
            float(np.sum(self.W_hq * self.W_hq))
        )
        loss = data_loss + reg_loss

        # backward: scores
        dscores = probs
        dscores[np.arange(N), y] -= 1.0
        dscores /= N  # (N,K)

        # grads init
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hq = np.zeros_like(self.W_hq)
        db_h = np.zeros_like(self.b_h)
        db_q = np.zeros_like(self.b_q)

        # output layer grads
        h_last = h_states[:, T, :]              # (N,H)
        dW_hq = h_last.T @ dscores              # (H,K)
        db_q = np.sum(dscores, axis=0)          # (K,)
        dh = dscores @ self.W_hq.T              # (N,H) gradient flowing into h_T

        # BPTT through time
        for t in reversed(range(T)):
            h_curr = h_states[:, t + 1, :]      # (N,H)
            h_prev = h_states[:, t, :]          # (N,H)
            x_t = X[:, t, :]                    # (N,D)

            # tanh' = 1 - h^2
            da = dh * (1.0 - h_curr * h_curr)   # (N,H)

            dW_xh += x_t.T @ da                 # (D,H)
            dW_hh += h_prev.T @ da              # (H,H)
            db_h += np.sum(da, axis=0)          # (H,)

            dh = da @ self.W_hh.T               # (N,H)

        # reg grads
        dW_xh += reg_strength * self.W_xh
        dW_hh += reg_strength * self.W_hh
        dW_hq += reg_strength * self.W_hq

        # clip by global norm
        grad_norm = clip_by_global_norm([dW_xh, dW_hh, dW_hq, db_h, db_q], max_norm=grad_clip)

        grads = {
            "W_xh": dW_xh, "W_hh": dW_hh, "W_hq": dW_hq,
            "b_h": db_h, "b_q": db_q
        }
        return loss, grads, grad_norm

    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        learning_rate: float = 2e-3,
        num_epochs: int = 200,
        batch_size: int = 128,
        reg_strength: float = 1e-4,
        grad_clip: float = 5.0,
        patience: int = 40,
        print_every: int = 10,
        lr_decay_step: int = 50,
        lr_decay_rate: float = 0.95,
        verbose: bool = True,
        eval_subset: int | None = 5000,
        seed: int = 42,
    ):
        """
        Mini-batch SGD + early stopping + step LR decay
        """
        rng = np.random.default_rng(seed)
        N = X_train.shape[0]
        iters_per_epoch = max(N // batch_size, 1)

        history = {
            "loss_history": {},          # epoch -> avg_loss
            "train_acc_history": [],
            "val_acc_history": [],
            "grad_norm_history": [],
            "epochs": [],
            "lr_history": [],
        }

        best_val_acc = 0.0
        best_params = None
        epochs_no_improve = 0
        lr = float(learning_rate)

        for epoch in range(num_epochs):
            # shuffle
            idx = np.arange(N)
            rng.shuffle(idx)
            Xs = X_train[idx]
            ys = y_train[idx]

            epoch_loss = 0.0
            epoch_gn = 0.0

            for it in range(iters_per_epoch):
                start = it * batch_size
                end = min(start + batch_size, N)
                Xb = Xs[start:end]
                yb = ys[start:end]

                loss, grads, gn = self._loss_and_grads(Xb, yb, reg_strength, grad_clip)
                epoch_loss += loss
                epoch_gn += gn

                # SGD update
                self.W_xh -= lr * grads["W_xh"]
                self.W_hh -= lr * grads["W_hh"]
                self.W_hq -= lr * grads["W_hq"]
                self.b_h  -= lr * grads["b_h"]
                self.b_q  -= lr * grads["b_q"]

            avg_loss = epoch_loss / iters_per_epoch
            avg_gn = epoch_gn / iters_per_epoch
            history["loss_history"][epoch] = float(avg_loss)
            history["grad_norm_history"].append(float(avg_gn))
            history["lr_history"].append(float(lr))

            # evaluate periodically
            if (epoch % print_every == 0) or (epoch == num_epochs - 1):
                train_acc = self.evaluate(X_train, y_train, batch_size=1000, num_samples=eval_subset, seed=seed + epoch)
                val_acc = self.evaluate(X_val, y_val, batch_size=1000, num_samples=None, seed=seed + 1000 + epoch)

                history["train_acc_history"].append(float(train_acc))
                history["val_acc_history"].append(float(val_acc))
                history["epochs"].append(int(epoch))

                if verbose:
                    print(
                        f"Epoch {epoch:3d}/{num_epochs} | "
                        f"Loss={avg_loss:.4f} | GradNorm={avg_gn:.2f} | "
                        f"TrainAcc={train_acc:.4f} | ValAcc={val_acc:.4f} | LR={lr:.6f}"
                    )

                if val_acc > best_val_acc:
                    best_val_acc = float(val_acc)
                    best_params = self._get_params_copy()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += print_every
                    if epochs_no_improve >= patience:
                        if verbose:
                            print(f"\nEarly stopping! No improvement for {epochs_no_improve} epochs. Best ValAcc={best_val_acc:.4f}")
                        break

            # step lr decay
            if lr_decay_step is not None and lr_decay_step > 0 and (epoch + 1) % lr_decay_step == 0:
                lr *= float(lr_decay_rate)

        # restore best
        if best_params is not None:
            self._set_params(best_params)
            if verbose:
                print(f"Training finished. Restored best model. Best ValAcc={best_val_acc:.4f}")

        return history

    def predict(self, X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        N = X.shape[0]
        preds = []
        for i in range(0, N, batch_size):
            _, scores = self._forward(X[i:i+batch_size])
            preds.append(np.argmax(scores, axis=1))
        return np.concatenate(preds, axis=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray, batch_size: int = 1000, num_samples: int | None = None, seed: int = 0) -> float:
        if num_samples is not None and X.shape[0] > num_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X.shape[0], size=num_samples, replace=False)
            X = X[idx]
            y = y[idx]
        y_pred = self.predict(X, batch_size=batch_size)
        return float(np.mean(y_pred == y))

    def _get_params_copy(self):
        return {
            "W_xh": self.W_xh.copy(),
            "W_hh": self.W_hh.copy(),
            "W_hq": self.W_hq.copy(),
            "b_h": self.b_h.copy(),
            "b_q": self.b_q.copy(),
        }

    def _set_params(self, params: dict):
        self.W_xh = params["W_xh"].copy()
        self.W_hh = params["W_hh"].copy()
        self.W_hq = params["W_hq"].copy()
        self.b_h = params["b_h"].copy()
        self.b_q = params["b_q"].copy()

    def save_model(self, path: str):
        params = self._get_params_copy()
        np.save(path, params)

    def load_model(self, path: str):
        params = np.load(path, allow_pickle=True).item()
        self._set_params(params)
