# rnn/gru_classifier.py
"""
GRU (Gated Recurrent Unit) 分类器 - CIFAR10 序列分类版

GRU核心特点：
1) 比LSTM更简洁：只有2个门（重置门、更新门）
2) 没有单独的cell state，直接更新hidden state
3) 参数更少，训练更快，性能接近LSTM
"""

from __future__ import annotations
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid激活函数"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def softmax(scores: np.ndarray) -> np.ndarray:
    """Softmax函数（数值稳定版）"""
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / (np.sum(exp_scores, axis=1, keepdims=True) + 1e-12)


def clip_by_global_norm(arrs, max_norm: float = 5.0) -> float:
    """全局范数梯度裁剪"""
    total = 0.0
    for a in arrs:
        total += float(np.sum(a * a))
    norm = float(np.sqrt(total))
    if max_norm is not None and max_norm > 0 and norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for a in arrs:
            a *= scale
    return norm


class GRUClassifier:
    """
    GRU many-to-one classifier:
      X: (N, T, D) -> GRU -> h_T -> scores: (N, K)
    
    GRU单元公式：
      z_t = sigmoid(W_xz @ x_t + W_hz @ h_{t-1} + b_z)  # 更新门
      r_t = sigmoid(W_xr @ x_t + W_hr @ h_{t-1} + b_r)  # 重置门
      h_tilde = tanh(W_xh @ x_t + W_hh @ (r_t * h_{t-1}) + b_h)  # 候选状态
      h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde  # 最终状态
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

        # Xavier初始化
        scale_x = 1.0 / np.sqrt(D)
        scale_h = 1.0 / np.sqrt(H)

        # 更新门 (update gate)
        self.W_xz = (rng.standard_normal((D, H)) * scale_x).astype(np.float32)
        self.W_hz = (rng.standard_normal((H, H)) * scale_h).astype(np.float32)
        self.b_z = np.zeros((H,), dtype=np.float32)

        # 重置门 (reset gate)
        self.W_xr = (rng.standard_normal((D, H)) * scale_x).astype(np.float32)
        self.W_hr = (rng.standard_normal((H, H)) * scale_h).astype(np.float32)
        self.b_r = np.zeros((H,), dtype=np.float32)

        # 候选hidden state
        self.W_xh = (rng.standard_normal((D, H)) * scale_x).astype(np.float32)
        self.W_hh = (rng.standard_normal((H, H)) * scale_h).astype(np.float32)
        self.b_h = np.zeros((H,), dtype=np.float32)

        # 输出层
        self.W_hq = (rng.standard_normal((H, K)) / np.sqrt(H)).astype(np.float32)
        self.b_q = np.zeros((K,), dtype=np.float32)

    def _forward(self, X: np.ndarray):
        """
        GRU前向传播
        X: (N, T, D)
        Returns:
          cache: dict with all intermediate values
          scores: (N, K)
        """
        N, T, D = X.shape
        H = self.hidden_dim

        # 存储所有时间步的状态
        h_states = np.zeros((N, T + 1, H), dtype=np.float32)
        
        # 缓存门控值和候选状态
        z_gates = np.zeros((N, T, H), dtype=np.float32)
        r_gates = np.zeros((N, T, H), dtype=np.float32)
        h_tilde = np.zeros((N, T, H), dtype=np.float32)

        for t in range(T):
            x_t = X[:, t, :]                    # (N, D)
            h_prev = h_states[:, t, :]          # (N, H)

            # 更新门
            z_t = sigmoid(x_t @ self.W_xz + h_prev @ self.W_hz + self.b_z)
            
            # 重置门
            r_t = sigmoid(x_t @ self.W_xr + h_prev @ self.W_hr + self.b_r)
            
            # 候选hidden state (使用重置后的h_prev)
            h_tilde_t = np.tanh(x_t @ self.W_xh + (r_t * h_prev) @ self.W_hh + self.b_h)
            
            # 最终hidden state (线性插值)
            h_next = (1.0 - z_t) * h_prev + z_t * h_tilde_t

            # 存储
            h_states[:, t + 1, :] = h_next
            z_gates[:, t, :] = z_t
            r_gates[:, t, :] = r_t
            h_tilde[:, t, :] = h_tilde_t

        # 输出层
        h_last = h_states[:, T, :]
        scores = h_last @ self.W_hq + self.b_q

        cache = {
            'X': X,
            'h_states': h_states,
            'z_gates': z_gates,
            'r_gates': r_gates,
            'h_tilde': h_tilde,
        }

        return cache, scores

    def _loss_and_grads(
        self,
        X: np.ndarray,
        y: np.ndarray,
        reg_strength: float = 1e-4,
        grad_clip: float = 5.0,
    ):
        """
        计算损失和梯度
        """
        N, T, D = X.shape

        # Forward
        cache, scores = self._forward(X)
        probs = softmax(scores)

        # Loss
        correct_logprobs = -np.log(probs[np.arange(N), y] + 1e-12)
        data_loss = float(np.mean(correct_logprobs))
        
        reg_loss = 0.5 * reg_strength * (
            float(np.sum(self.W_xz**2)) + float(np.sum(self.W_hz**2)) +
            float(np.sum(self.W_xr**2)) + float(np.sum(self.W_hr**2)) +
            float(np.sum(self.W_xh**2)) + float(np.sum(self.W_hh**2)) +
            float(np.sum(self.W_hq**2))
        )
        loss = data_loss + reg_loss

        # Backward
        dscores = probs
        dscores[np.arange(N), y] -= 1.0
        dscores /= N

        # 初始化梯度
        dW_xz = np.zeros_like(self.W_xz)
        dW_hz = np.zeros_like(self.W_hz)
        db_z = np.zeros_like(self.b_z)
        dW_xr = np.zeros_like(self.W_xr)
        dW_hr = np.zeros_like(self.W_hr)
        db_r = np.zeros_like(self.b_r)
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        dW_hq = np.zeros_like(self.W_hq)
        db_q = np.zeros_like(self.b_q)

        # 输出层梯度
        h_last = cache['h_states'][:, T, :]
        dW_hq = h_last.T @ dscores
        db_q = np.sum(dscores, axis=0)
        
        dh_next = dscores @ self.W_hq.T

        # BPTT through GRU
        for t in reversed(range(T)):
            h_prev = cache['h_states'][:, t, :]
            h_curr = cache['h_states'][:, t + 1, :]
            
            z_t = cache['z_gates'][:, t, :]
            r_t = cache['r_gates'][:, t, :]
            h_tilde_t = cache['h_tilde'][:, t, :]
            
            x_t = X[:, t, :]

            # dh_curr -> dh_tilde (候选状态)
            dh_tilde = dh_next * z_t
            dh_tilde_raw = dh_tilde * (1.0 - h_tilde_t**2)  # tanh导数
            
            dW_xh += x_t.T @ dh_tilde_raw
            dW_hh += (r_t * h_prev).T @ dh_tilde_raw
            db_h += np.sum(dh_tilde_raw, axis=0)

            # dh_curr -> dz (更新门)
            dz_t = dh_next * (h_tilde_t - h_prev)
            dz_raw = dz_t * z_t * (1.0 - z_t)  # sigmoid导数
            
            dW_xz += x_t.T @ dz_raw
            dW_hz += h_prev.T @ dz_raw
            db_z += np.sum(dz_raw, axis=0)

            # dh_curr -> dr (重置门，通过h_tilde)
            dr_t = (dh_tilde_raw @ self.W_hh.T) * h_prev
            dr_raw = dr_t * r_t * (1.0 - r_t)  # sigmoid导数
            
            dW_xr += x_t.T @ dr_raw
            dW_hr += h_prev.T @ dr_raw
            db_r += np.sum(dr_raw, axis=0)

            # 传播到前一个时间步
            dh_from_z = dz_raw @ self.W_hz.T
            dh_from_r = dr_raw @ self.W_hr.T
            dh_from_tilde = (dh_tilde_raw @ self.W_hh.T) * r_t
            dh_from_direct = dh_next * (1.0 - z_t)
            
            dh_next = dh_from_z + dh_from_r + dh_from_tilde + dh_from_direct

        # 正则化梯度
        dW_xz += reg_strength * self.W_xz
        dW_hz += reg_strength * self.W_hz
        dW_xr += reg_strength * self.W_xr
        dW_hr += reg_strength * self.W_hr
        dW_xh += reg_strength * self.W_xh
        dW_hh += reg_strength * self.W_hh
        dW_hq += reg_strength * self.W_hq

        # 梯度裁剪
        grad_list = [dW_xz, dW_hz, db_z, dW_xr, dW_hr, db_r,
                    dW_xh, dW_hh, db_h, dW_hq, db_q]
        grad_norm = clip_by_global_norm(grad_list, max_norm=grad_clip)

        grads = {
            'W_xz': dW_xz, 'W_hz': dW_hz, 'b_z': db_z,
            'W_xr': dW_xr, 'W_hr': dW_hr, 'b_r': db_r,
            'W_xh': dW_xh, 'W_hh': dW_hh, 'b_h': db_h,
            'W_hq': dW_hq, 'b_q': db_q,
        }

        return loss, grads, grad_norm

    def train(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        learning_rate: float = 1e-3,
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
        """训练GRU"""
        rng = np.random.default_rng(seed)
        N = X_train.shape[0]
        iters_per_epoch = max(N // batch_size, 1)

        history = {
            "loss_history": {},
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

                # SGD更新
                for key in grads:
                    param = getattr(self, key)
                    param -= lr * grads[key]

            avg_loss = epoch_loss / iters_per_epoch
            avg_gn = epoch_gn / iters_per_epoch
            history["loss_history"][epoch] = float(avg_loss)
            history["grad_norm_history"].append(float(avg_gn))
            history["lr_history"].append(float(lr))

            if (epoch % print_every == 0) or (epoch == num_epochs - 1):
                train_acc = self.evaluate(X_train, y_train, batch_size=1000, 
                                         num_samples=eval_subset, seed=seed + epoch)
                val_acc = self.evaluate(X_val, y_val, batch_size=1000, 
                                       num_samples=None, seed=seed + 1000 + epoch)

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
                            print(f"\nEarly stopping! Best ValAcc={best_val_acc:.4f}")
                        break

            if lr_decay_step is not None and (epoch + 1) % lr_decay_step == 0:
                lr *= float(lr_decay_rate)

        if best_params is not None:
            self._set_params(best_params)
            if verbose:
                print(f"Restored best model. Best ValAcc={best_val_acc:.4f}")

        return history

    def predict(self, X: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        N = X.shape[0]
        preds = []
        for i in range(0, N, batch_size):
            _, scores = self._forward(X[i:i+batch_size])
            preds.append(np.argmax(scores, axis=1))
        return np.concatenate(preds, axis=0)

    def evaluate(self, X: np.ndarray, y: np.ndarray, 
                batch_size: int = 1000, num_samples: int | None = None, 
                seed: int = 0) -> float:
        if num_samples is not None and X.shape[0] > num_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X.shape[0], size=num_samples, replace=False)
            X = X[idx]
            y = y[idx]
        y_pred = self.predict(X, batch_size=batch_size)
        return float(np.mean(y_pred == y))

    def _get_params_copy(self):
        return {
            'W_xz': self.W_xz.copy(), 'W_hz': self.W_hz.copy(), 'b_z': self.b_z.copy(),
            'W_xr': self.W_xr.copy(), 'W_hr': self.W_hr.copy(), 'b_r': self.b_r.copy(),
            'W_xh': self.W_xh.copy(), 'W_hh': self.W_hh.copy(), 'b_h': self.b_h.copy(),
            'W_hq': self.W_hq.copy(), 'b_q': self.b_q.copy(),
        }

    def _set_params(self, params: dict):
        for key, val in params.items():
            setattr(self, key, val.copy())

    def save_model(self, path: str):
        np.save(path, self._get_params_copy())

    def load_model(self, path: str):
        params = np.load(path, allow_pickle=True).item()
        self._set_params(params)