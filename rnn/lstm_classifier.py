# rnn/lstm_classifier.py
"""
LSTM (Long Short-Term Memory) 分类器 - CIFAR10 序列分类版

LSTM核心改进：
1) Cell State (c_t) 作为长期记忆
2) 三个门控机制：输入门、遗忘门、输出门
3) 缓解梯度消失问题，更适合长序列
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


class LSTMClassifier:
    """
    LSTM many-to-one classifier:
      X: (N, T, D) -> LSTM -> h_T -> scores: (N, K)
    
    LSTM单元公式：
      f_t = sigmoid(W_xf @ x_t + W_hf @ h_{t-1} + b_f)  # 遗忘门
      i_t = sigmoid(W_xi @ x_t + W_hi @ h_{t-1} + b_i)  # 输入门
      g_t = tanh(W_xg @ x_t + W_hg @ h_{t-1} + b_g)     # 候选cell
      o_t = sigmoid(W_xo @ x_t + W_ho @ h_{t-1} + b_o)  # 输出门
      c_t = f_t * c_{t-1} + i_t * g_t                    # 更新cell state
      h_t = o_t * tanh(c_t)                               # 输出hidden state
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

        # Xavier初始化 - LSTM有4组权重（f, i, g, o）
        scale_x = 1.0 / np.sqrt(D)
        scale_h = 1.0 / np.sqrt(H)

        # 遗忘门 (forget gate)
        self.W_xf = (rng.standard_normal((D, H)) * scale_x).astype(np.float32)
        self.W_hf = (rng.standard_normal((H, H)) * scale_h).astype(np.float32)
        self.b_f = np.ones((H,), dtype=np.float32)  # 初始化为1（倾向于记住）

        # 输入门 (input gate)
        self.W_xi = (rng.standard_normal((D, H)) * scale_x).astype(np.float32)
        self.W_hi = (rng.standard_normal((H, H)) * scale_h).astype(np.float32)
        self.b_i = np.zeros((H,), dtype=np.float32)

        # 候选cell (cell candidate)
        self.W_xg = (rng.standard_normal((D, H)) * scale_x).astype(np.float32)
        self.W_hg = (rng.standard_normal((H, H)) * scale_h).astype(np.float32)
        self.b_g = np.zeros((H,), dtype=np.float32)

        # 输出门 (output gate)
        self.W_xo = (rng.standard_normal((D, H)) * scale_x).astype(np.float32)
        self.W_ho = (rng.standard_normal((H, H)) * scale_h).astype(np.float32)
        self.b_o = np.zeros((H,), dtype=np.float32)

        # 输出层
        self.W_hq = (rng.standard_normal((H, K)) / np.sqrt(H)).astype(np.float32)
        self.b_q = np.zeros((K,), dtype=np.float32)

    def _forward(self, X: np.ndarray):
        """
        LSTM前向传播
        X: (N, T, D)
        Returns:
          cache: dict with all intermediate values
          scores: (N, K)
        """
        N, T, D = X.shape
        H = self.hidden_dim

        # 存储所有时间步的状态
        h_states = np.zeros((N, T + 1, H), dtype=np.float32)
        c_states = np.zeros((N, T + 1, H), dtype=np.float32)
        
        # 缓存门控值用于反向传播
        f_gates = np.zeros((N, T, H), dtype=np.float32)
        i_gates = np.zeros((N, T, H), dtype=np.float32)
        g_gates = np.zeros((N, T, H), dtype=np.float32)
        o_gates = np.zeros((N, T, H), dtype=np.float32)
        c_tanh = np.zeros((N, T, H), dtype=np.float32)

        for t in range(T):
            x_t = X[:, t, :]                    # (N, D)
            h_prev = h_states[:, t, :]          # (N, H)
            c_prev = c_states[:, t, :]          # (N, H)

            # 遗忘门
            f_t = sigmoid(x_t @ self.W_xf + h_prev @ self.W_hf + self.b_f)
            # 输入门
            i_t = sigmoid(x_t @ self.W_xi + h_prev @ self.W_hi + self.b_i)
            # 候选cell
            g_t = np.tanh(x_t @ self.W_xg + h_prev @ self.W_hg + self.b_g)
            # 输出门
            o_t = sigmoid(x_t @ self.W_xo + h_prev @ self.W_ho + self.b_o)

            # 更新cell state
            c_next = f_t * c_prev + i_t * g_t
            # 计算hidden state
            c_tanh_t = np.tanh(c_next)
            h_next = o_t * c_tanh_t

            # 存储
            h_states[:, t + 1, :] = h_next
            c_states[:, t + 1, :] = c_next
            f_gates[:, t, :] = f_t
            i_gates[:, t, :] = i_t
            g_gates[:, t, :] = g_t
            o_gates[:, t, :] = o_t
            c_tanh[:, t, :] = c_tanh_t

        # 输出层
        h_last = h_states[:, T, :]
        scores = h_last @ self.W_hq + self.b_q

        cache = {
            'X': X,
            'h_states': h_states,
            'c_states': c_states,
            'f_gates': f_gates,
            'i_gates': i_gates,
            'g_gates': g_gates,
            'o_gates': o_gates,
            'c_tanh': c_tanh,
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
            float(np.sum(self.W_xf**2)) + float(np.sum(self.W_hf**2)) +
            float(np.sum(self.W_xi**2)) + float(np.sum(self.W_hi**2)) +
            float(np.sum(self.W_xg**2)) + float(np.sum(self.W_hg**2)) +
            float(np.sum(self.W_xo**2)) + float(np.sum(self.W_ho**2)) +
            float(np.sum(self.W_hq**2))
        )
        loss = data_loss + reg_loss

        # Backward
        dscores = probs
        dscores[np.arange(N), y] -= 1.0
        dscores /= N

        # 初始化梯度
        dW_xf = np.zeros_like(self.W_xf)
        dW_hf = np.zeros_like(self.W_hf)
        db_f = np.zeros_like(self.b_f)
        dW_xi = np.zeros_like(self.W_xi)
        dW_hi = np.zeros_like(self.W_hi)
        db_i = np.zeros_like(self.b_i)
        dW_xg = np.zeros_like(self.W_xg)
        dW_hg = np.zeros_like(self.W_hg)
        db_g = np.zeros_like(self.b_g)
        dW_xo = np.zeros_like(self.W_xo)
        dW_ho = np.zeros_like(self.W_ho)
        db_o = np.zeros_like(self.b_o)
        dW_hq = np.zeros_like(self.W_hq)
        db_q = np.zeros_like(self.b_q)

        # 输出层梯度
        h_last = cache['h_states'][:, T, :]
        dW_hq = h_last.T @ dscores
        db_q = np.sum(dscores, axis=0)
        
        dh_next = dscores @ self.W_hq.T
        dc_next = np.zeros((N, self.hidden_dim), dtype=np.float32)

        # BPTT through LSTM
        for t in reversed(range(T)):
            h_prev = cache['h_states'][:, t, :]
            c_prev = cache['c_states'][:, t, :]
            c_curr = cache['c_states'][:, t + 1, :]
            
            f_t = cache['f_gates'][:, t, :]
            i_t = cache['i_gates'][:, t, :]
            g_t = cache['g_gates'][:, t, :]
            o_t = cache['o_gates'][:, t, :]
            c_tanh_t = cache['c_tanh'][:, t, :]
            
            x_t = X[:, t, :]

            # dh -> do (输出门)
            do_t = dh_next * c_tanh_t
            do_raw = do_t * o_t * (1.0 - o_t)  # sigmoid导数
            
            dW_xo += x_t.T @ do_raw
            dW_ho += h_prev.T @ do_raw
            db_o += np.sum(do_raw, axis=0)

            # dh -> dc
            dc_next += dh_next * o_t * (1.0 - c_tanh_t**2)  # tanh导数

            # dc -> df (遗忘门)
            df_t = dc_next * c_prev
            df_raw = df_t * f_t * (1.0 - f_t)
            
            dW_xf += x_t.T @ df_raw
            dW_hf += h_prev.T @ df_raw
            db_f += np.sum(df_raw, axis=0)

            # dc -> di (输入门)
            di_t = dc_next * g_t
            di_raw = di_t * i_t * (1.0 - i_t)
            
            dW_xi += x_t.T @ di_raw
            dW_hi += h_prev.T @ di_raw
            db_i += np.sum(di_raw, axis=0)

            # dc -> dg (候选cell)
            dg_t = dc_next * i_t
            dg_raw = dg_t * (1.0 - g_t**2)  # tanh导数
            
            dW_xg += x_t.T @ dg_raw
            dW_hg += h_prev.T @ dg_raw
            db_g += np.sum(dg_raw, axis=0)

            # 传播到前一个时间步
            dh_next = (do_raw @ self.W_ho.T + df_raw @ self.W_hf.T + 
                      di_raw @ self.W_hi.T + dg_raw @ self.W_hg.T)
            dc_next = dc_next * f_t

        # 正则化梯度
        dW_xf += reg_strength * self.W_xf
        dW_hf += reg_strength * self.W_hf
        dW_xi += reg_strength * self.W_xi
        dW_hi += reg_strength * self.W_hi
        dW_xg += reg_strength * self.W_xg
        dW_hg += reg_strength * self.W_hg
        dW_xo += reg_strength * self.W_xo
        dW_ho += reg_strength * self.W_ho
        dW_hq += reg_strength * self.W_hq

        # 梯度裁剪
        grad_list = [dW_xf, dW_hf, db_f, dW_xi, dW_hi, db_i,
                    dW_xg, dW_hg, db_g, dW_xo, dW_ho, db_o,
                    dW_hq, db_q]
        grad_norm = clip_by_global_norm(grad_list, max_norm=grad_clip)

        grads = {
            'W_xf': dW_xf, 'W_hf': dW_hf, 'b_f': db_f,
            'W_xi': dW_xi, 'W_hi': dW_hi, 'b_i': db_i,
            'W_xg': dW_xg, 'W_hg': dW_hg, 'b_g': db_g,
            'W_xo': dW_xo, 'W_ho': dW_ho, 'b_o': db_o,
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
        """训练LSTM"""
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
            'W_xf': self.W_xf.copy(), 'W_hf': self.W_hf.copy(), 'b_f': self.b_f.copy(),
            'W_xi': self.W_xi.copy(), 'W_hi': self.W_hi.copy(), 'b_i': self.b_i.copy(),
            'W_xg': self.W_xg.copy(), 'W_hg': self.W_hg.copy(), 'b_g': self.b_g.copy(),
            'W_xo': self.W_xo.copy(), 'W_ho': self.W_ho.copy(), 'b_o': self.b_o.copy(),
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