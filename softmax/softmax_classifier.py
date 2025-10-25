"""
Softmax线性分类器

实现多类别softmax分类器，包含：
- Softmax损失函数计算
- 梯度下降优化
- L2正则化
- 早停机制
"""

import numpy as np


class SoftmaxClassifier:
    """
    Softmax线性分类器
    
    使用交叉熵损失 + L2正则化训练多类别分类器
    
    Attributes:
        num_features: 输入特征维度（包含偏置项）
        num_classes: 类别数量
        reg_strength: L2正则化强度
        W: 权重矩阵 [D, K]，D是特征维度，K是类别数
    """
    
    def __init__(self, num_features, num_classes, reg_strength=1e-4):
        """
        初始化分类器
        
        Args:
            num_features: 特征维度（包含偏置）
            num_classes: 类别数
            reg_strength: L2正则化强度
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.reg_strength = reg_strength
        
        # 初始化权重矩阵（小随机数）
        self.W = np.random.randn(num_features, num_classes) * 0.0001
    
    def _compute_loss_and_gradient(self, X, y):
        """
        计算损失和梯度（向量化实现）
        
        Args:
            X: 输入数据 [N, D]
            y: 标签 [N]
            
        Returns:
            (loss, gradient): 损失值和梯度 [D, K]
        """
        N = X.shape[0]
        
        # 1. 前向传播：计算分数
        scores = X.dot(self.W)  # [N, K]
        
        # 2. 数值稳定的Softmax
        scores_max = np.max(scores, axis=1, keepdims=True)
        scores_stable = scores - scores_max
        
        # 计算softmax概率
        exp_scores = np.exp(scores_stable)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # 3. 计算交叉熵损失
        correct_class_probs = probs[np.arange(N), y]
        data_loss = -np.mean(np.log(correct_class_probs + 1e-10))
        
        # 4. 添加L2正则化（只对权重正则化，不包括偏置）
        reg_loss = 0.5 * self.reg_strength * np.sum(self.W[:-1, :] ** 2)
        
        total_loss = data_loss + reg_loss
        
        # 5. 反向传播：计算梯度
        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N
        
        # 权重梯度
        dW = X.T.dot(dscores)
        
        # 添加正则化梯度（只对权重，不包括偏置）
        dW[:-1, :] += self.reg_strength * self.W[:-1, :]
        
        return total_loss, dW
    
    def train(self, X_train, y_train, X_val, y_val,
              learning_rate=0.05, num_epochs=500, batch_size=500,
              patience=100, verbose=True, print_every=10):
        """
        训练分类器（Mini-batch梯度下降 + 早停）
        
        Args:
            X_train: 训练数据 [N_train, D]
            y_train: 训练标签 [N_train]
            X_val: 验证数据 [N_val, D]
            y_val: 验证标签 [N_val]
            learning_rate: 初始学习率
            num_epochs: 最大训练轮数
            batch_size: 批次大小
            patience: 早停耐心值
            verbose: 是否打印训练信息
            print_every: 打印间隔
            
        Returns:
            history: 训练历史字典
        """
        N = X_train.shape[0]
        iterations_per_epoch = max(N // batch_size, 1)
        
        # 训练历史
        history = {
            'loss_history': {},
            'train_acc_history': [],
            'val_acc_history': [],
            'epochs': []
        }
        
        # 早停相关
        best_val_acc = 0.0
        best_W = None
        epochs_without_improvement = 0
        
        # 学习率衰减
        current_lr = learning_rate
        
        for epoch in range(num_epochs):
            # 随机打乱数据
            indices = np.random.permutation(N)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Mini-batch训练
            epoch_losses = []
            for i in range(iterations_per_epoch):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, N)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # 计算损失和梯度
                loss, grad = self._compute_loss_and_gradient(X_batch, y_batch)
                epoch_losses.append(loss)
                
                # 更新权重
                self.W -= current_lr * grad
            
            # 记录平均损失
            avg_loss = np.mean(epoch_losses)
            history['loss_history'][epoch] = avg_loss
            
            # 每隔一定epoch评估
            if epoch % print_every == 0 or epoch == num_epochs - 1:
                train_acc = self.evaluate(X_train, y_train)
                val_acc = self.evaluate(X_val, y_val)
                
                history['train_acc_history'].append(train_acc)
                history['val_acc_history'].append(val_acc)
                history['epochs'].append(epoch)
                
                if verbose:
                    print(f"Epoch {epoch:4d}/{num_epochs}: "
                          f"Loss={avg_loss:.4f}, "
                          f"Train={train_acc:.4f}, "
                          f"Val={val_acc:.4f}, "
                          f"LR={current_lr:.6f}")
                
                # 早停检查
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_W = self.W.copy()
                    epochs_without_improvement = 0
                    if verbose:
                        print("  ✨ 新的最佳验证准确率！")
                else:
                    epochs_without_improvement += print_every
                
                # 触发早停
                if epochs_without_improvement >= patience:
                    if verbose:
                        print(f"\n  早停触发！已{epochs_without_improvement}个epoch未改善")
                        print(f"   最佳验证准确率: {best_val_acc:.4f}")
                    break
            
            # 学习率衰减（每50个epoch）
            if (epoch + 1) % 50 == 0:
                current_lr *= 0.95
                if verbose and epoch % print_every == 0:
                    print(f"  → 学习率衰减至: {current_lr:.6f}")
        
        # 恢复最佳权重
        if best_W is not None:
            self.W = best_W
            if verbose:
                print(f"\n✅ 训练完成！最佳验证准确率: {best_val_acc:.4f}")
        
        return history
    
    def predict(self, X):
        """
        预测类别
        
        Args:
            X: 输入数据 [N, D]
            
        Returns:
            predictions: 预测类别 [N]
        """
        scores = X.dot(self.W)
        return np.argmax(scores, axis=1)
    
    def evaluate(self, X, y):
        """
        计算准确率
        
        Args:
            X: 输入数据 [N, D]
            y: 真实标签 [N]
            
        Returns:
            accuracy: 准确率
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def save_model(self, filepath):
        """保存模型权重"""
        np.save(filepath, self.W)
    
    def load_model(self, filepath):
        """加载模型权重"""
        self.W = np.load(filepath)
