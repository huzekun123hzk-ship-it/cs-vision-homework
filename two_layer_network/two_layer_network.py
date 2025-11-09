"""
两层全连接神经网络分类器

改进版本包含：
- Xavier/He权重初始化
- Dropout正则化
- 学习率调度
- 更稳定的数值计算
"""

import numpy as np
import pickle


class TwoLayerNetwork:
    """
    两层全连接神经网络分类器
    
    网络架构：
        输入层 (D维) -> 隐藏层 (H维, ReLU) -> Dropout -> 输出层 (C维, Softmax)
    """
    
    def __init__(self, input_size, hidden_size, num_classes, 
                 weight_scale=None, reg=0.0, dropout=0.0):
        """
        初始化网络
        
        参数：
            input_size: 输入维度
            hidden_size: 隐藏层维度
            num_classes: 类别数
            weight_scale: 权重初始化标准差（None时使用He初始化）
            reg: L2正则化系数
            dropout: Dropout比率（0表示不使用）
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.reg = reg
        self.dropout = dropout
        
        # 初始化参数
        self.params = {}
        self._init_params(weight_scale)
        
        # 缓存
        self.cache = {}
    
    def _init_params(self, weight_scale):
        """
        使用He初始化（适用于ReLU）
        
        He初始化：std = sqrt(2 / fan_in)
        """
        D, H, C = self.input_size, self.hidden_size, self.num_classes
        
        if weight_scale is None:
            # He初始化（对ReLU更好）
            std1 = np.sqrt(2.0 / D)
            std2 = np.sqrt(2.0 / H)
        else:
            std1 = std2 = weight_scale
        
        self.params['W1'] = std1 * np.random.randn(D, H)
        self.params['b1'] = np.zeros(H)
        self.params['W2'] = std2 * np.random.randn(H, C)
        self.params['b2'] = np.zeros(C)
    
    def forward(self, X, train_mode=True):
        """
        前向传播
        
        参数：
            X: 输入数据 (N, D)
            train_mode: 是否训练模式（影响dropout）
        
        返回：
            scores: 输出分数 (N, C)
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        # 第一层
        z1 = X.dot(W1) + b1
        a1 = np.maximum(0, z1)
        
        # Dropout（仅训练时）
        dropout_mask = None
        if train_mode and self.dropout > 0:
            dropout_mask = (np.random.rand(*a1.shape) > self.dropout) / (1 - self.dropout)
            a1 *= dropout_mask
        
        # 第二层
        z2 = a1.dot(W2) + b2
        
        # 缓存
        self.cache = {
            'X': X,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'dropout_mask': dropout_mask
        }
        
        return z2
    
    def softmax_loss(self, scores, y):
        """
        计算Softmax损失和梯度
        
        使用数值稳定技巧：
        1. 减去最大值防止溢出
        2. 添加epsilon防止log(0)
        
        参数：
            scores: 网络输出 (N, C)
            y: 真实标签 (N,)
        
        返回：
            loss: 总损失（数据损失 + 正则化损失）
            dscores: scores的梯度 (N, C)
        """
        N = scores.shape[0]
        
        # 数值稳定的softmax
        scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # 交叉熵损失
        epsilon = 1e-8
        correct_log_probs = -np.log(probs[np.arange(N), y] + epsilon)
        data_loss = np.mean(correct_log_probs)
        
        # L2正则化损失
        reg_loss = self.reg * (
            np.sum(self.params['W1'] * self.params['W1']) +
            np.sum(self.params['W2'] * self.params['W2'])
        )
        
        total_loss = data_loss + reg_loss
        
        # 梯度
        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N
        
        return total_loss, dscores
    
    def backward(self, dscores):
        """
        反向传播计算梯度
        
        参数：
            dscores: 输出层梯度 (N, C)
        
        返回：
            grads: 参数梯度字典
        """
        X = self.cache['X']
        z1 = self.cache['z1']
        a1 = self.cache['a1']
        dropout_mask = self.cache['dropout_mask']
        
        W1 = self.params['W1']
        W2 = self.params['W2']
        
        grads = {}
        
        # 第二层梯度
        grads['W2'] = a1.T.dot(dscores) + 2 * self.reg * W2
        grads['b2'] = np.sum(dscores, axis=0)
        
        # 传播到隐藏层
        da1 = dscores.dot(W2.T)
        
        # Dropout反向传播
        if dropout_mask is not None:
            da1 *= dropout_mask
        
        # ReLU反向传播
        dz1 = da1.copy()
        dz1[z1 <= 0] = 0
        
        # 第一层梯度
        grads['W1'] = X.T.dot(dz1) + 2 * self.reg * W1
        grads['b1'] = np.sum(dz1, axis=0)
        
        return grads
    
    def train_step(self, X, y, learning_rate):
        """
        单步训练
        
        参数：
            X: 批次数据
            y: 批次标签
            learning_rate: 学习率
        
        返回：
            loss: 当前损失值
        """
        scores = self.forward(X, train_mode=True)
        loss, dscores = self.softmax_loss(scores, y)
        grads = self.backward(dscores)
        
        # 参数更新
        for param_name in self.params:
            self.params[param_name] -= learning_rate * grads[param_name]
        
        return loss
    
    def predict(self, X):
        """
        预测类别标签
        
        参数：
            X: 输入数据 (N, D)
        
        返回：
            predictions: 预测标签 (N,)
        """
        scores = self.forward(X, train_mode=False)
        predictions = np.argmax(scores, axis=1)
        return predictions
    
    def evaluate(self, X, y):
        """
        评估准确率
        
        参数：
            X: 输入数据
            y: 真实标签
        
        返回：
            accuracy: 准确率
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def save_model(self, filepath):
        """保存模型参数"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'params': self.params,
                'config': {
                    'input_size': self.input_size,
                    'hidden_size': self.hidden_size,
                    'num_classes': self.num_classes,
                    'reg': self.reg,
                    'dropout': self.dropout
                }
            }, f)
    
    def load_model(self, filepath):
        """加载模型参数"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.params = data['params']
            config = data['config']
            self.input_size = config['input_size']
            self.hidden_size = config['hidden_size']
            self.num_classes = config['num_classes']
            self.reg = config['reg']
            self.dropout = config['dropout']
