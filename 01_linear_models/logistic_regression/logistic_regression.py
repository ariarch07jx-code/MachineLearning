"""手写逻辑回归:二分类,梯度下降 + 交叉熵。"""

import numpy as np


def sigmoid(z):
    """1 / (1 + e^-z),把任意实数压到 (0, 1)。"""
    return 1.0 / (1.0 + np.exp(-z))


def bce(y_true, p):
    """二元交叉熵,概率先截一下,免得 log(0) 出 -inf。"""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


class LogisticRegression:
    """二分类,标签用 0 / 1。"""

    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]

        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iters):
            p = sigmoid(X @ self.coef_ + self.intercept_)
            self.coef_ = self.coef_ - self.lr * (X.T @ (p - y)) / n
            self.intercept_ = self.intercept_ - self.lr * (p - y).mean()
            self.loss_history_.append(bce(y, p))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return sigmoid(X @ self.coef_ + self.intercept_)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        """准确率。"""
        return float((self.predict(X) == np.asarray(y).ravel()).mean())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal([2, 2], 1.0, (100, 2)), rng.normal([5, 5], 1.2, (100, 2))]
    y = np.r_[np.zeros(100), np.ones(100)]

    model = LogisticRegression(lr=0.5, n_iters=20000).fit(X, y)
    print("coef     ", np.round(model.coef_, 3))
    print("intercept", round(model.intercept_, 3))
    print("loss     ", round(model.loss_history_[-1], 4))
    print("accuracy ", round(model.score(X, y), 4))
