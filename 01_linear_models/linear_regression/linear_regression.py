"""手写线性回归:梯度下降和正规方程两种解法。"""

import numpy as np


def mse(y_true, y_pred):
    """均方误差。"""
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


class LinearRegression:
    """最小化 MSE 的线性回归 y = X @ w + b。

    method="gd" 用梯度下降,method="normal" 用正规方程直接解。
    """

    def __init__(self, lr=0.01, n_iters=1000, method="gd"):
        self.lr = lr
        self.n_iters = n_iters
        self.method = method

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float).ravel()
        n = X.shape[0]

        if self.method == "normal":
            Xb = np.c_[np.ones(n), X]
            theta = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
            self.intercept_ = theta[0]
            self.coef_ = theta[1:]
            self.loss_history_ = [mse(y, Xb @ theta)]
        elif self.method == "gd":
            self.coef_ = np.zeros(X.shape[1])
            self.intercept_ = 0.0
            self.loss_history_ = []
            for _ in range(self.n_iters):
                err = X @ self.coef_ + self.intercept_ - y
                self.coef_ = self.coef_ - self.lr * 2 * (X.T @ err) / n
                self.intercept_ = self.intercept_ - self.lr * 2 * err.mean()
                self.loss_history_.append(mse(y, X @ self.coef_ + self.intercept_))
        else:
            raise ValueError("method 只能是 gd 或 normal")
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X @ self.coef_ + self.intercept_

    def score(self, X, y):
        """R²,1 表示完美拟合。"""
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return float(1 - ss_res / ss_tot)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.uniform(1, 9, 100)
    y = 2.5 * x + 5 + rng.normal(0, 3, 100)

    gd = LinearRegression(lr=0.01, n_iters=3000).fit(x, y)
    ne = LinearRegression(method="normal").fit(x, y)

    print("gd    ", round(gd.coef_[0], 4), round(gd.intercept_, 4))
    print("normal", round(ne.coef_[0], 4), round(ne.intercept_, 4))
    print("R2", round(gd.score(x, y), 4), round(ne.score(x, y), 4))
