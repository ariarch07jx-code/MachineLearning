# 线性回归

手写最小二乘线性回归。`linear_regression.py` 是能复用的实现,`notebook.ipynb` 是这部分的实验和推导过程。

## 文件

| 文件 | 内容 |
|---|---|
| `linear_regression.py` | `LinearRegression` 类,支持梯度下降和正规方程两种解法 |
| `notebook.ipynb` | 单特征实验(学习率、标准化、正规方程对比) + 糖尿病数据多特征实验 |
| `README.md` | 结论、踩坑、还没做的 |

## 用法

```python
from linear_regression import LinearRegression

model = LinearRegression(lr=0.01, n_iters=1000).fit(x, y)
model.coef_          # 权重,shape 是 (n_features,)
model.intercept_     # 截距
model.predict(x)
model.score(x, y)    # R²
```

走正规方程就换 `LinearRegression(method="normal")`,不用管 lr 和迭代次数。

单独看效果:`python linear_regression.py`,会打印两种解法在同一份数据上的系数。

## 结论

实验数据是 y = 2.5x + 5 + N(0, 3),100 个点,x 在 1 到 9 之间。

| 做法 | 步数 | w | b | MSE |
|---|---|---|---|---|
| 梯度下降 lr=0.01 | 1000 | 2.4770 | 4.8588 | 8.4652 |
| 标准化后 lr=0.1 | 100 | 2.4510 | 5.0266 | 8.4605 |
| 正规方程 | 1 | 2.4510 | 5.0266 | 8.4605 |

- 最优 MSE 是 8.4605,噪声方差就是 9,再往下调模型没有意义。
- 标准化之后 lr 能放大 10 倍,步数少一个数量级。学习率不是独立的超参,和特征尺度绑在一起。
- 正规方程一步给精确解,梯度下降跑了 1000 步系数还差 0.026。
- 糖尿病数据(10 个特征,353 条训练样本)上,GD 要跑两万步才和 sklearn 对上,系数最大差 2e-7,test R² 都是 0.3322。特征是相关的,损失面很扁,所以慢。

## 踩坑

- 忘了截距 b,模型只能拟合过原点的线,loss 怎么调都下不来。加 b 等价于给 X 补一列 1,正规方程里就是这么做的。
- lr 从 0.001 调到 0.05,loss 会直接冲到 5e166。梯度爆炸不抛异常,只会安静地给出 inf 或 nan,所以训练前先画 loss 曲线。
- 特征不标准化时,lr 的上限被 x 的量级卡死。x 在 1 到 9,梯度里带着 x²,有效的 lr 只有 0.01 这个量级。
- XᵀX 完全共线时 `solve` 抛 LinAlgError,但接近共线时不会报错,只是解出来的系数大得离谱,更难发现。数据里特征有线性关系时先查 VIF 或者直接换 `lstsq`。
- 实现里要把一维的 X 当成 (n, 1) 处理。传进来 `x.shape == (100,)` 时直接算矩阵乘会出错,或者得到一个数而不是向量。
- numpy 2.0 在这台机器的 Accelerate 上会对 `matmul` 误报浮点警告,结果其实是对的,notebook 里过滤掉了。

## 还没做

- L2 正则(Ridge),损失里加 λ||w||²,梯度多一项 2λw
- early stopping:连续若干步 loss 变化小于阈值就停
- mini-batch 和随机梯度下降
- 把 y 也标准化再训练,看系数和 R² 怎么变
- 多项式特征、特征交叉
- 用 `sklearn.datasets` 以外更真实的数据集(房价、销量这种)
