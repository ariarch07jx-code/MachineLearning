# 逻辑回归

手写二分类逻辑回归。`logistic_regression.py` 是能复用的实现,`notebook.ipynb` 是实验过程。

## 文件

| 文件 | 内容 |
|---|---|
| `logistic_regression.py` | `LogisticRegression` 类 + `sigmoid` / `bce` 两个函数 |
| `notebook.ipynb` | 二维两类点(学习率、决策边界) + 乳腺癌数据(标准化、混淆矩阵、和 sklearn 对比) |
| `README.md` | 结论、踩坑、还没做的 |

## 用法

```python
from logistic_regression import LogisticRegression

model = LogisticRegression(lr=0.5, n_iters=2000).fit(X, y)
model.predict_proba(X)   # 概率,shape (n,)
model.predict(X)         # 0 / 1
model.score(X, y)        # 准确率
```

标签必须是 0 和 1。单独跑一下:`python logistic_regression.py`。

## 结论

| 数据 | 设置 | loss | 准确率 |
|---|---|---|---|
| 二维两类点,200 个 | lr=0.5,1000 步 | 0.1013 | 0.970 |
| 同上 | lr=0.5,20000 步 | 0.0874 | 0.970 |
| 乳腺癌,455 训练 / 114 测试 | 标准化,lr=0.5,2000 步 | 0.0411 | 0.9912 / 0.9825 |
| 乳腺癌 + sklearn,penalty=None | lbfgs 跑到收敛 | - | 1.0 / 0.9474 |

- 两类点本身有重叠,两万步只是把 loss 从 0.101 压到 0.087,准确率一动不动,97% 就是上限。
- 同一个模型,梯度下降跑 2000 步就停(相当于早停)测试集 0.9825,sklearn 推到训练集 100% 测试集反而 0.9474,差 0.035。优化方式会直接影响泛化。
- 30 个特征不标准化直接训练,测试集只有 0.8684;标准化后 0.9825。
- 混淆矩阵里错的 2 个都是恶性判成良性,准确率好看不代表这类错误可以接受。

## 踩坑

- 概率没截断,`log(0)` 会得到 -inf,loss 变成 nan。`bce` 里把 p 截到 [1e-12, 1-1e-12]。
- 标签写成 -1 / 1 不会报错,但梯度 (p - y) 就算错了,模型还能训出个东西,只是结果不对。统一用 0 / 1。
- sigmoid 里的 exp 会溢出。lr 开太大时 z 能到 -800,`np.exp(-z)` 直接 inf,虽然 `1/(1+inf)` 还是 0,但会刷一屏 RuntimeWarning。
- 多特征一定要标准化,不然 2000 步下去还差得远。
- 阈值写死 0.5 不一定合适,筛查场景更怕漏判,要单独调。
- 本机 numpy 2.0 在 Accelerate 上对 `matmul` 误报浮点警告,notebook 里过滤掉了,和线性回归那边一样。

## 还没做

- L2 正则和 early stopping 的正式实现(现在只是靠步数少被动早停)
- 多分类的 softmax 版本
- 类别不平衡处理(class_weight / 重采样)
- PR 曲线和阈值选择
- 在 `make_moons` 上跑一遍,亲眼看一下直线边界错在哪
