# Machine Learning 学习笔记与代码复现

我的机器学习学习仓库,按主题存放复现代码、实验笔记和心得。以代码为主,理论卡住再回头补。

## 目录结构

| 目录 | 内容 | 状态 |
|---|---|---|
| `00_basics/` | Python 基础 + NumPy / Pandas / Matplotlib | ✅ 已完成 |
| `01_linear_models/` | 线性回归、逻辑回归、梯度下降(手写实现) | ✅ 已完成 |
| `02_classic_ml/` | 决策树、随机森林、SVM、KNN、聚类、降维 | ✅ 已完成 |
| `03_neural_network/` | 手写神经网络、反向传播 | ⬜ 待开始 |
| `04_pytorch/` | PyTorch 入门、CNN、RNN/LSTM、Transformer | ⬜ 待开始 |
| `05_projects/` | 完整项目 / Kaggle 比赛 | ⬜ 待开始 |
| `datasets/` | 所有数据集统一存放 | 空 |
| `utils/` | 复用的工具函数(画图、评估指标等) | 空 |

### 基础阶段 `00_basics/`

用 `0.x_主题.ipynb` 命名,一个 notebook 一个主题:

```
00_basics/
├── 0.1_python_basics.ipynb    # Python 基础语法
├── 0.2_numpy.ipynb            # NumPy 数组运算
├── 0.3_pandas.ipynb           # Pandas 表格处理
├── 0.4_matplotlib.ipynb       # Matplotlib 画图
├── sine_demo.png              # 0.4 里保存图片那节的输出
└── README.md
```

### 算法模块 `01~05`

每学一个算法,在对应模块下建一个子文件夹。01 阶段是手写的,带一个 `.py` 实现;02 以后用 sklearn 复现,只放 notebook 和 README:

```
01_linear_models/
└── linear_regression/
    ├── linear_regression.py   # 实现代码
    ├── notebook.ipynb         # 实验 / 复现笔记
    └── README.md              # 结论、踩坑、心得

02_classic_ml/
└── decision_tree/
    ├── notebook.ipynb         # sklearn 复现,不写单独实现
    └── README.md
```

notebook 的结构统一是:问题 → 数据 → 实验 → 小结,写到小结就结束。想到的延伸记在各自 README 的还没做里。

## 学习路线

- [x] 阶段 0 · 工具准备:Python 基础、NumPy、Pandas、Matplotlib
- [x] 阶段 1 · 手写核心算法:线性 / 逻辑回归、梯度下降
- [x] 阶段 2 · 经典机器学习:决策树、随机森林、SVM、KNN、聚类、降维
- [ ] 阶段 3 · 深度学习:神经网络、PyTorch、CNN、RNN
- [ ] 阶段 4 · 项目实战:完整 pipeline 项目 / 比赛

## 学习资源

- 主线(代码优先): 李沐《动手学深度学习》 https://d2l.ai
- 理论: 周志华《机器学习》(西瓜书)+ 南瓜书《机器学习公式详解》
- 理论: 李航《统计学习方法》(第 2 版)
- 课程: 吴恩达 Machine Learning Specialization(Coursera)、李宏毅《机器学习》(B 站)

## 环境

- Python 3.9,venv 虚拟环境,激活:`source .venv/bin/activate`
- 依赖见 `requirements.txt`
- 跑 notebook:激活环境后 `jupyter lab`,打开对应模块下的 `notebook.ipynb`,内核选 `.venv`

## 进度记录

| 时间 | 完成内容 |
|---|---|
| 2026-10-09 | Python 基础语法(变量 / 对象 / 内置函数 / 六大类型 / 文件读写 / 格式化) |
| 2026-10-25 | NumPy、Pandas、Matplotlib 三个 notebook(数组运算 / 表格处理 / 画图) |
| 2026-10-25 | 线性回归:手写梯度下降 + 正规方程,和 sklearn 对拍 |
| 2026-10-25 | 逻辑回归:sigmoid + 交叉熵,乳腺癌数据 test 0.98 |
| 2026-10-25 | 决策树:分裂、剪枝、特征重要性,配过拟合实验 |
| 2026-10-25 | 随机森林:bagging 方差实验、树的数量、OOB、特征重要性 |
| 2026-10-25 | SVM:间隔和支持向量、C、核函数、gamma、特征缩放 |
| 2026-10-25 | KNN:k 的扫描、概率面、距离权重、特征缩放 |
| 2026-10-25 | 聚类:K-Means 手写、肘部法与轮廓系数、初始化、DBSCAN |
| 2026-10-25 | 降维:PCA(SVD)、解释方差、重构误差、标准化对比 |
