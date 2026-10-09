# Machine Learning 学习笔记与代码复现

我的机器学习学习仓库,按主题存放**复现代码、实验笔记和心得**。

## 目录结构

| 目录 | 内容 | 状态 |
|---|---|---|
| `00_basics/` | Python 基础 + NumPy / Pandas / Matplotlib | 🔄 进行中 |
| `01_linear_models/` | 线性回归、逻辑回归、梯度下降(手写实现) | ⬜ 待开始 |
| `02_classic_ml/` | 决策树、随机森林、SVM、KNN、聚类、降维 | ⬜ 待开始 |
| `03_neural_network/` | 手写神经网络、反向传播 | ⬜ 待开始 |
| `04_pytorch/` | PyTorch 入门、CNN、RNN/LSTM、Transformer | ⬜ 待开始 |
| `05_projects/` | 完整项目 / Kaggle 比赛 | ⬜ 待开始 |
| `datasets/` | 所有数据集统一存放 | — |
| `notes/` | 学习笔记、读书笔记 | — |
| `utils/` | 复用的工具函数(画图、评估指标等) | — |

### 基础阶段 `00_basics/` 内部结构

基础阶段用 `0.x_主题.ipynb` 命名,一个 notebook 一个主题:

```
00_basics/
├── 0.1_python_basics.ipynb    # Python 基础语法 ✅ 已学完
├── 0.2_numpy.ipynb            # NumPy 数组运算
├── 0.3_pandas.ipynb           # Pandas 表格处理
├── 0.4_matplotlib.ipynb       # Matplotlib 画图
└── README.md
```

### 算法模块 `01~05` 的组织方式

每学一个算法,在对应模块下建一个子文件夹,代码 + 笔记 + 心得放一起:

```
01_linear_models/
└── linear_regression/
    ├── linear_regression.py   # 实现代码
    ├── notebook.ipynb         # 实验 / 复现笔记
    └── README.md              # 结论、踩坑、心得
```

## 学习路线(代码优先)

- [x] 阶段 0 · 工具准备 —— Python 基础 ✅,NumPy / Pandas / Matplotlib 🔄
- [ ] 阶段 1 · 手写核心算法 —— 线性 / 逻辑回归、梯度下降
- [ ] 阶段 2 · 经典机器学习 —— 决策树、SVM、聚类、降维(sklearn 复现)
- [ ] 阶段 3 · 深度学习 —— 神经网络、PyTorch、CNN、RNN
- [ ] 阶段 4 · 项目实战 —— 完整 pipeline 项目 / 比赛

## 学习资源

- **主线(代码优先)**: 李沐《动手学深度学习》 https://d2l.ai
- **理论**: 周志华《机器学习》(西瓜书)+ 南瓜书《机器学习公式详解》
- **理论**: 李航《统计学习方法》(第 2 版)
- **课程**: 吴恩达 Machine Learning Specialization(Coursera)、李宏毅《机器学习》(B 站)

## 环境

- Python 3.9,使用 `venv` 虚拟环境(激活:`source .venv/bin/activate`)
- 依赖见 `requirements.txt`

## 进度记录

| 时间 | 完成内容 |
|---|---|
| 2026-10-09 | Python 基础语法(变量 / 对象 / 内置函数 / 六大类型 / 文件读写 / 格式化) |
