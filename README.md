# Machine Learning 学习笔记与代码复现

我的机器学习学习仓库,按主题存放**复现代码、实验笔记和心得**。

## 目录结构

| 目录 | 内容 |
|---|---|
| `00_basics/` | Python / NumPy / Pandas / Matplotlib 基础练习 |
| `01_linear_models/` | 线性回归、逻辑回归、梯度下降(手写实现) |
| `02_classic_ml/` | 决策树、随机森林、SVM、KNN、朴素贝叶斯、聚类、降维 |
| `03_neural_network/` | 手写神经网络、反向传播 |
| `04_pytorch/` | PyTorch 入门、CNN、RNN/LSTM、Transformer |
| `05_projects/` | 完整项目 / Kaggle 比赛 |
| `datasets/` | 所有数据集统一存放 |
| `notes/` | 学习笔记、读书笔记 |
| `utils/` | 复用的工具函数(画图、评估指标等) |

## 每个模块的内部结构

每学一个算法,在对应模块下建一个子文件夹,代码 + 笔记 + 心得放一起:

```
01_linear_models/
└── linear_regression/
    ├── linear_regression.py   # 实现代码
    ├── notebook.ipynb         # 实验 / 复现笔记
    └── README.md              # 结论、踩坑记录、心得
```

## 学习路线(代码优先,打勾记录进度)

- [ ] 阶段 0 · 工具准备 —— Python、NumPy、Pandas、Matplotlib
- [ ] 阶段 1 · 手写核心算法 —— 线性 / 逻辑回归、梯度下降
- [ ] 阶段 2 · 经典机器学习 —— 决策树、SVM、聚类、降维(sklearn 复现)
- [ ] 阶段 3 · 深度学习 —— 神经网络、PyTorch、CNN、RNN
- [ ] 阶段 4 · 项目实战 —— 完整 pipeline 项目 / 比赛

## 学习资源

- **主线(代码优先)**: 李沐《动手学深度学习》 https://d2l.ai
- **理论**: 周志华《机器学习》(西瓜书)+ 南瓜书《机器学习公式详解》
- **理论**: 李航《统计学习方法》(第 2 版)
- **课程**: 吴恩达 Machine Learning Specialization(Coursera)、李宏毅《机器学习》(B 站)

## 进度记录

在这里记录每个模块的完成时间和学习心得。
