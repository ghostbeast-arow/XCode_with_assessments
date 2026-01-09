# Spaceship Titanic Machine Learning

本目录包含 Kaggle Spaceship Titanic 分类任务的训练代码与可视化输出脚本。

## 目录结构

```
MachineLearning/
├── train.py
├── requirement.txt
└── outputs/            # 运行脚本后自动生成
```

## 数据准备

1. 从 Kaggle 下载数据集：<https://www.kaggle.com/competitions/spaceship-titanic/data>
2. 在 `MachineLearning/` 下新建 `data/` 目录。
3. 将 `train.csv` 和 `test.csv` 放入 `data/` 目录。

```
MachineLearning/
└── data/
    ├── train.csv
    └── test.csv
```

## 环境安装

```bash
pip install -r requirement.txt
```

## 训练与评估

```bash
python train.py --data-dir data --output-dir outputs
```

运行后将生成：
- 5 折交叉验证指标表 `outputs/cv_metrics.csv`
- 缺失值分析表与图 `outputs/missing_values.csv` / `outputs/missing_values.png`
- 目标分布图 `outputs/target_distribution.png`
- 每个模型的 ROC 曲线与混淆矩阵图
- 最佳模型在测试集上的预测结果 `outputs/submission.csv`

## 说明

- 脚本默认比较 Logistic Regression 与 XGBoost（若未安装 xgboost，将自动跳过）。
- 特征工程包括客舱拆分、乘客组大小、姓名长度、消费总额等。
- 对分类特征进行独热编码，对数值特征进行标准化，以满足实验要求。
