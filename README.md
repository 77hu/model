<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-3498DB?style=for-the-badge&logo=python&logoColor=white)](https://xgboost.readthedocs.io/)
[![Genetic Algorithm](https://img.shields.io/badge/GA-Evolutionary-9B59B6?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/geneticalgorithm/)

</div>

<br/>

<h1 align="center">🧠 机器学习模型训练管线</h1>

<h3 align="center"><em>LR → RF → XGBoost → 遗传算法特征优化 · 多模型对比实验</em></h3>

<br/>

---

## 📑 目录

- [📖 项目概述](#-项目概述)
- [🧩 模型管线](#-模型管线)
- [📊 模型对比](#-模型对比)
- [🚀 快速开始](#-快速开始)
- [📁 项目结构](#-项目结构)

---

## 📖 项目概述

循序渐进的机器学习模型训练实验项目，从简单线性模型到复杂集成学习再到进化算法特征工程，全面探索模型性能提升路径。

### 训练策略

| 阶段 | 模型 | 目标 |
|------|------|------|
| Step 4 | Logistic Regression | 建立 Baseline |
| Step 5 | Random Forest | 集成学习提升 |
| Step 6 | XGBoost | 梯度提升优化 |
| Step 7 | Genetic Algorithm + Tree | 进化特征选择 |

---

## 🧩 模型管线

```
┌──────────────────────┐
│  数据预处理           │
│  merged_clean_data   │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Step 4: LR          │  线性 Baseline
│  step4_lr.py         │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Step 5: Random      │  集成树模型
│  Forest              │
│  step5_rf.py         │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Step 6: XGBoost     │  梯度提升
│  step6_xgb.py        │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Step 7: GA Tree     │  遗传算法
│  step7_ga_tree.py    │  特征优化
└──────────────────────┘
```

---

## 📊 模型对比

| 模型 | 复杂度 | 训练速度 | 准确率 | 可解释性 |
|------|--------|---------|--------|----------|
| Logistic Regression | 低 | ⚡⚡⚡⚡⚡ | ★★☆☆☆ | ★★★★★ |
| Random Forest | 中 | ⚡⚡⚡⚡ | ★★★★☆ | ★★★☆☆ |
| XGBoost | 中高 | ⚡⚡⚡ | ★★★★★ | ★★☆☆☆ |
| GA + Tree | 高 | ⚡⚡ | ★★★★★ | ★★☆☆☆ |

---

## 🚀 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/77hu/model.git
cd model

# 2. 安装依赖
pip install scikit-learn xgboost numpy pandas

# 3. 按顺序运行
python step4_lr.py      # Logistic Regression
python step5_rf.py      # Random Forest
python step6_xgb.py     # XGBoost
python step7_ga_tree.py # GA Tree
```

---

## 📁 项目结构

```
📦 model/
├── 📜 step4_lr.py              # Logistic Regression 训练
├── 📜 step5_rf.py              # Random Forest 训练
├── 📜 step6_xgb.py             # XGBoost 训练
├── 📜 step7_ga_tree.py         # 遗传算法 + 决策树
├── 📄 merged_clean_data.csv    # 预处理后的数据集
└── 📘 README.md                # 本文档
```

---

## 📄 License

本项目仅供学习和研究使用。
