<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-3498DB?style=for-the-badge&logo=python&logoColor=white)](https://xgboost.readthedocs.io/)
[![Genetic Algorithm](https://img.shields.io/badge/GA-Evolutionary-9B59B6?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/geneticalgorithm/)

</div>

<br/>

<h1 align="center">🧠 机器学习模型训练管线</h1>

<h3 align="center"><em>LR → Random Forest → XGBoost → 遗传算法特征优化 · 循序渐进式模型对比</em></h3>

<br/>

---

## 📖 项目概述

循序渐进的机器学习模型训练实验项目，从简单线性模型到复杂集成学习再到进化算法特征工程，全面探索模型性能提升路径。每步独立脚本，按复杂度递增。

### 四阶段训练策略

| 阶段 | 脚本 | 模型 | 复杂度 | 目标 |
|------|------|------|--------|------|
| Step 4 | `step4_lr.py` | **Logistic Regression** | ⭐ | 建立线性 Baseline |
| Step 5 | `step5_rf.py` | **Random Forest** | ⭐⭐⭐ | 集成树模型提升 |
| Step 6 | `step6_xgb.py` | **XGBoost** | ⭐⭐⭐⭐ | 梯度提升极致优化 |
| Step 7 | `step7_ga_tree.py` | **Genetic Algorithm + Tree** | ⭐⭐⭐⭐⭐ | 进化特征选择 |

### 模型对比

| 维度 | LR | Random Forest | XGBoost | GA + Tree |
|------|-----|-------------|---------|-----------|
| 训练速度 | ⚡⚡⚡⚡⚡ | ⚡⚡⚡⚡ | ⚡⚡⚡ | ⚡⚡ |
| 准确率 | ★★☆☆☆ | ★★★★☆ | ★★★★★ | ★★★★★ |
| 可解释性 | ★★★★★ | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ |
| 过拟合风险 | 低 | 低 | 中 | 中 |
| 超参数数量 | 少 | 中 | 多 | 多 |

---

## 🚀 快速开始

```bash
git clone https://github.com/77hu/model.git
cd model

pip install scikit-learn xgboost numpy pandas

python step4_lr.py      # Logistic Regression
python step5_rf.py      # Random Forest
python step6_xgb.py     # XGBoost
python step7_ga_tree.py # Genetic Algorithm
```

---

## 📁 项目结构

```
📦 model/
├── 📜 step4_lr.py              # Logistic Regression
├── 📜 step5_rf.py              # Random Forest
├── 📜 step6_xgb.py             # XGBoost
├── 📜 step7_ga_tree.py         # Genetic Algorithm
├── 📄 merged_clean_data.csv    # 预处理数据
└── 📘 README.md
```

---

## 📄 License

本项目仅供学习和研究使用。
