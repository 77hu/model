# model
上市公司ST/退市预测机器学习流水线，基于多算法对比和分组建模的金融预测系统。  
本项目针对中国上市公司财务困境预测场景，使用39,440条记录（5,151家公司，2012-2023年），通过4种算法（LR/RF/XGBoost/GA-优化决策树）在2个粒度层级（全局vs分组）上进行exhaustive对比实验。按行业/规模/产业链/公司年龄四维分组建模，产出100+结果文件。  
项目实现了从数据处理到模型训练、阈值优化、特征重要性分析的完整管线。

# 时间表
#### 2025.12.05
完成数据清洗和merged_clean_data.csv构建  
#### 2025.12.06
完成Logistic Regression全局模型开发（step4_lr.py）  
#### 2025.12.07
完成Random Forest全局模型开发（step5_rf.py）  
#### 2025.12.08
完成XGBoost全局模型开发（step6_xgb.py）  
#### 2025.12.09
完成GA-优化决策树开发（step7_ga_tree.py），集成GASearchCV  
#### 2025.12.10
完成四维分组建模脚本开发（_grouped.py系列）  
#### 2025.12.12
完成批量训练，产出100+结果文件  
#### 2025.12.14
完成结果分析和特征重要性统计

# 目录
<a href="#1-项目介绍">1 项目介绍</a>  
- <a href="#关于st退市预测">1.1 关于ST/退市预测</a>  
- <a href="#目录结构">1.2 目录结构</a>  
- <a href="#依赖">1.3 依赖</a>  
- <a href="#模型对比">1.4 模型对比</a>  

<a href="#如何使用">2 如何使用</a>  
- <a href="#安装依赖">2.1 安装依赖</a>  
- <a href="#全局模型训练">2.2 全局模型训练</a>  
- <a href="#分组建模">2.3 分组建模</a>  
- <a href="#查看结果">2.4 查看结果</a>  

<a href="#统计数据">3 统计数据</a>  
- <a href="#训练数据统计">3.1 训练数据统计</a>  

<a href="#开发说明">4 开发说明</a>  

<a href="#已知问题">5 已知问题</a>  


# 1 项目介绍
## 1.1 关于ST/退市预测
中国A股市场对财务状况异常的上市公司实施ST（Special Treatment）特别处理，预测公司未来是否会被ST或退市对投资决策和风险控制具有重要价值。

目前常见的方法对比：

| 方法名称 | 相关要点 |
| ------ | ------ |
| 单一模型全局训练 | 忽略行业/规模差异，异质性公司共用同一模型 |
| 传统统计方法 | 可解释性强但对非线性关系建模能力弱 |
| 分组建模 | 本项目采用的方案，按行业/规模/产业链/年龄分组，独立训练 |
| 深度学习 | 适合大规模数据但可解释性差 |

本项目使用**4种算法 × 2个粒度层级 × 4个分组维度**的系统性实验设计：

| 算法 | 全局/分组 | 分组维度 |
| ------ | ------ | ------ |
| Logistic Regression | 全局 + 分组 | 行业(11类) / 规模(SML) / 产业链(上中下) / 年龄(青中老年) |
| Random Forest (500 trees) | 全局 + 分组 | 同上 |
| XGBoost (scale_pos_weight) | 全局 + 分组 | 同上 |
| GA-优化决策树 | 全局 + 分组 | 同上 |

## 1.2 目录结构
### 1.2.1 全局模型
| 序号 | 文件名称 | 说明 |
| ------ | ------ | ------ |
| 1 | `merged_clean_data.csv` | 清洗后的主数据集（39,440×33） |
| 2 | `step4_lr.py` | LR模型（全局） |
| 3 | `step5_rf.py` | RF模型（全局） |
| 4 | `step6_xgb.py` | XGBoost模型（全局） |
| 5 | `step7_ga_tree.py` | GA-Tree模型（全局） |
| 6 | `lr_results/` | LR全局结果目录 |

### 1.2.2 分组模型
| 序号 | 文件名称 | 说明 |
| ------ | ------ | ------ |
| 1 | `model/model/step4_lr_grouped.py` | 分组LR |
| 2 | `model/model/step5_rf_grouped.py` | 分组RF |
| 3 | `model/model/step6_xgb_grouped.py` | 分组XGB |
| 4 | `model/model/step7_ga_tree_grouped.py` | 分组GA-Tree |
| 5 | `lr_results_grouped/` | 分组LR结果 |
| 6 | `rf_results_grouped/` | 分组RF结果 |
| 7 | `xgb_results_grouped/` | 分组XGB结果 |
| 8 | `ga_tree_results_grouped/` | 分组GA-Tree结果 |

### 1.2.3 分组子目录结构
```
ga_tree_results_grouped/
├── age/          (young / middle / old)
├── chain/        (downstream / midstream / upstream)
├── industry/     (A_农林牧渔 / B_采矿业 / C_制造业 / D_电力水气 / ...)
└── size/         (small / medium / large)
```
每个子组包含：`best_params.json` + `feature_importance.csv` + `test_pred.csv` + `threshold_search.csv` + `tree_top3.png`

## 1.3 依赖
```
pip install pandas numpy scikit-learn xgboost sklearn-genetic-opt matplotlib
```

## 1.4 模型对比
| 算法 | 关键配置 | 预处理 |
| ------ | ------ | ------ |
| Logistic Regression | class_weight="balanced", liblinear, max_iter=5000 | 中位数填充 + StandardScaler + OHE |
| Random Forest | n_estimators=500, min_samples_split=10 | 中位数填充 + OHE |
| XGBoost | n_estimators=500, max_depth=4, lr=0.05, scale_pos_weight动态计算 | 中位数填充 + 稀疏OHE |
| GA-Tree | GASearchCV(pop=20, gen=15, cv=3, scoring=f1) | 同RF |

# 2 如何使用
## 2.1 安装依赖
```
pip install pandas numpy scikit-learn xgboost sklearn-genetic-opt matplotlib
```

## 2.2 全局模型训练
```
python step4_lr.py    # Logistic Regression
python step5_rf.py    # Random Forest
python step6_xgb.py   # XGBoost
python step7_ga_tree.py  # GA-优化决策树
```

## 2.3 分组建模
```
cd model/model
python step4_lr_grouped.py
python step5_rf_grouped.py
python step6_xgb_grouped.py
python step7_ga_tree_grouped.py
```

## 2.4 查看结果
- 全局结果：`lr_results/`目录
- 分组结果：`*_results_grouped/`目录（按行业/规模/产业链/年龄分组）

# 3 统计数据
## 3.1 训练数据统计
| 项目 | 数值 |
| ------ | ------ |
| 样本数 | 39,440行 |
| 公司数 | 5,151家 |
| 时间跨度 | 2012-2023年 |
| 特征数 | 19个（18数值 + 1分类） |
| 正例率 | 0.79%（312/39,440，极度不平衡） |
| 全局模型数 | 4个（LR/RF/XGB/GA-Tree） |
| 分组模型数 | 100+个（4维度 × 分组 × 4算法） |
| 输出文件 | 100+个（CSV + JSON + PNG） |

# 4 开发说明
- 时间划分：train≤2020 / valid=2021 / test≥2022（严格时序防泄漏）
- 阈值优化：0.05-0.95搜索，以F1为目标（默认0.5在不平衡数据上效果差）
- 跳过逻辑：组内训练样本<30或单类别组自动跳过
- 小行业合并：<300样本或<5正例的行业合并为"其他小行业"

# 5 已知问题
1. 极度不平衡数据（0.79%正例率），需要class_weight/scale_pos_weight处理
2. GA-Tree训练时间较长（pop=20, gen=15, cv=3）
3. 部分小组样本不足，模型可能不稳定
