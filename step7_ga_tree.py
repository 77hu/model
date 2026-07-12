from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ========= 尝试导入遗传搜索 =========
try:
    from sklearn_genetic import GASearchCV
    from sklearn_genetic.space import Integer, Continuous, Categorical
except ImportError:
    raise ImportError(
        "缺少 sklearn-genetic-opt 包，请先运行：pip install sklearn-genetic-opt"
    )

# ==================================
# 1. 读取数据
# ==================================
DATA_PATH = Path(r"C:\Users\16286\Desktop\panel_final.csv")
df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

print("数据形状：", df.shape)

# ==================================
# 2. 特征与标签
# ==================================
feature_cols = [
    "CR","LEV","ICR","TAT","ART","IT",
    "ROE","NPM","ROA","CFOA","CFR",
    "GROWTH","TAG","Top1","Board","Indep",
    "MComp","Vol","PB","Audit"
]
target_col = "label_next"

df = df[["code", "name", "year"] + feature_cols + [target_col]].copy()

# ==================================
# 3. 划分训练/验证/测试集
# ==================================
train_df = df[df["year"] <= 2020].copy()
valid_df = df[df["year"] == 2021].copy()
test_df  = df[df["year"] >= 2022].copy()

X_train = train_df[feature_cols]
y_train = train_df[target_col].astype(int)

X_valid = valid_df[feature_cols]
y_valid = valid_df[target_col].astype(int)

X_test = test_df[feature_cols]
y_test = test_df[target_col].astype(int)

print("\n训练集：", X_train.shape, "正样本率=", y_train.mean())
print("验证集：", X_valid.shape, "正样本率=", y_valid.mean())
print("测试集：", X_test.shape, "正样本率=", y_test.mean())

# ==================================
# 4. 预处理
# ==================================
numeric_features = [
    "CR","LEV","ICR","TAT","ART","IT",
    "ROE","NPM","ROA","CFOA","CFR",
    "GROWTH","TAG","Top1","Board","Indep",
    "MComp","Vol","PB"
]
categorical_features = ["Audit"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ]
)

ga_tree_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ))
])

# ==================================
# 5. 遗传搜索空间
# ==================================
param_grid = {
    "clf__criterion": Categorical(["gini", "entropy"]),
    "clf__max_depth": Integer(2, 8),
    "clf__min_samples_split": Integer(2, 40),
    "clf__min_samples_leaf": Integer(1, 20),
    "clf__ccp_alpha": Continuous(0.0, 0.02)
}

# ==================================
# 6. 遗传搜索
#    scoring先用f1，与你前面阈值目标一致
# ==================================
ga_search = GASearchCV(
    estimator=ga_tree_pipe,
    cv=3,
    scoring="f1",
    population_size=20,
    generations=15,
    param_grid=param_grid,
    criteria="max",
    algorithm="eaMuPlusLambda",
    n_jobs=-1,
    verbose=True,
    keep_top_k=3
)

ga_search.fit(X_train, y_train)

ga_tree_model = ga_search.best_estimator_
print("\n最优参数：")
print(ga_search.best_params_)

# ==================================
# 7. 阈值搜索（验证集）
# ==================================
valid_prob = ga_tree_model.predict_proba(X_valid)[:, 1]

thresholds = np.arange(0.05, 0.96, 0.05)
best_thr = 0.5
best_f1 = -1

rows = []
for thr in thresholds:
    pred = (valid_prob >= thr).astype(int)
    p = precision_score(y_valid, pred, zero_division=0)
    r = recall_score(y_valid, pred, zero_division=0)
    f1 = f1_score(y_valid, pred, zero_division=0)
    rows.append([thr, p, r, f1])
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

thr_df = pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f1"])
print("\n验证集阈值搜索结果：")
print(thr_df)

print(f"\n最优阈值 = {best_thr:.2f}, 对应验证集 F1 = {best_f1:.4f}")

# ==================================
# 8. 评价函数
# ==================================
def evaluate_model(name, model, X, y, threshold):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)

    res = {
        "model": name,
        "threshold": threshold,
        "AUC": roc_auc_score(y, prob),
        "PR_AUC": average_precision_score(y, prob),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0)
    }

    print(f"\n===== {name} =====")
    print(res)
    print("Confusion Matrix:")
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred, digits=4, zero_division=0))

    pred_df = pd.DataFrame({
        "y_true": y.values,
        "y_prob": prob,
        "y_pred": pred
    })
    return res, pred_df

# ==================================
# 9. 验证集/测试集评价
# ==================================
valid_res, valid_pred_df = evaluate_model("GA_TREE_valid", ga_tree_model, X_valid, y_valid, best_thr)
test_res, test_pred_df = evaluate_model("GA_TREE_test", ga_tree_model, X_test, y_test, best_thr)

# ==================================
# 10. 导出特征重要性
# ==================================
feature_names = ga_tree_model.named_steps["prep"].get_feature_names_out()
tree_clf = ga_tree_model.named_steps["clf"]
importance = tree_clf.feature_importances_

imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
}).sort_values("importance", ascending=False)

# ==================================
# 11. 画树
# ==================================
SAVE_DIR = Path(r"C:\Users\16286\Desktop\ga_tree_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(24, 12))
plot_tree(
    tree_clf,
    feature_names=feature_names,
    class_names=["NoRisk", "Risk"],
    filled=True,
    max_depth=3,
    fontsize=8
)
plt.tight_layout()
plt.savefig(SAVE_DIR / "ga_tree_top3.png", dpi=300)
plt.close()

# ==================================
# 12. 保存结果
# ==================================
thr_df.to_csv(SAVE_DIR / "ga_tree_threshold_search.csv", index=False, encoding="utf-8-sig")
valid_pred_df.to_csv(SAVE_DIR / "ga_tree_valid_pred.csv", index=False, encoding="utf-8-sig")
test_pred_df.to_csv(SAVE_DIR / "ga_tree_test_pred.csv", index=False, encoding="utf-8-sig")

summary_df = pd.DataFrame([valid_res, test_res])
summary_df.to_csv(SAVE_DIR / "ga_tree_summary.csv", index=False, encoding="utf-8-sig")

imp_df.to_csv(SAVE_DIR / "ga_tree_feature_importance.csv", index=False, encoding="utf-8-sig")

print(f"\n结果已保存到：{SAVE_DIR}")
print("\n进化树重要性前20：")
print(imp_df.head(20))