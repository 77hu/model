from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

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

# XGBoost 需要先把预处理后的矩阵拿出来
X_train_trans = preprocess.fit_transform(X_train)
X_valid_trans = preprocess.transform(X_valid)
X_test_trans = preprocess.transform(X_test)

# 稀疏矩阵可直接给 xgboost
pos = (y_train == 1).sum()
neg = (y_train == 0).sum()
scale_pos_weight = neg / max(pos, 1)

print("\nscale_pos_weight =", scale_pos_weight)

# ==================================
# 5. XGBoost模型
# ==================================
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    min_child_weight=5,
    gamma=0.0,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train_trans, y_train)

# ==================================
# 6. 验证集阈值搜索
# ==================================
valid_prob = xgb_model.predict_proba(X_valid_trans)[:, 1]

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
# 7. 评价函数
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
# 8. 验证集/测试集评价
# ==================================
valid_res, valid_pred_df = evaluate_model("XGB_valid", xgb_model, X_valid_trans, y_valid, best_thr)
test_res, test_pred_df = evaluate_model("XGB_test", xgb_model, X_test_trans, y_test, best_thr)

# ==================================
# 9. 导出结果
# ==================================
SAVE_DIR = Path(r"C:\Users\16286\Desktop\xgb_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

thr_df.to_csv(SAVE_DIR / "xgb_threshold_search.csv", index=False, encoding="utf-8-sig")
valid_pred_df.to_csv(SAVE_DIR / "xgb_valid_pred.csv", index=False, encoding="utf-8-sig")
test_pred_df.to_csv(SAVE_DIR / "xgb_test_pred.csv", index=False, encoding="utf-8-sig")

summary_df = pd.DataFrame([valid_res, test_res])
summary_df.to_csv(SAVE_DIR / "xgb_summary.csv", index=False, encoding="utf-8-sig")

# ==================================
# 10. 导出特征重要性
# ==================================
feature_names = preprocess.get_feature_names_out()
importance = xgb_model.feature_importances_

imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
}).sort_values("importance", ascending=False)

imp_df.to_csv(SAVE_DIR / "xgb_feature_importance.csv", index=False, encoding="utf-8-sig")

print(f"\n结果已保存到：{SAVE_DIR}")

print("\nXGBoost重要性前20：")
print(imp_df.head(20))