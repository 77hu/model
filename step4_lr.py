from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
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

# 只保留建模必需列
df = df[["code", "name", "year"] + feature_cols + [target_col]].copy()

# ==================================
# 3. 划分训练/验证/测试集
#    用 t 年特征预测 t+1 年标签
# ==================================
train_df = df[df["year"] <= 2020].copy()   # 2012-2020 -> 预测2013-2021
valid_df = df[df["year"] == 2021].copy()   # 2021 -> 预测2022
test_df  = df[df["year"] >= 2022].copy()   # 2022-2023 -> 预测2023-2024

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
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ]
)

# ==================================
# 5. 逻辑回归模型
#    类别极不平衡，所以用 balanced
# ==================================
lr_model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
        random_state=42
    ))
])

lr_model.fit(X_train, y_train)

# ==================================
# 6. 阈值搜索（在验证集上找最优F1阈值）
# ==================================
valid_prob = lr_model.predict_proba(X_valid)[:, 1]

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
# 8. 在验证集、测试集上评价
# ==================================
valid_res, valid_pred_df = evaluate_model("LR_valid", lr_model, X_valid, y_valid, best_thr)
test_res, test_pred_df = evaluate_model("LR_test", lr_model, X_test, y_test, best_thr)

# ==================================
# 9. 导出结果
# ==================================
SAVE_DIR = Path(r"C:\Users\16286\Desktop\lr_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

thr_df.to_csv(SAVE_DIR / "lr_threshold_search.csv", index=False, encoding="utf-8-sig")
valid_pred_df.to_csv(SAVE_DIR / "lr_valid_pred.csv", index=False, encoding="utf-8-sig")
test_pred_df.to_csv(SAVE_DIR / "lr_test_pred.csv", index=False, encoding="utf-8-sig")

summary_df = pd.DataFrame([valid_res, test_res])
summary_df.to_csv(SAVE_DIR / "lr_summary.csv", index=False, encoding="utf-8-sig")

# ==================================
# 10. 导出系数表
# ==================================
feature_names = lr_model.named_steps["prep"].get_feature_names_out()
coef = lr_model.named_steps["clf"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": coef,
    "abs_coef": np.abs(coef)
}).sort_values("abs_coef", ascending=False)

coef_df.to_csv(SAVE_DIR / "lr_coefficients.csv", index=False, encoding="utf-8-sig")

print(f"\n结果已保存到：{SAVE_DIR}")

print("\n逻辑回归系数前20：")
print(coef_df.head(20))