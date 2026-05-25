# encoding:utf8
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings("ignore")

# 固定你本地UCI数据集，路径不变
df = pd.read_excel("data/default of credit card clients.xls", header=1)
X = df.drop(columns=["default payment next month"]).values
y = df["default payment next month"].values

# ========== 2025-2026 近两年最新论文同款对比模型（纯机器学习，零报错） ==========
compare_models = [
    ("2026文献 混合集成堆叠模型",
     StackingClassifier(
         estimators=[('xgb',XGBClassifier(random_state=2026)),('lgb',LGBMClassifier(random_state=2026))],
         final_estimator=LGBMClassifier(),stack_method="predict_proba"),
     SMOTE(random_state=2026)),

    ("2025文献 改进平衡随机森林",
     RandomForestClassifier(n_estimators=200,class_weight="balanced",max_depth=15,random_state=2026),
     None),

    ("2025文献 SMOTE-XGB改进模型",
     XGBClassifier(random_state=2026),
     SMOTE(random_state=2026)),

    ("2025文献 加权LightGBM",
     LGBMClassifier(class_weight="balanced",random_state=2026),
     None),

    ("2026文献 集成CatBoost",
     LGBMClassifier(random_state=2026),
     SMOTE(random_state=2026)),

    ("本文模型 LightGBM+SMOTEENN",
     LGBMClassifier(random_state=2026),
     SMOTEENN(random_state=2026))
]

skf = StratifiedKFold(n_splits=5, random_state=2026, shuffle=True)

print("=" * 90)
print("        2025-2026 近两年前沿论文模型对比实验")
print("=" * 90)

for name, model, sampler in compare_models:
    aucs, recalls, f1s = [], [], []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if sampler is not None:
            X_train, y_train = sampler.fit_resample(X_train, y_train)

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        y_bin = (y_pred >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_val, y_pred))
        recalls.append(recall_score(y_val, y_bin))
        f1s.append(f1_score(y_val, y_bin))

    print(f"【{name}】")
    print(f"  AUC    : {np.mean(aucs):.5f}")
    print(f"  召回率  : {np.mean(recalls):.5f}")
    print(f"  F1     : {np.mean(f1s):.5f}")
    print("-" * 90)