# encoding:utf8
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
import warnings

warnings.filterwarnings("ignore")

# 数据集路径和之前完全一致，不用改
df = pd.read_excel("data/default of credit card clients.xls", header=1)
X = df.drop(columns=["default payment next month"]).values
y = df["default payment next month"].values

# 消融实验：拆解每个模块的贡献
ablation_models = [
    ("基线：逻辑回归LR", LogisticRegression(random_state=2026), None),
    ("仅LightGBM（无采样）", LGBMClassifier(random_state=2026), None),
    ("LightGBM + SMOTE", LGBMClassifier(random_state=2026), SMOTE(random_state=2026)),
    ("本文模型：LightGBM + SMOTEENN", LGBMClassifier(random_state=2026), SMOTEENN(random_state=2026)),
]

skf = StratifiedKFold(n_splits=5, random_state=2026, shuffle=True)

print("=" * 80)
print("                消融实验结果")
print("=" * 80)

for name, model, sampler in ablation_models:
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
    print("-" * 80)