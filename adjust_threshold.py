# encoding:utf8
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTEENN
import warnings

warnings.filterwarnings("ignore")

# 数据集（路径写死，不用改）
df = pd.read_excel("data/default of credit card clients.xls", header=1)
X = df.drop(columns=["default payment next month"]).values
y = df["default payment next month"].values

# 用你第一次的最优模型（召回率最高的那个）
model = LGBMClassifier(random_state=2026)
sampler = SMOTEENN(random_state=2026)
skf = StratifiedKFold(n_splits=5, random_state=2026, shuffle=True)

print("=" * 80)
print("【阈值调优：不同风险偏好下的模型性能】")
print("=" * 80)

# 测试三个不同阈值，对应银行不同的风险偏好
thresholds = [0.5, 0.55, 0.6]
for thresh in thresholds:
    aucs, recalls, precisions, f1s = [], [], [], []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        X_train, y_train = sampler.fit_resample(X_train, y_train)
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_val)[:, 1]
        y_bin = (y_pred_prob >= thresh).astype(int)

        aucs.append(roc_auc_score(y_val, y_pred_prob))
        recalls.append(recall_score(y_val, y_bin))
        precisions.append(precision_score(y_val, y_bin))
        f1s.append(f1_score(y_val, y_bin))

    print(f"▶ 阈值 = {thresh}")
    print(f"  AUC    : {np.mean(aucs):.5f}")
    print(f"  召回率  : {np.mean(recalls):.5f}")
    print(f"  精确率  : {np.mean(precisions):.5f}")
    print(f"  F1     : {np.mean(f1s):.5f}")
    print("-" * 80)