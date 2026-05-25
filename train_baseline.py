# encoding:utf8
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings("ignore")

# 读取 UCI 数据
df = pd.read_excel("data/default of credit card clients.xls", header=1)
df.rename(columns={"default payment next month": "label"}, inplace=True)
df = df.drop(columns=["ID"])
X = df.drop(columns=["label"]).values
y = df["label"].values

# 3个对比模型
models = {
    "逻辑回归(LR)": LogisticRegression(max_iter=1000),
    "随机森林(RF)": RandomForestClassifier(random_state=2017),
    "XGBoost": XGBClassifier(random_state=2017)
}

skf = StratifiedKFold(n_splits=5, random_state=2017, shuffle=True)

print("="*60)
print("           UCI 对比模型结果")
print("="*60)

for name, model in models.items():
    aucs, recalls, f1s = [], [], []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 采样
        try:
            smote = SMOTEENN(random_state=2017)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        except:
            pass

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:,1]
        y_bin = (y_pred >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_val, y_pred))
        recalls.append(recall_score(y_val, y_bin))
        f1s.append(f1_score(y_val, y_bin))

    print(f"{name}")
    print(f"  AUC: {np.mean(aucs):.5f}")
    print(f"  召回率: {np.mean(recalls):.5f}")
    print(f"  F1: {np.mean(f1s):.5f}")
    print("-"*60)