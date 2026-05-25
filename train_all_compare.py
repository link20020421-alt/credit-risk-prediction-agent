# encoding:utf8
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings("ignore")

# ========== 直接加载 GiveMeSomeCredit 数据集 ==========
df = pd.read_csv("data/cs-training.csv")
df = df.drop(columns=["Unnamed: 0"]).fillna(0)
X = df.drop(columns=["SeriousDlqin2yrs"]).values
y = df["SeriousDlqin2yrs"].values

models = {
    "逻辑回归(LR)": LogisticRegression(max_iter=1000),
    "随机森林(RF)": RandomForestClassifier(random_state=2017),
    "XGBoost": XGBClassifier(random_state=2017),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=2017),
    "原生LightGBM": LGBMClassifier(random_state=2017),
    "HistGB": HistGradientBoostingClassifier(random_state=2017)
}

skf = StratifiedKFold(n_splits=5, random_state=2017, shuffle=True)
sampler = SMOTEENN(random_state=2017)

print("=" * 70)
print("        GiveMeSomeCredit 多模型对比实验结果")
print("=" * 70)

for name, model in models.items():
    aucs, recalls, f1s = [], [], []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
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
    print("-" * 70)