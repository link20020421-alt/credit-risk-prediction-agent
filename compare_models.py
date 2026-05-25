# encoding:utf8
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

# 机器学习模型
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier  # 这里是正确导入
from lightgbm import LGBMClassifier

# 工具
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score
from imblearn.combine import SMOTEENN

# ====================== 加载你的数据 ======================
train_file = 'D:/Jupyter/2017JDD-Loan_Forecasting_Qualification-master/data/training2.pkl'
data_set = pickle.load(open(train_file, 'rb'))
data_set.fillna(0., inplace=True)

label = np.where(data_set['label'].values >= 0.5, 1, 0).astype(int)
feature_list = list(data_set.columns)
feature_list.remove('uid')
feature_list.remove('label')
X = data_set[feature_list].values
y = label

# ====================== 定义所有对比模型 ======================
models = {
    "逻辑回归": LogisticRegression(max_iter=1000),
    "随机森林": RandomForestClassifier(n_estimators=200, random_state=2017),
    "SVM": SVC(probability=True, random_state=2017),
    "XGBoost": XGBClassifier(random_state=2017),  # 这里已经修正为 XGBClassifier
    "LightGBM(基线)": LGBMClassifier(random_state=2017),
}

# ====================== 开始跑所有对比模型 ======================
results = []

print("=" * 60)
print("           开始运行所有对比模型（5折验证）")
print("=" * 60)

for name, model in models.items():
    print(f"\n🚀 正在运行模型：【{name}】")
    auc_list, recall_list, f1_list = [], [], []

    for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=2017).split(X, y), 1):
        print(f"   ↳ {name} 第 {fold} 折 🏃")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        auc = roc_auc_score(y_val, y_pred_proba)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)

        auc_list.append(auc)
        recall_list.append(recall)
        f1_list.append(f1)

    auc_avg = np.mean(auc_list)
    recall_avg = np.mean(recall_list)
    f1_avg = np.mean(f1_list)

    results.append([name, round(auc_avg, 4), round(recall_avg, 4), round(f1_avg, 4)])
    print(f"\n✅ {name:12s} | AUC={auc_avg:.4f} | 召回率={recall_avg:.4f} | F1={f1_avg:.4f}")

# ====================== 运行你的最终模型（LightGBM+SMOTEENN）======================
print("\n" + "=" * 60)
print("           运行本文模型：LightGBM + SMOTEENN")
print("=" * 60)

model_final = LGBMClassifier(random_state=2017)
auc_list, recall_list, f1_list = [], [], []

for fold, (train_idx, val_idx) in enumerate(StratifiedKFold(n_splits=5, shuffle=True, random_state=2017).split(X, y), 1):
    print(f"\n🚀 本文模型 第 {fold} 折 🏃")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    smote = SMOTEENN(random_state=2017)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    model_final.fit(X_train, y_train)
    y_pred_proba = model_final.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_val, y_pred_proba)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    auc_list.append(auc)
    recall_list.append(recall)
    f1_list.append(f1)

auc_avg = np.mean(auc_list)
recall_avg = np.mean(recall_list)
f1_avg = np.mean(f1_list)

results.append(["本文模型(LGB+SMOTEENN)", round(auc_avg, 4), round(recall_avg, 4), round(f1_avg, 4)])

print("\n✅ 本文模型最终结果：")
print(f"   AUC    = {auc_avg:.5f}")
print(f"   召回率  = {recall_avg:.5f}")
print(f"   F1     = {f1_avg:.5f}")

# ====================== 输出最终对比表 ======================
print("\n" + "=" * 60)
print("                   模型对比总表")
print("=" * 60)
for r in results:
    print(f"{r[0]:20s} | AUC={r[1]:.4f} | 召回率={r[2]:.4f} | F1={r[3]:.4f}")
print("=" * 60)
