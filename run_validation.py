import pandas as pd
import numpy as np
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')  # 忽略满屏的警告信息


def run_ablation_study(X, y):
    print("\n========== 1. 开始运行消融实验 ==========")
    # 划分 80% 训练集, 20% 测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # [基线] 1. 原始 LightGBM (无采样)
    clf_base = lgb.LGBMClassifier(random_state=42, n_estimators=100, verbosity=-1)
    clf_base.fit(X_train, y_train)
    y_pred_base = clf_base.predict(X_test)
    y_prob_base = clf_base.predict_proba(X_test)[:, 1]
    print(
        f"[无采样]   AUC: {roc_auc_score(y_test, y_prob_base):.4f} | Recall: {recall_score(y_test, y_pred_base):.4f} | F1: {f1_score(y_test, y_pred_base):.4f}")

    # [对比] 2. LightGBM + SMOTE (仅过采样)
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train, y_train)
    clf_smote = lgb.LGBMClassifier(random_state=42, n_estimators=100, verbosity=-1)
    clf_smote.fit(X_smote, y_smote)
    y_pred_smote = clf_smote.predict(X_test)
    y_prob_smote = clf_smote.predict_proba(X_test)[:, 1]
    print(
        f"[仅SMOTE]  AUC: {roc_auc_score(y_test, y_prob_smote):.4f} | Recall: {recall_score(y_test, y_pred_smote):.4f} | F1: {f1_score(y_test, y_pred_smote):.4f}")

    # [本文] 3. LightGBM + SMOTEENN (混合采样)
    smoteenn = SMOTEENN(random_state=42)
    X_sme, y_sme = smoteenn.fit_resample(X_train, y_train)
    clf_sme = lgb.LGBMClassifier(random_state=42, n_estimators=100, verbosity=-1)
    clf_sme.fit(X_sme, y_sme)
    y_pred_sme = clf_sme.predict(X_test)
    y_prob_sme = clf_sme.predict_proba(X_test)[:, 1]
    print(
        f"[SMOTEENN] AUC: {roc_auc_score(y_test, y_prob_sme):.4f} | Recall: {recall_score(y_test, y_pred_sme):.4f} | F1: {f1_score(y_test, y_pred_sme):.4f}")


def run_5fold_cv(X, y):
    print("\n========== 2. 开始运行 5 折交叉验证 (稳定性测试) ==========")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores, recall_scores, f1_scores = [], [], []

    # 转换为 NumPy 数组以便使用索引切片
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y

    fold = 1
    for train_idx, test_idx in skf.split(X_arr, y_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        # 注意：只在训练集上做 SMOTEENN 采样，测试集必须保持原始分布，绝对不能污染！
        smoteenn = SMOTEENN(random_state=42)
        X_resampled, y_resampled = smoteenn.fit_resample(X_train, y_train)

        clf = lgb.LGBMClassifier(random_state=42, n_estimators=100, verbosity=-1)
        clf.fit(X_resampled, y_resampled)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Fold {fold} - AUC: {auc:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        auc_scores.append(auc)
        recall_scores.append(recall)
        f1_scores.append(f1)
        fold += 1

    print("\n--- 5折交叉验证 最终统计 ---")
    print(f"AUC     平均值: {np.mean(auc_scores):.4f} | 标准差: {np.std(auc_scores):.4f}")
    print(f"Recall  平均值: {np.mean(recall_scores):.4f} | 标准差: {np.std(recall_scores):.4f}")
    print(f"F1      平均值: {np.mean(f1_scores):.4f} | 标准差: {np.std(f1_scores):.4f}\n")


if __name__ == "__main__":
    # =====================================================================
    # ！！！你只需要修改下面这两行！！！
    # =====================================================================

    # 修改 1：把 'data.csv' 换成你真实的数据集文件名（例如 'JDD_cleaned.csv'）
    # 如果数据不在同一个文件夹，写绝对路径，例如 'D:/Jupyter/2017JDD/data.csv'
    file_path = 'data.csv'
    print(f"正在读取数据: {file_path} ... (可能需要几秒钟)")
    df = pd.read_csv(file_path)

    # 修改 2：把 'target' 换成你数据表中代表“是否违约”的那个列名（例如 'label', 'is_default' 等）
    target_column = 'target'

    # =====================================================================

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # 开始执行
    run_ablation_study(X, y)
    run_5fold_cv(X, y)
    print("========== 所有实验运行完毕 ==========")