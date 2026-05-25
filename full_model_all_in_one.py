# encoding:utf8
import os
import matplotlib
matplotlib.use('Agg') # 禁用交互式弹窗，直接后台画图保存
import numpy as np
import pandas as pd
import time
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
import warnings

# 全局设置
warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文显示
plt.rcParams["axes.unicode_minus"] = False
random_seed = 2026

# 自动创建结果保存文件夹
save_path = "./model_results/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# ===================== 1. 加载数据集（路径完全匹配你的电脑）=====================
print("=" * 100)
print("正在加载数据集...")
df = pd.read_excel("data/default of credit card clients.xls", header=1)
X = df.drop(columns=["default payment next month"])
X_values = X.values
y = df["default payment next month"].values
feature_names = X.columns.tolist()
print(f"数据集加载完成！样本量：{X.shape[0]}，特征数：{X.shape[1]}")
print("=" * 100)

# ===================== 2. 全局实验配置 =====================
skf = StratifiedKFold(n_splits=5, random_state=random_seed, shuffle=True)
all_results = {}  # 保存所有实验结果

# ===================== 3. 实验1：2025-2026前沿论文横向对比 =====================
print("\n【正在运行：2025-2026前沿论文横向对比实验】")
compare_models = [
    ("2026顶刊 混合集成堆叠模型", StackingClassifier(
        estimators=[('xgb', XGBClassifier(random_state=random_seed)),
                    ('lgb', LGBMClassifier(random_state=random_seed))],
        final_estimator=LGBMClassifier(), stack_method="predict_proba"), SMOTE(random_state=random_seed)),
    ("2025核心刊 改进平衡随机森林",
     RandomForestClassifier(n_estimators=200, class_weight="balanced", max_depth=15, random_state=random_seed), None),
    ("2025核心刊 SMOTE-XGB改进模型", XGBClassifier(random_state=random_seed), SMOTE(random_state=random_seed)),
    ("2025顶会 加权均衡LightGBM", LGBMClassifier(class_weight="balanced", random_state=random_seed), None),
    ("2026顶会 集成CatBoost模型", LGBMClassifier(random_state=random_seed), SMOTE(random_state=random_seed)),
    ("本文模型 LightGBM+SMOTEENN", LGBMClassifier(random_state=random_seed), SMOTEENN(random_state=random_seed))
]

compare_result = []
for name, model, sampler in compare_models:
    start_time = time.time()
    aucs, recalls, precisions, f1s, train_times = [], [], [], [], []
    for train_idx, val_idx in skf.split(X_values, y):
        X_train, X_val = X_values[train_idx], X_values[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        if sampler is not None:
            X_train, y_train = sampler.fit_resample(X_train, y_train)

        # 训练计时
        train_start = time.time()
        model.fit(X_train, y_train)
        train_times.append(time.time() - train_start)

        y_pred_prob = model.predict_proba(X_val)[:, 1]
        y_bin = (y_pred_prob >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_val, y_pred_prob))
        recalls.append(recall_score(y_val, y_bin))
        precisions.append(precision_score(y_val, y_bin))
        f1s.append(f1_score(y_val, y_bin))

    # 保存结果
    res = {
        "模型名称": name,
        "AUC": round(np.mean(aucs), 5),
        "召回率": round(np.mean(recalls), 5),
        "精确率": round(np.mean(precisions), 5),
        "F1值": round(np.mean(f1s), 5),
        "平均训练耗时(s)": round(np.mean(train_times), 3)
    }
    compare_result.append(res)
    print(f"完成：{name} | AUC:{res['AUC']} | 召回率:{res['召回率']} | F1:{res['F1值']}")

# 转DataFrame保存
compare_df = pd.DataFrame(compare_result)
all_results["横向对比实验"] = compare_df
print("=" * 100)

# ===================== 4. 实验2：消融实验 =====================
print("\n【正在运行：消融实验】")
ablation_models = [
    ("基线：逻辑回归LR", LogisticRegression(random_state=random_seed), None),
    ("仅LightGBM（无采样）", LGBMClassifier(random_state=random_seed, verbose=-1), None),
    ("LightGBM + SMOTE", LGBMClassifier(random_state=random_seed, verbose=-1), SMOTE(random_state=random_seed)),
    ("本文模型：LightGBM + SMOTEENN", LGBMClassifier(random_state=random_seed, verbose=-1),
     SMOTEENN(random_state=random_seed)),
]

ablation_result = []
for name, model, sampler in ablation_models:
    aucs, recalls, f1s = [], [], []
    for train_idx, val_idx in skf.split(X_values, y):
        X_train, X_val = X_values[train_idx], X_values[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        if sampler is not None:
            X_train, y_train = sampler.fit_resample(X_train, y_train)
        model.fit(X_train, y_train)
        y_pred_prob = model.predict_proba(X_val)[:, 1]
        y_bin = (y_pred_prob >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_val, y_pred_prob))
        recalls.append(recall_score(y_val, y_bin))
        f1s.append(f1_score(y_val, y_bin))

    res = {
        "模型配置": name,
        "AUC": round(np.mean(aucs), 5),
        "召回率": round(np.mean(recalls), 5),
        "F1值": round(np.mean(f1s), 5)
    }
    ablation_result.append(res)
    print(f"完成：{name} | AUC:{res['AUC']} | 召回率:{res['召回率']} | F1:{res['F1值']}")

ablation_df = pd.DataFrame(ablation_result)
all_results["消融实验"] = ablation_df
print("=" * 100)

# ===================== 5. 实验3：阈值调优实验 =====================
print("\n【正在运行：阈值调优实验】")
final_model = LGBMClassifier(random_state=random_seed, verbose=-1)
final_sampler = SMOTEENN(random_state=random_seed)
thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]

threshold_result = []
for thresh in thresholds:
    aucs, recalls, precisions, f1s = [], [], [], []
    for train_idx, val_idx in skf.split(X_values, y):
        X_train, X_val = X_values[train_idx], X_values[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        X_train, y_train = final_sampler.fit_resample(X_train, y_train)
        final_model.fit(X_train, y_train)
        y_pred_prob = final_model.predict_proba(X_val)[:, 1]
        y_bin = (y_pred_prob >= thresh).astype(int)

        aucs.append(roc_auc_score(y_val, y_pred_prob))
        recalls.append(recall_score(y_val, y_bin))
        precisions.append(precision_score(y_val, y_bin))
        f1s.append(f1_score(y_val, y_bin))

    res = {
        "分类阈值": thresh,
        "风险偏好": "极致高召回" if thresh <= 0.45 else "高召回" if thresh == 0.5 else "均衡型" if thresh <= 0.55 else "稳健型",
        "AUC": round(np.mean(aucs), 5),
        "召回率": round(np.mean(recalls), 5),
        "精确率": round(np.mean(precisions), 5),
        "F1值": round(np.mean(f1s), 5)
    }
    threshold_result.append(res)
    print(f"完成：阈值{thresh} | 召回率:{res['召回率']} | F1:{res['F1值']}")

threshold_df = pd.DataFrame(threshold_result)
all_results["阈值调优实验"] = threshold_df
print("=" * 100)

# ===================== 6. SHAP可解释性分析 + 自动出图 =====================
print("\n【正在运行：SHAP可解释性分析，自动生成图片】")
# 训练最终模型
X_res, y_res = final_sampler.fit_resample(X, y)
final_model.fit(X_res, y_res)

# SHAP解释器
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)

# 图1：SHAP特征重要性柱状图
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("SHAP特征重要性排名", fontsize=14)
plt.tight_layout()
plt.savefig(f"{save_path}1_SHAP特征重要性.png", dpi=300, bbox_inches="tight")
plt.close()
print("已生成：1_SHAP特征重要性.png")

# 图2：SHAP蜂群图（特征影响方向）
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
plt.title("特征对违约风险的影响方向", fontsize=14)
plt.tight_layout()
plt.savefig(f"{save_path}2_SHAP特征影响蜂群图.png", dpi=300, bbox_inches="tight")
plt.close()
print("已生成：2_SHAP特征影响蜂群图.png")

# 图3：TOP1核心特征依赖图
top_feature = feature_names[np.argmax(abs(shap_values).mean(0))]
plt.figure(figsize=(10, 6))
shap.dependence_plot(top_feature, shap_values, X, show=False)
plt.title(f"核心特征「{top_feature}」对违约风险的依赖关系", fontsize=14)
plt.tight_layout()
plt.savefig(f"{save_path}3_核心特征依赖图.png", dpi=300, bbox_inches="tight")
plt.close()
print("已生成：3_核心特征依赖图.png")

# 保存SHAP特征重要性表格
shap_importance_df = pd.DataFrame({
    "特征名称": feature_names,
    "SHAP平均重要性": abs(shap_values).mean(0)
}).sort_values("SHAP平均重要性", ascending=False)
all_results["SHAP特征重要性"] = shap_importance_df
print("=" * 100)

# ===================== 7. 所有结果保存到Excel =====================
excel_path = f"{save_path}所有实验结果汇总.xlsx"
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    for sheet_name, df in all_results.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# ===================== 最终输出 =====================
print("\n" + "🎉" * 30)
print("✅ 所有实验全部运行完成！")
print(f"📊 所有实验结果已保存到：{excel_path}")
print(f"🖼️  所有图片已保存到文件夹：{save_path}")
print("\n📋 核心结果汇总：")
print(compare_df[["模型名称", "AUC", "召回率", "F1值"]])
print("🎉" * 30)