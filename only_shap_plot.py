# encoding:utf8
import os
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # 强制使用非GUI后端，解决报错
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from imblearn.combine import SMOTEENN
import warnings

warnings.filterwarnings("ignore")
random_seed = 2026
save_path = "./model_results/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 加载数据
print("正在加载数据...")
df = pd.read_excel("data/default of credit card clients.xls", header=1)
X = df.drop(columns=["default payment next month"])
y = df["default payment next month"].values
feature_names = X.columns.tolist()

# 训练最终模型（只训练一次，很快）
print("正在训练模型...")
sampler = SMOTEENN(random_state=random_seed)
X_res, y_res = sampler.fit_resample(X, y)
final_model = LGBMClassifier(random_state=random_seed, verbose=-1)
final_model.fit(X_res, y_res)

# SHAP解释
print("正在生成SHAP图片...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X)

# 图1：SHAP特征重要性柱状图
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
plt.title("SHAP特征重要性排名", fontsize=14)
plt.tight_layout()
plt.savefig(f"{save_path}1_SHAP特征重要性.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ 已生成：1_SHAP特征重要性.png")

# 图2：SHAP蜂群图
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, show=False)
plt.title("特征对违约风险的影响方向", fontsize=14)
plt.tight_layout()
plt.savefig(f"{save_path}2_SHAP特征影响蜂群图.png", dpi=300, bbox_inches="tight")
plt.close()
print("✅ 已生成：2_SHAP特征影响蜂群图.png")

# 保存SHAP特征重要性表格
shap_importance_df = pd.DataFrame({
    "特征名称": feature_names,
    "SHAP平均重要性": abs(shap_values).mean(0)
}).sort_values("SHAP平均重要性", ascending=False)
shap_importance_df.to_excel(f"{save_path}SHAP特征重要性.xlsx", index=False)
print("✅ 已生成：SHAP特征重要性.xlsx")

print("\n🎉 所有图片和表格生成完成！")
print(f"📂 保存位置：{save_path}")