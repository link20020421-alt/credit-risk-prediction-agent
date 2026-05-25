import pandas as pd
import matplotlib.pyplot as plt

# 1. 读取三个结果
lgb_res = pd.read_csv('submission.csv', header=None)
xgb_res = pd.read_csv('submission_xgb.csv', header=None)
# 假设你已经运行过融合逻辑，读取生成的 final
final_res = pd.read_csv('final_submission.csv', header=None)

# 2. 提取预测值列（第二列，索引为1）
lgb_vals = lgb_res.iloc[:, 1]
xgb_vals = xgb_res.iloc[:, 1]
final_vals = final_res.iloc[:, 1]

# 3. 开始绘图对比
plt.figure(figsize=(10, 6))

# 设置中文字体（防止乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 画分布直方图，alpha是透明度，这样重叠部分能看清
plt.hist(lgb_vals, bins=50, alpha=0.5, label='LightGBM 预测分布', color='blue')
plt.hist(xgb_vals, bins=50, alpha=0.5, label='XGBoost 预测分布', color='green')
plt.hist(final_vals, bins=50, alpha=0.3, label='融合后 (Final) 分布', color='red')

plt.xlabel('预测贷款金额')
plt.ylabel('用户数量')
plt.title('三个模型预测结果对比分布图')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

print("--- 正在弹出对比图，请查看窗口 ---")
plt.show()
