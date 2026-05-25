# 信贷违约预测多Agent系统

基于AI Agent驱动的端到端信贷风险建模框架，解决了传统风控建模中数据泄露风险高、类别不平衡、模型黑盒不可解释等核心痛点。

## 核心功能
- 多Agent协作架构：数据预处理Agent、特征工程Agent、模型训练Agent、可解释性分析Agent
- 支持SMOTEENN混合采样，解决信贷数据类别不平衡问题
- 集成LightGBM、XGBoost、CatBoost等主流模型
- 基于SHAP博弈论的模型全局与局部可解释性分析
- 多随机种子稳定性验证，确保模型工业级鲁棒性

## 技术栈
- Python 3.9
- Scikit-learn, LightGBM, XGBoost, CatBoost
- Imbalanced-learn (SMOTEENN)
- SHAP (可解释性分析)

## 验证结果
在京东JDD、UCI信用卡、GiveMeSomeCredit三个公开数据集上验证：
- 京东JDD数据集：AUC 0.8607，召回率 0.7401
- 较无采样基线召回率提升36.5个百分点
- 多随机种子测试标准差<0.005，稳定性优异

## 项目用途
本项目为硕士论文研究成果，旨在为金融信贷风控领域提供高效、可解释、可审计的AI建模解决方案。
