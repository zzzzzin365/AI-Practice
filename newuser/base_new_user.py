import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

time
# 1. 数据加载
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./testA_data.csv')
submit = test_df[['did']]

full_df = pd.concat([train_df, test_df], axis=0)

# 2. 时间特征工程
for df in [train_df, test_df, full_df]:
    # 转换为时间戳
    df['ts'] = pd.to_datetime(df['common_ts'], unit='ms')
    
    # 提取时间特征
    df['day'] = df['ts'].dt.day
    df['dayofweek'] = df['ts'].dt.dayofweek
    df['hour'] = df['ts'].dt.hour
    
    # 删除原始时间列
    df.drop(['ts'], axis=1, inplace=True)


############################### 简单分析
# 获取 train 和 test 中唯一的 did
train_dids = set(train_df['did'].unique())
test_dids = set(test_df['did'].unique())

# 计算交集
overlap_dids = train_dids & test_dids

# 数量统计
num_overlap = len(overlap_dids)
num_train = len(train_dids)
num_test = len(test_dids)

# 占比
ratio_in_train = num_overlap / num_train if num_train > 0 else 0
ratio_in_test = num_overlap / num_test if num_test > 0 else 0

# 输出结果
print(f"重叠 did 数量: {num_overlap}")
print(f"占 train 比例: {ratio_in_train:.4f} ({num_overlap}/{num_train})")
print(f"占 test 比例: {ratio_in_test:.4f} ({num_overlap}/{num_test})")

# 需要编码的特征列表
cat_features = [
    'device_brand', 'ntt', 'operator', 'common_country',
    'common_province', 'common_city', 'appver', 'channel',
    'os_type', 'udmap'
]
# 初始化编码器字典
label_encoders = {}

for feature in cat_features:
    # 创建编码器，将类别特征转为0-N的自然数
    le = LabelEncoder()
    
    # 合并训练集和测试集的所有类别
    all_values = pd.concat([train_df[feature], test_df[feature]]).astype(str)
    
    # 训练编码器（使用所有可能值）
    le.fit(all_values)
    
    # 保存编码器
    label_encoders[feature] = le
    
    # 应用编码
    train_df[feature] = le.transform(train_df[feature].astype(str))
    test_df[feature] = le.transform(test_df[feature].astype(str))

    # 基础特征 + 目标编码特征 + 聚合特征
features = [
    # 原始特征
    'mid', 'eid', 'device_brand', 'ntt', 'operator', 
    'common_country', 'common_province', 'common_city',
    'appver', 'channel', 'os_type', 'udmap',
    # 时间特征
    'hour', 'dayofweek', 'day', 'common_ts'
]

# 准备训练和测试数据
X_train = train_df[features]
y_train = train_df['is_new_did']
X_test = test_df[features]

# 6. F1阈值优化函数
def find_optimal_threshold(y_true, y_pred_proba):
    """寻找最大化F1分数的阈值"""
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in [0.1,0.15,0.2,0.25,0.3,0.35,0.4]:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1

# 7. 模型训练与交叉验证
import time
# 动态生成随机种子（基于当前时间）
seed = int(time.time()) % 1000000  # 取当前时间戳模一个数，避免太大
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'max_depth': '12',
    'num_leaves': 63,
    'learning_rate': 0.1,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 10,
    'verbose': -1,
    'n_jobs':8,
    'seed': seed  # 使用动态生成的 seed
}

# 五折交叉验证，使用五折构建特征时的切分规则，保证切分一致
n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
test_preds = np.zeros(len(X_test))
fold_thresholds = []
fold_f1_scores = []
models = []
oof_preds = np.zeros(len(X_train))
oof_probas = np.zeros(len(X_train))

print("\n开始模型训练...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"\n======= Fold {fold+1}/{n_folds} =======")
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # 创建数据集（指定类别特征）
    train_set = lgb.Dataset(X_tr, label=y_tr)
    val_set = lgb.Dataset(X_val, label=y_val)
    
    # 模型训练
    model = lgb.train(
        params,train_set,
        num_boost_round=1000,
        valid_sets=[train_set, val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    models.append(model)
    
    # 验证集预测
    val_pred_proba = model.predict(X_val)
    oof_probas[val_idx] = val_pred_proba
    
    # 阈值优化
    best_threshold, best_f1 = find_optimal_threshold(y_val, val_pred_proba)
    fold_thresholds.append(best_threshold)
    
    # 使用优化阈值计算F1
    val_pred_labels = (val_pred_proba >= best_threshold).astype(int)
    fold_f1 = f1_score(y_val, val_pred_labels)
    fold_f1_scores.append(fold_f1)
    oof_preds[val_idx] = val_pred_labels
    
    print(f"Fold {fold+1} Optimal Threshold: {best_threshold:.4f}")
    print(f"Fold {fold+1} F1 Score: {fold_f1:.5f}")
    
    # 测试集预测
    test_preds += model.predict(X_test) / n_folds

    # 8. 整体结果评估
# 使用交叉验证平均阈值
avg_threshold = np.mean(fold_thresholds)
final_oof_preds = (oof_probas >= avg_threshold).astype(int)
final_f1 = f1_score(y_train, final_oof_preds)

print("\n===== Final Results =====")
print(f"Average Optimal Threshold: {avg_threshold:.4f}")
print(f"Fold F1 Scores: {[f'{s:.5f}' for s in fold_f1_scores]}")
print(f"Average Fold F1: {np.mean(fold_f1_scores):.5f}")
print(f"OOF F1 Score: {final_f1:.5f}")

# 9. 测试集预测与提交文件生成
# 使用平均阈值进行预测
test_pred_labels = (test_preds >= avg_threshold).astype(int)
submit['is_new_did'] = test_pred_labels

# 保存提交文件
submit[['is_new_did']].to_csv('submit.csv', index=False)
print("\nSubmission file saved: submit.csv")
print(f"Predicted new user ratio: {test_pred_labels.mean():.4f}")
print(f"Test set size: {len(test_pred_labels)}")

# 10. 特征重要性分析
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': models[0].feature_importance(importance_type='gain')
}).sort_values('Importance', ascending=False)

print("\nTop 10 Features:")
print(feature_importance.head(10))