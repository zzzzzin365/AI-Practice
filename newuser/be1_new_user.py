import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class NewUserPredictor:
    """新用户预测器"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.models = []
        self.feature_names = []
        
    def load_data(self, train_path='./train.csv', test_path='./testA_data.csv'):
        """数据加载函数"""
        print("正在加载数据...")
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.submit = self.test_df[['did']].copy()
 
        self.full_df = pd.concat([self.train_df, self.test_df], axis=0, ignore_index=True)
        print(f"数据加载完成 - 训练集: {len(self.train_df)}, 测试集: {len(self.test_df)}")
        
    def data_analysis(self):
        """数据分析 - 了解数据特点"""
        print("\n数据分析中...")

        train_dids = set(self.train_df['did'].unique())
        test_dids = set(self.test_df['did'].unique())
        overlap_dids = train_dids & test_dids
        
        print(f"训练集唯一did数量: {len(train_dids)}")
        print(f"测试集唯一did数量: {len(test_dids)}")
        print(f"重叠did数量: {len(overlap_dids)}")
        print(f"重叠比例: {len(overlap_dids)/len(test_dids):.4f}")
        
        # 分析目标变量分布
        new_user_ratio = self.train_df['is_new_did'].mean()
        print(f"新用户比例: {new_user_ratio:.4f}")
        
    def feature_engineering(self):
        """特征工程 - 创建有用的特征"""
        print("\n特征工程中...")
    
        categorical_cols = ['device_brand', 'ntt', 'operator', 'common_country',
                           'common_province', 'common_city', 'appver', 'channel',
                           'os_type', 'udmap']
        
        for col in categorical_cols:
            if col in self.train_df.columns:
                self.train_df[col] = self.train_df[col].astype(str)
            if col in self.test_df.columns:
                self.test_df[col] = self.test_df[col].astype(str)
            if col in self.full_df.columns:
                self.full_df[col] = self.full_df[col].astype(str)
        
        # 1. 时间特征
        for df in [self.train_df, self.test_df, self.full_df]:
            df['ts'] = pd.to_datetime(df['common_ts'], unit='ms')
            df['hour'] = df['ts'].dt.hour
            df['dayofweek'] = df['ts'].dt.dayofweek
            df['day'] = df['ts'].dt.day
            df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
            
            # 时间段划分 - 直接使用字符串而不是Categorical
            df['time_period'] = pd.cut(df['hour'], 
                                     bins=[0, 6, 12, 18, 24], 
                                     labels=['夜间', '上午', '下午', '晚上']).astype(str)
            
            # 2. 设备特征组合
            df['device_os'] = df['device_brand'].astype(str) + '_' + df['os_type'].astype(str)
            df['location'] = df['common_province'].astype(str) + '_' + df['common_city'].astype(str)
            
        # 3. 统计特征 - 基于全量数据
        print("创建统计特征...")
        
        # 确保full_df中的关键列是字符串类型
        self.full_df['did'] = self.full_df['did'].astype(str)
        self.full_df['device_brand'] = self.full_df['device_brand'].astype(str)
        self.full_df['common_city'] = self.full_df['common_city'].astype(str)
        
        # did相关统计
        did_stats = self.full_df.groupby('did').agg({
            'mid': 'count',  # 该did的活跃次数
            'eid': 'nunique',  # 该did的唯一事件数
            'hour': ['min', 'max', 'std'],  # 活跃时间特征
            'common_ts': lambda x: x.max() - x.min() if len(x) > 0 else 0  # 时间跨度
        }).reset_index()
        
        # 扁平化列名
        did_stats.columns = ['did', 'did_activity_count', 'did_unique_events', 
                           'did_hour_min', 'did_hour_max', 'did_hour_std', 'did_timespan']
        
        # 设备品牌统计
        brand_stats = self.full_df.groupby('device_brand').agg({
            'did': 'nunique',
            'mid': 'count'
        }).reset_index()
        brand_stats.columns = ['device_brand', 'brand_unique_users', 'brand_total_activity']
        
        # 地理位置统计
        city_stats = self.full_df.groupby('common_city').agg({
            'did': 'nunique',
            'mid': 'count'
        }).reset_index()
        city_stats.columns = ['common_city', 'city_unique_users', 'city_total_activity']
        
        # 确保合并前的key列是相同类型
        self.train_df['did'] = self.train_df['did'].astype(str)
        self.train_df['device_brand'] = self.train_df['device_brand'].astype(str)
        self.train_df['common_city'] = self.train_df['common_city'].astype(str)
        
        self.test_df['did'] = self.test_df['did'].astype(str)
        self.test_df['device_brand'] = self.test_df['device_brand'].astype(str)
        self.test_df['common_city'] = self.test_df['common_city'].astype(str)
        
        # 合并统计特征 - 修复：正确更新原始dataframe
        self.train_df = self.train_df.merge(did_stats, on='did', how='left')
        self.train_df = self.train_df.merge(brand_stats, on='device_brand', how='left')
        self.train_df = self.train_df.merge(city_stats, on='common_city', how='left')
        
        self.test_df = self.test_df.merge(did_stats, on='did', how='left')
        self.test_df = self.test_df.merge(brand_stats, on='device_brand', how='left')
        self.test_df = self.test_df.merge(city_stats, on='common_city', how='left')
        
        # 填充缺失值
        self.train_df = self.train_df.fillna(0)
        self.test_df = self.test_df.fillna(0)
            
        # 4. 标签编码
        print("标签编码中...")
        cat_features = [
            'device_brand', 'ntt', 'operator', 'common_country',
            'common_province', 'common_city', 'appver', 'channel',
            'os_type', 'udmap', 'time_period', 'device_os', 'location'
        ]
        
        for feature in cat_features:
            if feature in self.train_df.columns and feature in self.test_df.columns:
                try:
                    le = LabelEncoder()
                    
                    # 确保数据是字符串类型
                    train_values = self.train_df[feature].astype(str).fillna('unknown')
                    test_values = self.test_df[feature].astype(str).fillna('unknown')
                    
                    # 获取所有可能的值
                    all_values = pd.concat([train_values, test_values]).unique()
                    
                    # 拟合编码器
                    le.fit(all_values)
                    self.label_encoders[feature] = le
                    
                    # 转换数据
                    self.train_df[feature] = le.transform(train_values)
                    self.test_df[feature] = le.transform(test_values)
                    
                    print(f"   {feature}: {len(all_values)} 个唯一值")
                    
                except Exception as e:
                    print(f"   警告: {feature} 编码失败: {str(e)}")
                    # 如果编码失败，保持原始数值
                    if self.train_df[feature].dtype == 'object':
                        self.train_df[feature] = pd.factorize(self.train_df[feature])[0]
                    if self.test_df[feature].dtype == 'object':
                        self.test_df[feature] = pd.factorize(self.test_df[feature])[0]
            
        # 删除临时列
        for df in [self.train_df, self.test_df]:
            if 'ts' in df.columns:
                df.drop(['ts'], axis=1, inplace=True)
            
        print("特征工程完成")
        
    def prepare_features(self):
        """准备最终特征列表"""
        potential_features = [
            # 基础特征
            'mid', 'eid', 'device_brand', 'ntt', 'operator', 
            'common_country', 'common_province', 'common_city',
            'appver', 'channel', 'os_type', 'udmap',
            # 时间特征
            'hour', 'dayofweek', 'day', 'is_weekend', 'time_period',
            'common_ts',
            # 组合特征
            'device_os', 'location',
            # 统计特征
            'did_activity_count', 'did_unique_events', 'did_hour_min', 
            'did_hour_max', 'did_hour_std', 'did_timespan',
            'brand_unique_users', 'brand_total_activity',
            'city_unique_users', 'city_total_activity'
        ]
        
        # 只保留存在的特征
        self.feature_names = [f for f in potential_features if f in self.train_df.columns]
        
        # 确保测试集也有这些特征
        self.feature_names = [f for f in self.feature_names if f in self.test_df.columns]
        
        print(f"最终特征数量: {len(self.feature_names)}")
        print(f"特征列表: {self.feature_names}")
        
    def find_optimal_threshold(self, y_true, y_pred_proba):
        """寻找最优阈值"""
        best_threshold = 0.5
        best_f1 = 0
        
        # 扩大搜索范围，更细致
        for threshold in np.arange(0.1, 0.6, 0.02):
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        return best_threshold, best_f1
    
    def train_lgb_model(self, X_train, y_train, X_val, y_val):
        """训练LightGBM模型"""
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'max_depth': 8,
            'num_leaves': 64,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'verbose': -1,
            'seed': self.random_state
        }
        
        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val)
        
        model = lgb.train(
            params, train_set,
            num_boost_round=1000,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0)
            ]
        )
        
        return model
    
    def train_xgb_model(self, X_train, y_train, X_val, y_val):
        """训练XGBoost模型"""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 10,
            'seed': self.random_state,
            'verbosity': 0
        }
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        model = xgb.train(
            params, dtrain,
            num_boost_round=1000,
            evals=[(dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=0
        )
        
        return model
    
    def train_models(self):
        """训练多个模型并融合"""
        print("\n开始模型训练...")
        
        X_train = self.train_df[self.feature_names]
        y_train = self.train_df['is_new_did']
        X_test = self.test_df[self.feature_names]
        
        print(f"训练集特征形状: {X_train.shape}")
        print(f"测试集特征形状: {X_test.shape}")
        
        # 检查是否有无穷大或NaN值
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_test = X_test.replace([np.inf, -np.inf], 0)
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        # 交叉验证
        n_folds = 5
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
        
        # 存储预测结果
        lgb_test_preds = np.zeros(len(X_test))
        xgb_test_preds = np.zeros(len(X_test))
        
        lgb_oof_preds = np.zeros(len(X_train))
        xgb_oof_preds = np.zeros(len(X_train))
        
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            print(f"训练第 {fold+1}/{n_folds} 折...")
            
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # 训练LightGBM
            lgb_model = self.train_lgb_model(X_tr, y_tr, X_val, y_val)
            lgb_val_pred = lgb_model.predict(X_val)
            lgb_oof_preds[val_idx] = lgb_val_pred
            lgb_test_preds += lgb_model.predict(X_test) / n_folds
            
            # 训练XGBoost
            xgb_model = self.train_xgb_model(X_tr, y_tr, X_val, y_val)
            xgb_val_pred = xgb_model.predict(xgb.DMatrix(X_val))
            xgb_oof_preds[val_idx] = xgb_val_pred
            xgb_test_preds += xgb_model.predict(xgb.DMatrix(X_test)) / n_folds
            
            # 模型融合预测
            ensemble_val_pred = 0.6 * lgb_val_pred + 0.4 * xgb_val_pred
            
            # 寻找最优阈值
            threshold, f1 = self.find_optimal_threshold(y_val, ensemble_val_pred)
            fold_scores.append(f1)
            
            print(f"   第{fold+1}折 F1: {f1:.5f}, 阈值: {threshold:.4f}")
            
        # 最终预测
        self.ensemble_oof_preds = 0.6 * lgb_oof_preds + 0.4 * xgb_oof_preds
        self.ensemble_test_preds = 0.6 * lgb_test_preds + 0.4 * xgb_test_preds
        
        # 全局最优阈值
        self.best_threshold, self.best_f1 = self.find_optimal_threshold(y_train, self.ensemble_oof_preds)
        
        print(f"\n训练完成!")
        print(f"平均F1分数: {np.mean(fold_scores):.5f}")
        print(f"最终F1分数: {self.best_f1:.5f}")
        print(f"最优阈值: {self.best_threshold:.4f}")
        
    def make_prediction(self):
        """生成最终预测"""
        print("\n生成预测结果...")
        
        # 使用最优阈值
        test_pred_labels = (self.ensemble_test_preds >= self.best_threshold).astype(int)
        self.submit['is_new_did'] = test_pred_labels
        
        # 保存结果
        self.submit.to_csv('submit_optimized.csv', index=False)
        
        print(f"预测完成!")
        print(f"预测新用户比例: {test_pred_labels.mean():.4f}")
        print(f"预测文件已保存: submit_optimized.csv")
        
    def run_full_pipeline(self):
        """运行完整流程"""
        print("开始新用户预测项目")
        print("=" * 50)
        
        try:
            # 1. 数据加载
            self.load_data()
            
            # 2. 数据分析
            self.data_analysis()
            
            # 3. 特征工程
            self.feature_engineering()
            
            # 4. 准备特征
            self.prepare_features()
            
            # 5. 模型训练
            self.train_models()
            
            # 6. 生成预测
            self.make_prediction()
            
            print("\n项目完成!")
            print("=" * 50)
            
        except Exception as e:
            print(f"执行过程中出现错误: {str(e)}")
            import traceback
            traceback.print_exc()

# 使用示例
if __name__ == "__main__":
    # 创建预测器实例
    predictor = NewUserPredictor(random_state=42)
    
    # 运行完整流程
    predictor.run_full_pipeline()