import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import tensorflow as tf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import matplotlib

# 设置字体，能够显示中文
matplotlib.rc("font", family='宋体', weight="bold")

DIANZHAN_ID: int = int(input("请输入电站ID: "))

# 配置环境
plt.style.use('seaborn-whitegrid')
sns.set_palette('coolwarm')
np.random.seed(42)
tf.random.set_random_seed(42)

# 1. 增强型数据预处理
def enhanced_preprocessing(df):
    # 复制数据避免修改原始数据
    df = df.copy()
    
    # 转换时间列
    df['时间'] = pd.to_datetime(df['时间'])
    
    # 预先创建时间序列特征列（初始化为0）
    df['季节分量'] = 0.0
    df['趋势分量'] = 0.0
    df['残差分量'] = 0.0
    
    # 计算小时发电量
    df['小时发电量'] = df.groupby(df['时间'].dt.date)['估计累计发电量kwh'].diff().fillna(df['估计累计发电量kwh'])
    
    # 创建时间特征
    df['年'] = df['时间'].dt.year
    df['月'] = df['时间'].dt.month
    df['日'] = df['时间'].dt.day
    df['时'] = df['时间'].dt.hour
    df['一年中第几天'] = df['时间'].dt.dayofyear
    df['一周中第几天'] = df['时间'].dt.dayofweek
    
    # 计算太阳位置特征
    df['日出时间'] = pd.to_datetime(df['日出时间'])
    df['日落时间'] = pd.to_datetime(df['日落时间'])
    
    df['日出时差(小时)'] = (df['日出时间'] - df['时间']).dt.total_seconds() / 3600
    df['日落时差(小时)'] = (df['日落时间'] - df['时间']).dt.total_seconds() / 3600
    
    # 计算日光持续时间和有效光照时间
    daylight_duration = (df['日落时间'] - df['日出时间']).dt.total_seconds() / 3600
    # 避免除以0
    daylight_duration = daylight_duration.replace(0, 1)
    
    # 计算太阳高度因子，避免无效值
    hour_from_sunrise = (df['时间'] - df['日出时间']).dt.total_seconds() / 3600
    hour_from_sunrise = np.clip(hour_from_sunrise, 0, daylight_duration)
    df['太阳高度因子'] = np.sin(np.pi * hour_from_sunrise / daylight_duration)
    
    # 添加假日特征
    try:
        cn_holidays = holidays.CountryHoliday('CN')
        df['是否假日'] = df['时间'].dt.date.apply(lambda d: 1 if d in cn_holidays else 0)
    except:
        df['是否假日'] = 0
    
    # 添加滑动窗口统计特征
    for window in [6, 24, 72]:  # 6小时、24小时、72小时窗口
        df[f'辐照强度_{window}h_均值'] = df['估计辐照强度w/m2'].rolling(window, min_periods=1).mean()
        df[f'温度_{window}h_均值'] = df['当前温度'].rolling(window, min_periods=1).mean()
    
    # 创建交互特征
    df['辐照-温度_交互'] = df['估计辐照强度w/m2'] * df['当前温度']
    df['辐照-湿度_交互'] = df['估计辐照强度w/m2'] * df['湿度']
    df['温度-湿度_交互'] = df['当前温度'] * df['湿度']
    
    # 天气状况编码
    weather_map = {'晴': 1.0, '多云': 0.8, '阴': 0.6, '雾': 0.4, '雨': 0.2, '雪': 0.0}
    df['天气编码'] = df['天气'].map(weather_map).fillna(0.5)
    
    # 风向编码
    wind_dir_map = {'北风': 0, '东北风': 45, '东风': 90, '东南风': 135, 
                   '南风': 180, '西南风': 225, '西风': 270, '西北风': 315}
    df['风向角度'] = df['风向'].map(wind_dir_map).fillna(0)
    
    # 仅在有足够数据的情况下尝试季节分解
    if len(df) > 48:  # 至少需要48小时数据
        try:
            # 季节分解（简化版，不使用分组）
            ts = df.set_index('时间')['小时发电量'].asfreq('H').fillna(0)
            
            if len(ts) > 24 * 2:  # 至少需要48小时数据
                decomposition = seasonal_decompose(ts, model='additive', period=24)
                
                # 确保分解结果与原始数据长度一致
                seasonal_vals = decomposition.seasonal.values[:len(df)]
                trend_vals = decomposition.trend.values[:len(df)]
                resid_vals = decomposition.resid.values[:len(df)]
                
                # 填充结果
                df['季节分量'] = seasonal_vals if len(seasonal_vals) == len(df) else np.zeros(len(df))
                df['趋势分量'] = trend_vals if len(trend_vals) == len(df) else np.zeros(len(df))
                df['残差分量'] = resid_vals if len(resid_vals) == len(df) else np.zeros(len(df))
        except Exception as e:
            print(f"季节分解失败: {str(e)}")
    
    # 选择有光照数据（7:00-18:00且辐照>0）
    daylight_mask = (df['时'] >= 7) & (df['时'] <= 18) & (df['估计辐照强度w/m2'] > 0)
    df_daylight = df[daylight_mask].copy()
    
    # 数据清洗：处理无穷值和NaN
    df_daylight = df_daylight.replace([np.inf, -np.inf], np.nan)
    df_daylight = df_daylight.fillna(0)
    
    # 确保所有数值特征在合理范围内
    numeric_cols = df_daylight.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # 处理过大值
        if df_daylight[col].abs().max() > 1e6:
            df_daylight[col] = df_daylight[col].clip(-1e6, 1e6)
    
    return df, df_daylight, daylight_mask

# 2. 简化混合模型
class HybridModel:
    def __init__(self, n_features):
        self.models = {
            'lgbm': lgb.LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=8,
                num_leaves=40,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'gbdt': GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=10,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
        }
        
        # 创建LSTM模型
        self.lstm_model = self._create_lstm_model(n_features)
        self.feature_columns = None
        
    def _create_lstm_model(self, n_features):
        # 创建时间序列模型
        inputs = Input(shape=(24, n_features))
        
        # LSTM层
        x = LSTM(64, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = LSTM(48)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # 全连接层
        x = Dense(48, activation='relu')(x)
        outputs = Dense(1, activation='relu')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model
    
    def _clean_data(self, X, y):
        """清洗数据，处理NaN和无穷值"""
        # 合并特征和目标变量
        data = X.copy()
        data['target'] = y
        
        # 处理无穷值
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # 填充NaN值
        data = data.fillna(0)
        
        # 分离回X和y
        X_clean = data.drop(columns=['target'])
        y_clean = data['target']
        
        return X_clean, y_clean
    
    def fit(self, X_train, y_train, feature_columns):
        self.feature_columns = feature_columns
        
        # 数据清洗
        X_train_clean, y_train_clean = self._clean_data(X_train[self.feature_columns], y_train)
        
        # 训练梯度提升模型
        early_stop = lgb.early_stopping(stopping_rounds=50, verbose=False)
        self.models['lgbm'].fit(
            X_train_clean, 
            y_train_clean,
            eval_set=[(X_train_clean, y_train_clean)],
            callbacks=[early_stop]
        )
        
        # 训练GBDT模型
        self.models['gbdt'].fit(X_train_clean, y_train_clean)
        
        # 训练Ridge模型
        self.models['ridge'].fit(X_train_clean, y_train_clean)
        
        # 仅在数据足够时训练LSTM
        if len(X_train) >= 24:
            try:
                # 准备时序数据
                X_lstm, y_lstm = self._prepare_ts_data(X_train, y_train)
                
                if len(X_lstm) > 0:
                    # 清洗时序数据
                    X_lstm_clean = np.nan_to_num(X_lstm, nan=0, posinf=0, neginf=0)
                    y_lstm_clean = np.nan_to_num(y_lstm, nan=0, posinf=0, neginf=0)
                    
                    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
                    self.lstm_model.fit(
                        X_lstm_clean, 
                        y_lstm_clean,
                        epochs=50, 
                        batch_size=16,
                        callbacks=[early_stopping],
                        verbose=0
                    )
            except Exception as e:
                print(f"LSTM训练失败: {str(e)}")
    
    def predict(self, X):
        # 确保特征列存在
        if not all(col in X.columns for col in self.feature_columns):
            missing = [col for col in self.feature_columns if col not in X.columns]
            print(f"警告: 缺失特征列 {missing}，使用0填充")
            for col in missing:
                X[col] = 0
        
        # 选择特征
        X_features = X[self.feature_columns].copy()
        
        # 数据清洗
        X_features = X_features.replace([np.inf, -np.inf], np.nan)
        X_features = X_features.fillna(0)
        
        # 树模型预测
        lgb_pred = self.models['lgbm'].predict(X_features)
        gbdt_pred = self.models['gbdt'].predict(X_features)
        ridge_pred = self.models['ridge'].predict(X_features)
        
        # LSTM预测（如果有足够的历史数据）
        lstm_pred = np.zeros(len(X))
        if hasattr(self, 'lstm_model'):
            try:
                # 准备时序数据
                X_ts, valid_idx = self._prepare_ts_data(X, predict=True)
                
                if len(X_ts) > 0:
                    # 清洗时序数据
                    X_ts_clean = np.nan_to_num(X_ts, nan=0, posinf=0, neginf=0)
                    lstm_pred_ts = self.lstm_model.predict(X_ts_clean).flatten()
                    for i, idx in enumerate(valid_idx):
                        lstm_pred[idx] = lstm_pred_ts[i]
            except:
                pass
        
        # 模型融合（加权平均）
        tree_models_avg = (lgb_pred + gbdt_pred) / 2
        final_pred = (0.5 * tree_models_avg) + (0.3 * ridge_pred) + (0.2 * lstm_pred)
        
        # 确保预测值非负
        final_pred[final_pred < 0] = 0
        
        return final_pred
    
    def _prepare_ts_data(self, X, y=None, predict=False):
        # 在训练时，准备时序样本
        if not predict:
            # 为每一天创建时序样本
            daily_groups = X.groupby(X['时间'].dt.date)
            
            X_ts_list = []
            y_ts_list = []
            
            for date, group in daily_groups:
                day_features = group[self.feature_columns]
                day_target = y.loc[group.index]
                
                # 确保完整的一天数据
                if len(day_features) >= 24:
                    # 取前24小时数据
                    X_ts = day_features.head(24).values.reshape(1, 24, len(self.feature_columns))
                    X_ts_list.append(X_ts)
                    
                    # 目标值使用第25小时的值（如果存在）
                    if len(day_features) >= 25:
                        y_ts_list.append(day_target.iloc[24])
                    else:
                        # 如果最后一天不足25小时，使用平均发电量
                        avg_y = day_target.mean()
                        y_ts_list.append(avg_y)
            
            if len(X_ts_list) > 0:
                return np.vstack(X_ts_list), np.array(y_ts_list)
            return [], []
        else:
            # 在预测时，使用最后的24小时数据
            X_ts = []
            valid_idx = []
            
            # 为每一天创建时序样本
            daily_groups = X.groupby(X['时间'].dt.date)
            
            for date, group in daily_groups:
                if len(group) >= 24:
                    # 创建时序样本
                    day_features = group[self.feature_columns]
                    X_ts_sample = day_features.values[-24:].reshape(1, 24, len(self.feature_columns))
                    X_ts.append(X_ts_sample)
                    
                    # 记录最后24小时的索引
                    valid_idx.extend(group.index[-24:])
            
            if len(X_ts) > 0:
                return np.vstack(X_ts), valid_idx
            return [], []

# 3. 增强型模型评估与可视化
def evaluate_and_visualize(full_df, mae, rmse, r2, daylight_mask):
    # 准备绘图
    plt.figure(figsize=(18, 15))
    
    # 实际与预测累计发电量对比
    plt.subplot(2, 2, 1)
    for date, group in full_df.groupby(full_df['时间'].dt.date):
        if '预测累计发电量' in group.columns and group['预测累计发电量'].notna().any():
            plt.plot(group['时间'], group['估计累计发电量kwh'], 'b-', 
                     alpha=0.7, linewidth=1.5, label=f'实际值 ({date})')
            plt.plot(group['时间'], group['预测累计发电量'], 'r--', 
                     alpha=0.8, linewidth=1.5, label=f'预测值 ({date})')
    plt.title(f'Comparison of daily actual and predicted cumulative power generation\nR²={r2:.4f}, MAE={mae:.2f} kWh', fontsize=14)
    plt.xlabel('Time')
    plt.ylabel('Accumulated power generation (kWh)')
    plt.grid(True)
    plt.legend()
    
    # 小时发电量对比图
    plt.subplot(2, 2, 2)
    if daylight_mask is not None:
        daylight = full_df[daylight_mask].copy()
        if not daylight.empty:
            plt.scatter(daylight['时间'], daylight['小时发电量'], alpha=0.5, s=30, label='Actual hourly power generation')
            plt.scatter(daylight['时间'], daylight['预测小时发电量'], alpha=0.5, s=30, c='r', label='Estimated hourly power generation')
            
            # 添加趋势线
            plt.plot(daylight['时间'], daylight['小时发电量'].rolling(24, min_periods=1).mean(), 
                     'g-', linewidth=2, label='Actual 24h average')
            plt.plot(daylight['时间'], daylight['预测小时发电量'].rolling(24, min_periods=1).mean(), 
                     'm-', linewidth=2, label='Estimated 24h average')
            
            plt.title('Comparison of actual and estimated hourly power generation', fontsize=14)
            plt.xlabel('Time')
            plt.ylabel('Hourly power generation (kWh)')
            plt.legend()
            plt.grid(True)
    
    # 残差分析
    plt.subplot(2, 2, 3)
    if '小时发电量' in full_df.columns and '预测小时发电量' in full_df.columns:
        full_df['残差'] = full_df['小时发电量'] - full_df.get('预测小时发电量', 0)
        daylight = full_df[daylight_mask].copy() if daylight_mask is not None else full_df
        
        if not daylight.empty and '残差' in daylight.columns:
            sns.histplot(daylight['残差'], kde=True, bins=30)
            plt.axvline(x=0, color='r', linestyle='--')
            plt.title(f'Distribution of residuals (MAE={mae:.2f}, RMSE={rmse:.2f})', fontsize=14)
            plt.xlabel('Residual (kWh)')
            plt.ylabel('Frequency')
            plt.grid(True)
    
    # 辐照强度与发电量关系
    plt.subplot(2, 2, 4)
    if daylight_mask is not None:
        daylight = full_df[daylight_mask].copy()
        if not daylight.empty:
            plt.scatter(daylight['估计辐照强度w/m2'], daylight['小时发电量'], 
                        c=daylight['当前温度'], cmap='viridis', alpha=0.6)
            plt.colorbar(label='温度 (°C)')
            
            # 添加最佳拟合线
            if len(daylight) > 1:
                coef = np.polyfit(daylight['估计辐照强度w/m2'], daylight['小时发电量'], 1)
                poly1d_fn = np.poly1d(coef)
                sorted_irrad = np.sort(daylight['估计辐照强度w/m2'])
                plt.plot(sorted_irrad, poly1d_fn(sorted_irrad), 'r-', linewidth=2)
            
            plt.title('Relationship', fontsize=14)
            plt.xlabel('Radiation intensity (w/m2)')
            plt.ylabel('Hourly power generation (kWh)')
            plt.grid(True)
    
    # 优化布局并保存
    plt.tight_layout()
    plt.savefig(f'附件/题目二_电站{DIANZHAN_ID}.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
    return plt

# 主程序
def main():
    print("===== 光伏发电高精度预测模型 =====")
    
    # 1. 数据读取
    print("步骤1/5: 读取数据...")
    try:
        df = pd.read_csv(f'附件/电站{DIANZHAN_ID}_估计_汇总.csv', parse_dates=['时间'])
        print(f"数据读取成功: {len(df)}行记录")
    except Exception as e:
        print(f"数据读取失败: {e}")
        return
    
    # 2. 高级数据预处理
    print("步骤2/5: 高级数据预处理...")
    try:
        full_df, daylight_df, daylight_mask = enhanced_preprocessing(df)
        print(f"预处理完成，有效光照数据: {len(daylight_df)}条")
    except Exception as e:
        print(f"数据预处理失败: {str(e)}")
        return
    
    # 3. 选择特征列
    feature_cols = [
        '估计辐照强度w/m2', '当前温度', '湿度', '风速', 
        '太阳高度因子', '日出时差(小时)', '日落时差(小时)', 
        '辐照-温度_交互', '辐照-湿度_交互', '温度-湿度_交互',
        '天气编码', '风向角度', '一年中第几天', '时',
        '辐照强度_6h_均值', '温度_24h_均值'
    ]
    
    # 根据数据可用性添加特征
    optional_features = ['季节分量', '趋势分量', '残差分量']
    for feat in optional_features:
        if feat in full_df.columns:
            feature_cols.append(feat)
            print(f"添加可选特征: {feat}")
    
    print(f"使用特征数: {len(feature_cols)}")
    
    # 4. 构建混合模型
    print("步骤3/5: 构建混合模型...")
    model = HybridModel(len(feature_cols))
    
    # 使用第一天作为训练数据
    train_date = pd.to_datetime('2024-05-01').date()
    train_mask = (full_df['时间'].dt.date == train_date) & daylight_mask
    X_train = full_df[train_mask].copy()
    y_train = full_df.loc[train_mask, '小时发电量']
    
    # 模型训练
    print("训练模型...")
    model.fit(X_train, y_train, feature_cols)
    
    # 5. 预测所有数据
    print("步骤4/5: 预测理论发电量...")
    # 在日光时段进行预测
    if daylight_mask is not None:
        pred_daylight_mask = full_df.index.isin(daylight_df.index)
        daylight_df = full_df[pred_daylight_mask].copy()
    else:
        daylight_df = full_df.copy()
    
    # 预测小时发电量
    daylight_df.loc[:, '预测小时发电量'] = model.predict(daylight_df)
    
    # 合并回完整数据集
    full_df['预测小时发电量'] = 0.0
    full_df.loc[daylight_df.index, '预测小时发电量'] = daylight_df['预测小时发电量']
    
    # 计算预测累计发电量
    full_df['预测累计发电量'] = full_df.groupby(
        full_df['时间'].dt.date)['预测小时发电量'].cumsum()
    
    # 使用实际值填充夜间时段
    if daylight_mask is not None:
        full_df.loc[~pred_daylight_mask, '预测累计发电量'] = full_df.loc[~pred_daylight_mask, '估计累计发电量kwh']
    
    # 6. 计算评估指标
    print("步骤5/5: 模型评估...")
    # 仅在有日光时段数据时评估
    if daylight_mask is not None:
        eval_mask = full_df['预测累计发电量'].notna() & (full_df['预测累计发电量'] >= 0) & pred_daylight_mask
    else:
        eval_mask = full_df['预测累计发电量'].notna() & (full_df['预测累计发电量'] >= 0)
        
    if any(eval_mask):
        y_true = full_df.loc[eval_mask, '估计累计发电量kwh']
        y_pred = full_df.loc[eval_mask, '预测累计发电量']
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        print(f"\n模型评估指标:")
        print(f"MAE = {mae:.2f} kWh")
        print(f"RMSE = {rmse:.2f} kWh")
        print(f"R² = {r2:.4f}")
    else:
        print("警告: 没有足够的数据用于模型评估")
        mae, rmse, r2 = 0, 0, 0
    
    # 7. 可视化与结果保存
    evaluate_and_visualize(full_df, mae, rmse, r2, daylight_mask)
    
    # 保存数据
    filename = f'附件/电站{DIANZHAN_ID}_估计_汇总_预测.csv'
    full_df.to_csv(filename, index=False)
    print("\n结果已保存至: " + filename)
    print("可视化图片已保存")
    
    # 附加性能提示
    print("\n===== 分析完成 =====")
    if r2 > 0:
        print(f"模型性能: R² = {r2:.4f}")
        if r2 < 0.5:
            print("提示: 添加更多历史数据可进一步提升模型精度")
        elif r2 < 0.8:
            print("提示: 模型性能良好，可以尝试微调参数")
        else:
            print("提示: 模型性能优秀!")
    else:
        print("提示: 模型未能成功训练，请检查数据质量")

if __name__ == "__main__":
    main()
