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

# 配置环境
plt.style.use('seaborn-whitegrid')
sns.set_palette('coolwarm')
np.random.seed(42)
tf.random.set_seed(42)

# 1. 增强型数据预处理
def enhanced_preprocessing(df):
    # 复制数据避免修改原始数据
    df = df.copy()
    
    # 转换时间列
    df['时间'] = pd.to_datetime(df['时间'])
    
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
    df['太阳高度因子'] = np.sin(np.pi * (df['时'] - 6) / daylight_duration)
    
    # 添加假日特征
    cn_holidays = holidays.CountryHoliday('CN')
    df['是否假日'] = df['时间'].dt.date.apply(lambda d: 1 if d in cn_holidays else 0)
    
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
    
    # 季节分解
    def seasonal_decomposition(group):
        try:
            # 创建时间序列索引
            ts = group.set_index('时间')['小时发电量'].asfreq('H', fill_value=0)
            
            # 进行季节分解
            if len(ts) > 24 * 2:  # 至少需要48小时数据
                decomposition = seasonal_decompose(ts, model='additive', period=24)
                group['季节分量'] = decomposition.seasonal.values
                group['趋势分量'] = decomposition.trend.values
                group['残差分量'] = decomposition.resid.values
            return group
        except:
            return group
    
    # 按天分组应用季节分解
    df = df.groupby(df['时间'].dt.date, group_keys=False).apply(seasonal_decomposition).fillna(0)
    
    # 选择有光照数据（7:00-18:00且辐照>0）
    daylight_mask = (df['时'] >= 7) & (df['时'] <= 18) & (df['估计辐照强度w/m2'] > 0)
    df_daylight = df[daylight_mask].copy()
    
    return df, df_daylight, daylight_mask

# 2. 多模型融合预测系统
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
    
    def fit(self, X_train, y_train, timestamps, feature_columns):
        self.feature_columns = feature_columns
        
        # 准备时序数据
        X_lstm, y_lstm = self._prepare_ts_data(X_train, y_train, timestamps)
        
        # 训练梯度提升模型
        early_stop = lgb.early_stopping(stopping_rounds=50, verbose=False)
        self.models['lgbm'].fit(
            X_train[self.feature_columns], 
            y_train,
            eval_set=[(X_train[self.feature_columns], y_train)],
            callbacks=[early_stop]
        )
        
        # 训练GBDT模型
        self.models['gbdt'].fit(X_train[self.feature_columns], y_train)
        
        # 训练Ridge模型
        self.models['ridge'].fit(X_train[self.feature_columns], y_train)
        
        # 训练LSTM模型
        if len(X_lstm) > 0:
            early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
            self.lstm_model.fit(
                X_lstm, 
                y_lstm,
                epochs=100, 
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
    
    def predict(self, X, timestamps):
        # 树模型预测
        lgb_pred = self.models['lgbm'].predict(X[self.feature_columns])
        gbdt_pred = self.models['gbdt'].predict(X[self.feature_columns])
        ridge_pred = self.models['ridge'].predict(X[self.feature_columns])
        
        # LSTM预测（如果有足够的历史数据）
        lstm_pred = np.zeros(len(X))
        if hasattr(self, 'last_day_features') and not X.empty:
            # 准备时序数据
            X_ts, _, valid_idx = self._prepare_ts_data(X, None, timestamps, predict=True)
            
            if len(X_ts) > 0:
                lstm_pred_ts = self.lstm_model.predict(X_ts).flatten()
                for i, idx in enumerate(valid_idx):
                    lstm_pred[idx] = lstm_pred_ts[i]
        
        # 模型融合（加权平均）
        tree_models_avg = (lgb_pred + gbdt_pred) / 2
        final_pred = (0.4 * tree_models_avg) + (0.3 * ridge_pred) + (0.3 * lstm_pred)
        
        # 确保预测值非负
        final_pred[final_pred < 0] = 0
        
        return final_pred
    
    def _prepare_ts_data(self, X, y, timestamps, predict=False):
        if predict:
            # 在预测时，使用最后的24小时数据
            last_24h = X.iloc[-24:].copy()
            if len(last_24h) < 24:
                return [], [], []
            
            # 创建时序样本
            X_ts = last_24h[self.feature_columns].values.reshape(1, 24, len(self.feature_columns))
            return X_ts, None, list(range(len(X)-24, len(X)))
        else:
            # 在训练时，为每一天创建时序样本
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
            
            return np.vstack(X_ts_list), np.array(y_ts_list), []

# 3. 增强型模型评估与可视化
def evaluate_and_visualize(full_df, predictions, daylight_mask, mae, rmse, r2):
    # 准备绘图
    plt.figure(figsize=(18, 20))
    
    # 实际与预测累计发电量对比
    plt.subplot(3, 2, 1)
    for date, group in full_df.groupby(full_df['时间'].dt.date):
        if group['预测累计发电量'].notna().any():
            plt.plot(group['时间'], group['估计累计发电量kwh'], 'b-', 
                     alpha=0.7, linewidth=1.5, label=f'实际值 ({date})')
            plt.plot(group['时间'], group['预测累计发电量'], 'r--', 
                     alpha=0.8, linewidth=1.5, label=f'预测值 ({date})')
    plt.title(f'每日实际与预测累计发电量对比\nR²={r2:.4f}, MAE={mae:.2f} kWh', fontsize=14)
    plt.xlabel('时间')
    plt.ylabel('累计发电量 (kWh)')
    plt.grid(True)
    plt.legend()
    
    # 小时发电量对比图
    plt.subplot(3, 2, 2)
    daylight = full_df[daylight_mask].copy()
    plt.scatter(daylight['时间'], daylight['小时发电量'], alpha=0.5, s=30, label='实际小时发电量')
    plt.scatter(daylight['时间'], daylight['预测小时发电量'], alpha=0.5, s=30, c='r', label='预测小时发电量')
    
    # 添加趋势线
    plt.plot(daylight['时间'], daylight['小时发电量'].rolling(24, min_periods=1).mean(), 
             'g-', linewidth=2, label='实际24h平均')
    plt.plot(daylight['时间'], daylight['预测小时发电量'].rolling(24, min_periods=1).mean(), 
             'm-', linewidth=2, label='预测24h平均')
    
    plt.title('每小时实际与预测发电量对比', fontsize=14)
    plt.xlabel('时间')
    plt.ylabel('小时发电量 (kWh)')
    plt.legend()
    plt.grid(True)
    
    # 残差分析
    plt.subplot(3, 2, 3)
    daylight['残差'] = daylight['小时发电量'] - daylight['预测小时发电量']
    sns.histplot(daylight['残差'], kde=True, bins=30)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'残差分布 (MAE={mae:.2f}, RMSE={rmse:.2f})', fontsize=14)
    plt.xlabel('残差 (kWh)')
    plt.ylabel('频率')
    plt.grid(True)
    
    # 残差随时间变化
    plt.subplot(3, 2, 4)
    plt.scatter(daylight['时间'], daylight['残差'], alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    
    # 添加7日移动平均
    daylight['残差_7d_avg'] = daylight['残差'].rolling(24 * 7, min_periods=1).mean()
    plt.plot(daylight['时间'], daylight['残差_7d_avg'], 'g-', linewidth=2, label='7日移动平均')
    
    plt.title('残差时间序列', fontsize=14)
    plt.xlabel('时间')
    plt.ylabel('残差 (kWh)')
    plt.legend()
    plt.grid(True)
    
    # 辐照强度与发电量关系
    plt.subplot(3, 2, 5)
    plt.scatter(daylight['估计辐照强度w/m2'], daylight['小时发电量'], 
                c=daylight['当前温度'], cmap='viridis', alpha=0.6)
    plt.colorbar(label='温度 (°C)')
    
    # 添加最佳拟合线
    coef = np.polyfit(daylight['估计辐照强度w/m2'], daylight['小时发电量'], 1)
    poly1d_fn = np.poly1d(coef)
    sorted_irrad = np.sort(daylight['估计辐照强度w/m2'])
    plt.plot(sorted_irrad, poly1d_fn(sorted_irrad), 'r-', linewidth=2)
    
    plt.title('辐照强度、温度与发电量关系', fontsize=14)
    plt.xlabel('辐照强度 (w/m2)')
    plt.ylabel('小时发电量 (kWh)')
    plt.grid(True)
    
    # 模型评估指标雷达图
    plt.subplot(3, 2, 6, polar=True)
    
    metrics = ['R²', 'Precision', 'Recall', 'MAE', 'RMSE']
    values = [
        max(0, r2),  # R²
        min(1.0, max(0, 1 - mae/(mae+rmse))),  # 自定义精度
        min(1.0, max(0, 1 - rmse/(mae+rmse))),  # 自定义召回
        1 - mae/(max(daylight['小时发电量']) - min(daylight['小时发电量'])),  # MAE相对值
        1 - rmse/(max(daylight['小时发电量']) - min(daylight['小时发电量']))  # RMSE相对值
    ]
    
    # 确保值在合理范围内
    values = [max(0, min(1, v)) for v in values]
    
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # 绘图
    plt.polar(angles, values + [values[0]], color='b', linewidth=2)
    plt.fill(angles, values + [values[0]], color='b', alpha=0.25)
    
    # 添加标签
    plt.xticks(angles[:-1], metrics)
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], ['0', '0.25', '0.5', '0.75', '1.0'])
    plt.title('模型性能评估', fontsize=14)
    
    # 优化布局并保存
    plt.tight_layout()
    plt.savefig('enhanced_pv_model_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return plt

# 主程序
def main():
    print("===== 光伏发电高精度预测模型 =====")
    
    # 1. 数据读取
    print("步骤1/5: 读取数据...")
    try:
        df = pd.read_csv('data.csv', parse_dates=['时间'])
        print(f"数据读取成功: {len(df)}行记录")
    except Exception as e:
        print(f"数据读取失败: {e}")
        return
    
    # 2. 高级数据预处理
    print("步骤2/5: 高级数据预处理...")
    full_df, daylight_df, daylight_mask = enhanced_preprocessing(df)
    print("特征工程完成，新增特征数量:", len(full_df.columns) - len(df.columns))
    
    # 3. 选择特征列
    feature_cols = [
        '估计辐照强度w/m2', '当前温度', '湿度', '风速', 
        '太阳高度因子', '日出时差(小时)', '日落时差(小时)', 
        '辐照-温度_交互', '辐照-湿度_交互', '温度-湿度_交互',
        '天气编码', '风向角度', '一年中第几天', '时',
        '辐照强度_6h_均值', '温度_24h_均值',
        '季节分量', '趋势分量', '残差分量'
    ]
    
    # 4. 构建混合模型
    print("步骤3/5: 构建混合模型...")
    model = HybridModel(len(feature_cols))
    
    # 使用第一天作为训练数据
    train_date = pd.to_datetime('2024-05-01').date()
    train_mask = (full_df['时间'].dt.date == train_date) & daylight_mask
    X_train = full_df[train_mask].copy()
    y_train = full_df.loc[train_mask, '小时发电量']
    timestamps = X_train['时间']
    
    # 模型训练
    model.fit(X_train, y_train, timestamps, feature_cols)
    
    # 5. 预测所有数据
    print("步骤4/5: 预测理论发电量...")
    # 在日光时段进行预测
    pred_daylight_mask = full_df.index.isin(daylight_df.index)
    daylight_df = full_df[pred_daylight_mask].copy()
    
    # 预测小时发电量
    daylight_df.loc[:, '预测小时发电量'] = model.predict(daylight_df, daylight_df['时间'])
    
    # 合并回完整数据集
    full_df['预测小时发电量'] = 0.0
    full_df.loc[daylight_df.index, '预测小时发电量'] = daylight_df['预测小时发电量']
    
    # 计算预测累计发电量
    full_df['预测累计发电量'] = full_df.groupby(
        full_df['时间'].dt.date)['预测小时发电量'].cumsum()
    
    # 使用实际值填充夜间时段
    full_df.loc[~pred_daylight_mask, '预测累计发电量'] = full_df.loc[~pred_daylight_mask, '估计累计发电量kwh']
    
    # 6. 计算评估指标
    print("步骤5/5: 模型评估...")
    # 仅在有日光时段数据时评估
    eval_mask = full_df['预测累计发电量'].notna() & (full_df['预测累计发电量'] >= 0) & pred_daylight_mask
    y_true = full_df.loc[eval_mask, '估计累计发电量kwh']
    y_pred = full_df.loc[eval_mask, '预测累计发电量']
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n模型评估指标:")
    print(f"MAE = {mae:.2f} kWh")
    print(f"RMSE = {rmse:.2f} kWh")
    print(f"R² = {r2:.4f}")
    
    # 7. 可视化与结果保存
    plt = evaluate_and_visualize(full_df, y_pred, daylight_mask, mae, rmse, r2)
    
    # 保存数据
    full_df.to_csv('data_estimated.csv', index=False)
    print("\n结果已保存至: data_estimated.csv")
    print("可视化已保存为: enhanced_pv_model_results.png")
    
    # 附加性能提示
    print("\n性能提升提示:")
    if r2 < 0.5:
        print("- 尝试使用更多历史数据训练模型")
        print("- 检查数据集质量，可能有数据质量问题")
    elif r2 < 0.7:
        print("- 添加更多天气历史数据可进一步提升精度")
        print("- 尝试不同模型权重组合")
    else:
        print("- 模型性能优异，已接近物理限制")

if __name__ == "__main__":
    main()