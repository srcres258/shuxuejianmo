import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import interpolate
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib

# 设置字体，能够显示中文
matplotlib.rc("font",family='SimSun',weight="bold")

DIANZHAN_ID: int = 1
TIME_COL: str = '时间'
VALUE_COL: str = '辐照强度w/m2'

# 加载数据
data = pd.read_excel(f"附件/电站{DIANZHAN_ID}环境检测仪数据.xlsx", engine="openpyxl")

print(data.dtypes)

# 假设原始数据已经加载为data DataFrame
# 列名：[TIME_COL, VALUE_COL]
print("原始数据预览:")
print(data.head())

## 1. 数据预处理
# 去除时间重复的值
data = data.drop_duplicates(subset=[TIME_COL], keep='first', inplace=False).copy()
# 确保时间列是datetime类型并设为索引
data[TIME_COL] = pd.to_datetime(data[TIME_COL])
data.set_index(TIME_COL, inplace=True)
data.sort_index(inplace=True)  # 确保按时间排序

## 2. 数据清洗

def clean_data(df: pd.DataFrame):
    """
    数据清洗函数：处理缺失值、异常值和误差较大的数据
    返回清洗后的DataFrame和异常点标记
    """
    cleaned_df = df.copy()
    
    # 2.1 处理缺失值
    print(f"原始数据缺失值数量: {cleaned_df[VALUE_COL].isna().sum()}")
    # 简单线性插值填充缺失值（后续会再检测异常值）
    cleaned_df[VALUE_COL] = cleaned_df[VALUE_COL].interpolate(method='time')
    
    # 2.2 检测并处理异常值
    # 方法1：基于Z-score的异常值检测
    z_scores = np.abs(stats.zscore(cleaned_df[VALUE_COL]))
    threshold = 3  # 3个标准差
    z_outliers = z_scores > threshold
    
    # 方法2：使用隔离森林检测异常值
    scaler = StandardScaler()
    X = scaler.fit_transform(cleaned_df[VALUE_COL].values.reshape(-1, 1))
    iso_forest = IsolationForest(contamination=0.05, random_state=42)  # 假设5%异常值
    iso_outliers = iso_forest.fit_predict(X) == -1
    
    # 综合两种方法的结果
    combined_outliers = z_outliers | iso_outliers
    print(f"检测到的异常值数量: {combined_outliers.sum()}")
    
    # 标记异常值（不直接删除，先查看）
    cleaned_df['is_outlier'] = combined_outliers
    cleaned_df['原始值'] = df[VALUE_COL]  # 保留原始值
    
    # 可视化异常值
    plt.figure(figsize=(14, 6))
    plt.plot(cleaned_df.index, cleaned_df[VALUE_COL], 'b-', label='正常数据')
    plt.scatter(cleaned_df.index[combined_outliers], 
                cleaned_df.loc[combined_outliers, VALUE_COL], 
                c='r', marker='x', s=100, label='异常值')
    plt.title('异常值检测结果')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 移除异常值（用NaN替代）
    cleaned_df.loc[combined_outliers, VALUE_COL] = np.nan
    
    # 再次插值填充（使用更稳健的方法）
    cleaned_df[VALUE_COL] = cleaned_df[VALUE_COL].interpolate(method='time')
    
    # 移除辅助列
    cleaned_df.drop(['is_outlier', '原始值'], axis=1, inplace=True)
    
    return cleaned_df

# 执行数据清洗
cleaned_data = clean_data(data)

## 3. 建立更精确的单值函数模型

def build_model(df):
    """
    构建时间序列预测模型
    返回可以预测任意时间点的函数
    """
    # 转换为数值时间戳（秒）和值
    timestamps = df.index.astype(np.int64) // 10**9
    values = df[VALUE_COL].values
    
    # 3.1 创建插值函数（用于已知范围内的点）
    interp_func = interpolate.interp1d(
        timestamps, values, 
        kind='cubic', 
        fill_value='extrapolate'  # 谨慎使用外推
    )
    
    # 3.2 创建时间序列预测模型（用于未来预测）
    # 使用三重指数平滑（Holt-Winters）
    model = ExponentialSmoothing(
        values,
        trend='add',
        seasonal='add',  # 假设有季节性
        seasonal_periods=24,  # 假设日周期（24小时）
        damped_trend=True
    )
    model_fit = model.fit()
    
    def predict_func(target_time):
        """
        综合预测函数
        target_time: pd.Timestamp或datetime对象
        """
        target_ts = pd.to_datetime(target_time).value // 10**9
        
        if target_time <= df.index[-1]:
            # 在已知范围内使用插值
            return float(interp_func(target_ts))
        else:
            # 未来预测
            # 计算需要预测多少步
            last_ts = df.index[-1].value // 10**9
            steps = int((target_ts - last_ts) / 3600)  # 按小时计步
            if steps <= 0:
                steps = 1
            forecast = model_fit.forecast(steps)
            return float(forecast[-1])
    
    return predict_func

# 构建预测函数
predict_function = build_model(cleaned_data)

## 4. 生成整小时预测数据集

def generate_hourly_estimates(df, predict_func):
    """
    生成整小时刻度的估计数据集
    """
    # 确定时间范围（扩展到整小时）
    start_time = df.index[0].floor('H')  # 前推到整点
    end_time = df.index[-1].ceil('H')    # 后推到整点
    
    # 生成每小时时间序列
    hourly_index = pd.date_range(start=start_time, end=end_time, freq='H')
    
    # 预测每个小时的值
    estimated_values = [predict_func(t) for t in hourly_index]
    
    # 创建DataFrame
    estimated_data = pd.DataFrame({
        TIME_COL: hourly_index,
        '估计值': estimated_values
    })
    estimated_data.set_index(TIME_COL, inplace=True)
    
    return estimated_data

# 生成估计数据集
estimated_data = generate_hourly_estimates(cleaned_data, predict_function)

## 5. 结果可视化与验证

# 5.1 绘制清洗前后对比
plt.figure(figsize=(14, 8))
plt.plot(data.index, data[VALUE_COL], 'b-', alpha=0.5, label='原始数据')
plt.plot(cleaned_data.index, cleaned_data[VALUE_COL], 'r-', linewidth=2, label='清洗后数据')
plt.title('数据清洗前后对比')
plt.legend()
plt.grid(True)
plt.show()

# 5.2 绘制整小时预测结果
plt.figure(figsize=(14, 8))
plt.plot(cleaned_data.index, cleaned_data[VALUE_COL], 'bo-', markersize=4, label='清洗后数据点')
plt.plot(estimated_data.index, estimated_data['估计值'], 'g-', linewidth=1.5, label='整小时估计')
plt.title('整小时估计结果')
plt.legend()
plt.grid(True)
plt.show()

# 输出结果
print("\n清洗后数据统计信息:")
print(cleaned_data.describe())

print("\n整小时估计数据集预览:")
print(estimated_data.head())
print("...")
print(estimated_data.tail())
