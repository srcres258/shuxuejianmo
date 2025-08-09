import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from ruptures.detection import Binseg, Pelt
import pvlib
import matplotlib

# 设置字体，能够显示中文
matplotlib.rc("font", family='SimSun', weight="bold")

# 加载数据
data = pd.read_csv('附件/电站1_估计_汇总_预测_灰尘指数.csv')

# 数据清洗
def clean_data(df):
    # 处理缺失值
    df = df.dropna(subset=['Dust_Index', '估计辐照强度w/m2', '小时发电量'])

    # 处理异常值：使用Hampel滤波器
    def hampel_filter(series, window=7, k=3):
        median = series.rolling(window, center=True).median()
        mad = stats.median_abs_deviation(series, scale='normal')
        threshold = k * mad
        return series[(series - median).abs() <= threshold]

    df['Dust_Index_clean'] = hampel_filter(df['Dust_Index'])
    df['Dust_Index_clean'] = df['Dust_Index_clean'].fillna(method='ffill').fillna(method='bfill')

    # 筛选白天数据
    df['is_daytime'] = (df['估计辐照强度w/m2'] >= 150) | (df['太阳高度因子'] >= 0.25)
    df = df[df['is_daytime']]

    return df

data_clean = clean_data(data)

# 特征工程
def add_features(df):
    # 时间特征
    df['date'] = pd.to_datetime(df['时间'])
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_week'] = df['date'].dt.dayofweek
    df['season'] = (df['date'].dt.month % 12 + 3) // 3  # 1:冬, 2:春, 3:夏, 4:秋
    df['is_holiday'] = df['是否假日'].astype(bool)

    # 气象特征
    df['rain_prob'] = df['天气'].apply(lambda x: 0.9 if '雨' in x else (0.3 if '扬沙' in x else 0))
    df['wind_speed'] = df['风速']
    df['humidity'] = df['湿度']

    return df

data_clean = add_features(data_clean)

# 计算积灰影响指数
def calculate_dust_index(df):
    # 日聚合
    daily_df = df.groupby(pd.Grouper(key='date', freq='D')).agg({
        'Dust_Index_clean': 'median',
        '估计辐照强度w/m2': 'mean',
        '小时发电量': 'sum',
        '预测小时发电量': 'sum'
    }).reset_index()

    # 计算积灰影响指数
    daily_df['DII'] = daily_df['Dust_Index_clean']
    daily_df['SR'] = 1 - daily_df['DII']

    return daily_df

daily_data = calculate_dust_index(data_clean)

# 计算基准清洁产能
def calculate_base_energy(df, daily_df):
    # 计算基准清洁产能
    df['y_base'] = df['预测小时发电量'] * 1.0  # 假设预测值已经是基准
    daily_df['E0'] = df.groupby(pd.Grouper(key='date', freq='D'))['y_base'].sum().values

    return daily_df

daily_data = calculate_base_energy(data_clean, daily_data)

# 动态清洗决策
def dynamic_cleaning_strategy(daily_df, c_clean=2, Cap_kW=1000, p_bar=0.4):
    C = c_clean * Cap_kW  # 一次清洗总成本
    daily_df['LostE'] = daily_df['DII'] * daily_df['E0']
    daily_df['LostRMB'] = daily_df['LostE'] * p_bar

    # 初始化
    daily_df['L'] = 0.0
    daily_df['clean'] = False
    last_clean_day = None

    # 滚动计算累计损失收益债务
    for i in range(len(daily_df)):
        if last_clean_day is None:
            daily_df.loc[i, 'L'] = daily_df.loc[i, 'LostRMB']
        else:
            daily_df.loc[i, 'L'] = daily_df.loc[last_clean_day:i, 'LostRMB'].sum()

        if daily_df.loc[i, 'L'] >= C:
            daily_df.loc[i, 'clean'] = True
            last_clean_day = i
            daily_df.loc[i, 'L'] = 0  # 复位

    return daily_df

daily_data = dynamic_cleaning_strategy(daily_data)

# 价格敏感性分析
def price_sensitivity_analysis(daily_df, Cap_kW=1000, p_bar=0.4):
    c_clean_values = [0.5, 1, 2, 3, 4, 5]
    results = []

    for c_clean in c_clean_values:
        temp_df = daily_data.copy()
        temp_df = dynamic_cleaning_strategy(temp_df, c_clean=c_clean, Cap_kW=Cap_kW, p_bar=p_bar)

        # 计算总成本、节约电量、增收、净收益
        total_cost = temp_df['clean'].sum() * c_clean * Cap_kW
        saved_energy = temp_df.loc[temp_df['clean'], 'LostE'].sum()
        increased_revenue = saved_energy * p_bar
        net_benefit = increased_revenue - total_cost

        results.append({
            'c_clean': c_clean,
            'total_cost': total_cost,
            'saved_energy': saved_energy,
            'increased_revenue': increased_revenue,
            'net_benefit': net_benefit
        })

    return pd.DataFrame(results)

sensitivity_results = price_sensitivity_analysis(daily_data)
print(sensitivity_results)

# 可视化结果
def plot_results(daily_df, sensitivity_results):
    plt.figure(figsize=(12, 6))

    # 绘制积灰影响指数和清洗决策
    plt.subplot(2, 1, 1)
    plt.plot(daily_df['date'], daily_df['DII'], label='DII')
    plt.scatter(daily_df[daily_df['clean']]['date'],
                daily_df[daily_df['clean']]['DII'],
                color='red', label='Cleaning')
    plt.title('Dust Impact Index and Cleaning Decisions')
    plt.xlabel('Date')
    plt.ylabel('DII')
    plt.legend()

    # 绘制价格敏感性分析
    plt.subplot(2, 1, 2)
    plt.plot(sensitivity_results['c_clean'], sensitivity_results['net_benefit'], marker='o')
    plt.title('Net Benefit vs Cleaning Cost')
    plt.xlabel('Cleaning Cost (元/kW)')
    plt.ylabel('Net Benefit (元)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_results(daily_data, sensitivity_results)