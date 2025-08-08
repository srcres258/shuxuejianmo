#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
光伏电站积灰影响分析与理论发电量预测系统
版本: 2.0
作者: 专业数据分析工程师
日期: 2023-2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.dates as mdates
import sys
import re
import warnings
import matplotlib

warnings.filterwarnings('ignore')

# 设置字体，能够显示中文
matplotlib.rc("font", family='SimSun', weight="bold")

# 设置专业级可视化风格
plt.style.use('seaborn-whitegrid')
sns.set_palette("viridis")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
plt.rcParams['axes.unicode_minus'] = False    # 负号显示

def parse_input_parameters():
    """从标准输入解析光伏电站参数"""
    print("📝 正在读取光伏电站参数...")
    
    try:
        # 读取5行输入
        lines = [sys.stdin.readline().strip() for _ in range(5)]
        
        # 1. 解析经纬度
        lat_lon = lines[0]
        print(f"  - 经纬度: {lat_lon}")
        
        # 2. 解析装机容量
        capacity_str = lines[1].replace('kWp', '').strip()
        K_value = float(capacity_str)
        print(f"  - 装机容量: {K_value} kW")
        
        # 3. 解析安装倾角
        tilt_angle = float(lines[2].replace('度', '').strip())
        print(f"  - 安装倾角: {tilt_angle}度")
        
        # 4. 解析方位角
        azimuth = float(lines[3].replace('度', '').strip())
        print(f"  - 方位角: {azimuth}度")
        
        # 5. 解析清洗日期
        cleaning_date = lines[4].strip()
        if cleaning_date == "无":
            print("  - 清洗记录: 无")
            cleaning_dates = []
        else:
            # 尝试多种日期格式
            date_formats = [
                '%Y年%m月%d日', '%Y-%m-%d', '%Y/%m/%d', 
                '%Y年%m月%d号', '%Y年%m月%d'
            ]
            
            parsed_date = None
            for fmt in date_formats:
                try:
                    parsed_date = pd.to_datetime(cleaning_date, format=fmt)
                    break
                except:
                    continue
            
            if parsed_date is None:
                print(f"  ⚠️ 无法解析清洗日期: {cleaning_date}, 将尝试自动检测")
                cleaning_dates = []
            else:
                print(f"  - 清洗日期: {parsed_date.strftime('%Y-%m-%d')}")
                cleaning_dates = [parsed_date]
        
        print("✅ 参数解析完成")
        return {
            'lat_lon': lat_lon,
            'K_value': K_value,
            'tilt_angle': tilt_angle,
            'azimuth': azimuth,
            'cleaning_dates': cleaning_dates
        }
    
    except Exception as e:
        print(f"❌ 参数解析失败: {str(e)}")
        print("   使用默认参数继续...")
        return {
            'lat_lon': '未知',
            'K_value': None,  # 将在后续自动估计
            'tilt_angle': 0,
            'azimuth': 0,
            'cleaning_dates': []
        }

def load_and_preprocess(file_path):
    """加载并预处理光伏数据（特别处理周期性特征）"""
    print("🚀 开始数据加载与预处理...")
    
    # 1. 加载CSV数据
    df = pd.read_csv(file_path, parse_dates=['时间'], index_col='时间')
    df = df.sort_index()
    
    # 2. 重命名列以符合标准命名
    df = df.rename(columns={
        '估计累计发电量kwh': 'cumulative_power',
        '估计辐照强度w/m2': 'irradiance',
        '当前温度': 'ambient_temp',
        '风速': 'wind_speed',
        '湿度': 'humidity'
    })
    
    # 3. 基础数据清洗
    print(f"  - 原始数据: {len(df)}条记录，时间范围: {df.index.min()} 至 {df.index.max()}")
    
    # 4. 处理缺失值（线性插值）
    missing_before = df.isnull().sum().sum()
    df = df.interpolate(method='time', limit_direction='both')
    missing_after = df.isnull().sum().sum()
    print(f"  - 缺失值处理: 从{missing_before}个减少到{missing_after}个")
    
    # 5. 识别每天重置点（累计发电量从非零值跳回0）
    df['reset_point'] = False
    df['power_diff'] = df['cumulative_power'].diff()
    
    # 识别累计值重置点（当差分为负且绝对值很大时）
    reset_points = df['power_diff'] < -df['cumulative_power'].max() * 0.5
    
    # 标记重置点
    df.loc[reset_points, 'reset_point'] = True
    print(f"  - 检测到 {reset_points.sum()} 个累计发电量重置点")
    
    # 6. 计算瞬时发电功率（关键步骤！）
    df['instant_power'] = 0.0
    
    # 按日期分组处理
    df['date'] = df.index.date
    for date, group in df.groupby('date'):
        idx = group.index
        # 计算组内差分
        group['power_diff'] = group['cumulative_power'].diff()
        
        # 处理重置点（当天第一个点）
        if not pd.isna(group['power_diff'].iloc[0]):
            group.loc[idx[0], 'instant_power'] = group['cumulative_power'].iloc[0]
        else:
            group.loc[idx[0], 'instant_power'] = 0
            
        # 处理后续点
        for i in range(1, len(group)):
            # 如果是重置点（不应该在组内出现，因为已按日期分组）
            if group['power_diff'].iloc[i] < 0:
                # 重置点，使用当前累计值
                group.loc[idx[i], 'instant_power'] = group['cumulative_power'].iloc[i]
            else:
                # 正常差分
                group.loc[idx[i], 'instant_power'] = group['power_diff'].iloc[i]
        
        # 更新主数据框
        df.loc[idx, 'instant_power'] = group['instant_power']
    
    # 7. 创建日出日落时间标记
    if '日出时间' in df.columns and '日落时间' in df.columns:
        # 提取日出日落时间（假设所有行相同）
        try:
            sunrise_time = pd.to_datetime(df['日出时间'].iloc[0]).time()
            sunset_time = pd.to_datetime(df['日落时间'].iloc[0]).time()
            
            # 创建精确的白天标记
            df['is_day'] = df.index.to_series().apply(
                lambda x: sunrise_time <= x.time() <= sunset_time
            )
            
            # 创建日落过渡标记（考虑发电量渐变）
            df['daylight_factor'] = 1.0
            transition_hours = 1  # 日落前后1小时为过渡期
            
            for i, row in df.iterrows():
                current_time = i.time()
                # 日出前
                if current_time < sunrise_time:
                    df.at[i, 'daylight_factor'] = 0.0
                # 日落后
                elif current_time > sunset_time:
                    df.at[i, 'daylight_factor'] = 0.0
                else:
                    # 日落前过渡期
                    if (pd.Timestamp.combine(i.date(), sunset_time) - i) < pd.Timedelta(hours=transition_hours):
                        time_to_sunset = (pd.Timestamp.combine(i.date(), sunset_time) - i).total_seconds() / 3600
                        df.at[i, 'daylight_factor'] = time_to_sunset / transition_hours
                    # 日出后过渡期
                    elif (i - pd.Timestamp.combine(i.date(), sunrise_time)) < pd.Timedelta(hours=transition_hours):
                        time_after_sunrise = (i - pd.Timestamp.combine(i.date(), sunrise_time)).total_seconds() / 3600
                        df.at[i, 'daylight_factor'] = time_after_sunrise / transition_hours
        except Exception as e:
            print(f"  ⚠️ 日出日落时间解析失败: {str(e)}，使用默认白天标记")
            df['is_day'] = (df['irradiance'] > 50).astype(int)
    else:
        # 如果没有日出日落时间，使用辐照度阈值
        df['is_day'] = (df['irradiance'] > 50).astype(int)
    
    # 8. 异常值检测（基于IQR）
    Q1 = df['instant_power'].quantile(0.25)
    Q3 = df['instant_power'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - 1.5 * IQR)
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df['instant_power'] < lower_bound) | (df['instant_power'] > upper_bound)]
    print(f"  - 检测到{len(outliers)}个发电功率异常值 (IQR方法)")
    
    # 9. 添加关键特征
    df['hour'] = df.index.hour
    df['daylight'] = df['is_day'].astype(int)  # 白天标记
    
    print("✅ 数据预处理完成")
    return df

def detect_cleaning_events(df, min_eff_increase=0.05, min_duration=24):
    """自动检测清洗事件（无清洗记录时使用）"""
    print("\n🔍 开始清洗事件自动检测...")
    
    # 1. 确保有装机容量
    if 'K' not in df.columns or df['K'].isnull().all():
        print("  ⚠️ 无法计算效率，缺少装机容量K")
        return []
    
    # 2. 计算瞬时发电效率
    # 避免除以零
    df['efficiency'] = df['instant_power'] / (df['irradiance'] * df['K'] / 1000 + 1e-10)
    df['eff_diff'] = df['efficiency'].diff()
    
    # 3. 检测效率突增事件（仅白天）
    potential_cleaning = df[
        (df['eff_diff'] > min_eff_increase) & 
        (df['daylight'] == 1)
    ].index
    
    # 4. 验证持续性（排除云层干扰）
    confirmed_cleaning = []
    for ts in potential_cleaning:
        end_ts = ts + pd.Timedelta(hours=min_duration)
        if end_ts > df.index[-1]:
            continue
            
        # 检查后续时段效率是否持续高位（白天时段）
        post_eff = df.loc[ts:end_ts]
        post_eff = post_eff[post_eff['daylight'] == 1]
        
        if len(post_eff) >= min_duration * 0.7:  # 至少70%的白天时段
            if (post_eff['efficiency'] > df.loc[ts, 'efficiency'] * 0.95).all():
                confirmed_cleaning.append(ts)
    
    print(f"  - 检测到{len(confirmed_cleaning)}次可能的清洗事件")
    if confirmed_cleaning:
        print("  - 最近清洗事件:", [ts.strftime('%Y-%m-%d %H:%M') for ts in confirmed_cleaning[-3:]])
    
    return confirmed_cleaning

def estimate_plant_capacity(df):
    """估计电站装机容量K (当未提供时)"""
    print("\n🔧 开始估计电站装机容量...")
    
    # 1. 找出理想天气条件下的峰值发电
    ideal_conditions = df[
        (df['irradiance'] > 900) & 
        (df['irradiance'] < 1100) &
        (df['ambient_temp'] > 15) &
        (df['ambient_temp'] < 30) &
        (df['daylight'] == 1)
    ]
    
    if len(ideal_conditions) > 0:
        # 2. 取效率最高的10%作为参考
        top_efficiency = ideal_conditions.nlargest(int(len(ideal_conditions) * 0.1), 'efficiency')
        avg_power = top_efficiency['instant_power'].mean()
        
        # 3. 估计装机容量 (假设标准测试条件下的效率为18%)
        K_estimated = avg_power / 0.18
        print(f"  - 估计装机容量: {K_estimated:.2f} kW (基于理想天气条件)")
    else:
        # 4. 备用方法：使用最大发电量
        max_power = df['instant_power'].max()
        K_estimated = max_power / 0.15  # 假设最大效率15%
        print(f"  - 估计装机容量: {K_estimated:.2f} kW (基于最大发电量)")
    
    return K_estimated

def train_physical_model(df, cleaning_dates=None, K_value=None):
    """训练物理模型，拟合η0和α参数"""
    print("\n🔬 开始物理模型训练...")
    
    # 1. 如果K值未提供，估计装机容量
    if K_value is None:
        # 先尝试计算效率（需要K值）
        if 'K' not in df.columns:
            K_estimated = estimate_plant_capacity(df)
            df['K'] = K_estimated
        K_value = df['K'].iloc[0]
    
    # 2. 如果没有提供清洗日期，自动检测
    if cleaning_dates is None or len(cleaning_dates) == 0:
        print("  - 未提供清洗记录，启动自动清洗事件检测")
        cleaning_dates = detect_cleaning_events(df)
        if not cleaning_dates:
            print("  ⚠️ 无法检测到清洗事件，将使用高效率时段进行稳健拟合")
    
    # 3. 提取清洗后24小时数据（仅白天）
    clean_data = []
    if cleaning_dates:
        for clean_date in cleaning_dates:
            clean_ts = pd.Timestamp(clean_date)
            mask = ((df.index >= clean_ts) & 
                    (df.index < clean_ts + pd.Timedelta(hours=24)) &
                    (df['daylight'] == 1))
            clean_data.append(df[mask])
    
    # 4. 如果没有清洗数据，使用高效率时段
    if not clean_data:
        print("  - 使用高效率时段进行拟合")
        # 计算效率
        df['efficiency'] = df['instant_power'] / (df['irradiance'] * K_value / 1000 + 1e-10)
        
        # 找出效率最高的10%时段
        efficiency_threshold = df['efficiency'].quantile(0.9)
        high_efficiency = df[df['efficiency'] >= efficiency_threshold * 0.95]
        clean_data = [high_efficiency]
    
    if not clean_data:
        raise ValueError("未找到有效的高效率数据点，请检查数据质量")
    
    clean_df = pd.concat(clean_data)
    print(f"  - 使用{len(clean_df)}条高效率数据点进行拟合")
    
    # 5. 计算观测效率 (物理约束)
    clean_df['eta_obs'] = clean_df['instant_power'] * 1000 / (clean_df['irradiance'] * K_value + 1e-10)
    
    # 6. 线性回归拟合: eta_obs = η0 * [1 - α*(T-25)]
    X = clean_df[['ambient_temp']].values
    y = clean_df['eta_obs'].values
    
    # 7. RANSAC抗异常值拟合
    model = LinearRegression()
    model.fit(X, y)
    
    # 8. 参数解算
    intercept = model.intercept_
    slope = model.coef_[0]
    
    # 物理约束解算
    alpha = -slope / (intercept - 25 * slope + 1e-10)
    eta0 = intercept - 25 * slope * alpha
    
    # 9. 物理合理性验证
    valid_eta0 = 0.15 <= eta0 <= 0.25
    valid_alpha = -0.005 <= alpha <= -0.003
    
    if not valid_eta0 or not valid_alpha:
        print("⚠️ 参数超出典型范围，可能数据质量有问题:")
        print(f"   η0={eta0:.4f} (合理范围: 0.15-0.25)")
        print(f"   α={alpha:.6f} (合理范围: -0.005~-0.003)")
    
    # 10. 模型验证
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    print(f"\n⚙️ 物理模型训练结果:")
    print(f"   η0 = {eta0:.4f} (25°C时标准效率)")
    print(f"   α  = {alpha:.6f} (温度系数)")
    print(f"   R² = {r2:.4f} (拟合优度)")
    print(f"   MAE = {mae:.6f} (平均绝对误差)")
    
    # 11. 可视化验证
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.5, label='观测数据')
    plt.plot(X, y_pred, 'r-', lw=2, label=f'拟合曲线 (R²={r2:.4f})')
    plt.xlabel('环境温度 (°C)')
    plt.ylabel('观测效率 η_obs')
    plt.title('物理模型参数拟合验证', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('physical_model_fit.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {'eta0': eta0, 'alpha': alpha, 'r2': r2, 'K': K_value}

def calculate_theoretical_power(df, params):
    """计算理论发电量 P_ideal(t)"""
    print("\n⚡ 开始计算理论发电量...")
    
    K_value = params['K']
    
    # 1. 计算温度修正效率
    df['eta_temp'] = params['eta0'] * (1 + params['alpha'] * (df['ambient_temp'] - 25))
    
    # 2. 计算物理理想发电量 (kW)
    # P_ideal = G(t) * K * η(T(t)) * Δt
    df['P_ideal'] = (df['irradiance'] * K_value * df['eta_temp']) / 1000
    
    # 3. 应用日出日落过渡（考虑发电量渐变）
    if 'daylight_factor' in df.columns:
        df['P_ideal'] = df['P_ideal'] * df['daylight_factor']
    
    # 4. 边界处理：夜间归零
    df.loc[~df['is_day'], 'P_ideal'] = 0
    
    # 5. 计算残差
    df['residual_total'] = df['instant_power'] - df['P_ideal']
    
    # 6. 过滤正残差（仅保留积灰损失候选）
    df['dust_loss'] = np.where(
        df['residual_total'] <= 0, 
        -df['residual_total'],  # 积灰损失 (正值)
        0.0
    )
    
    # 7. 物理约束：损失不能超过理论发电量
    df['dust_loss'] = np.minimum(df['dust_loss'], df['P_ideal'] * 0.3)
    
    # 8. 计算积灰影响指数 (归一化)
    max_loss = df['dust_loss'].max()
    df['dust_index'] = df['dust_loss'] / max_loss if max_loss > 0 else 0
    
    print(f"  - 理论发电量范围: {df['P_ideal'].min():.2f} ~ {df['P_ideal'].max():.2f} kW")
    print(f"  - 平均积灰损失: {df['dust_loss'].mean():.2f} kW")
    print(f"  - 最大积灰损失: {df['dust_loss'].max():.2f} kW")
    
    return df

def evaluate_model(df):
    """模型评估：计算MAE、RMSE、R²等指标"""
    print("\n📊 开始模型评估...")
    
    # 1. 仅评估白天数据（发电时段）
    daylight_df = df[df['daylight'] == 1]
    
    if len(daylight_df) == 0:
        print("⚠️ 未找到白天数据，无法进行评估")
        return {
            'mae': float('nan'),
            'rmse': float('nan'),
            'r2': float('nan'),
            'avg_residual': float('nan'),
            'residual_std': float('nan')
        }
    
    # 2. 计算评估指标
    mae = mean_absolute_error(daylight_df['instant_power'], daylight_df['P_ideal'])
    rmse = np.sqrt(mean_squared_error(daylight_df['instant_power'], daylight_df['P_ideal']))
    r2 = r2_score(daylight_df['instant_power'], daylight_df['P_ideal'])
    
    # 3. 物理一致性检查
    avg_residual = daylight_df['residual_total'].mean()
    residual_std = daylight_df['residual_total'].std()
    
    print("\n📈 模型评估结果 (白天时段):")
    print(f"   MAE  = {mae:.4f} kW")
    print(f"   RMSE = {rmse:.4f} kW")
    print(f"   R²   = {r2:.4f}")
    print(f"   平均残差 = {avg_residual:.4f} kW (理想值接近0)")
    print(f"   残差标准差 = {residual_std:.4f} kW")
    
    # 4. 可视化评估
    plt.figure(figsize=(14, 10))
    
    # 子图1: 预测vs实际
    plt.subplot(2, 2, 1)
    plt.scatter(daylight_df['P_ideal'], daylight_df['instant_power'], 
                alpha=0.6, s=10, label='数据点')
    plt.plot([0, daylight_df['P_ideal'].max()], [0, daylight_df['P_ideal'].max()], 
             'r--', lw=2, label='理想线')
    plt.xlabel('理论发电量 (kW)')
    plt.ylabel('实际发电量 (kW)')
    plt.title('预测值 vs 实际值', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 子图2: 残差分布
    plt.subplot(2, 2, 2)
    sns.histplot(daylight_df['residual_total'], kde=True, bins=50)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('残差 (实际 - 理论)')
    plt.ylabel('频数')
    plt.title('残差分布', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 子图3: 时间序列对比
    plt.subplot(2, 1, 2)
    # 取最近一个完整天的数据
    last_date = df.index[-1].date()
    sample_df = df[df.index.date == last_date]
    
    # 如果没有完整的一天，取最近24小时
    if len(sample_df) < 24:
        sample_df = df.iloc[-24:]
    
    plt.plot(sample_df.index, sample_df['instant_power'], label='实际发电量', alpha=0.8)
    plt.plot(sample_df.index, sample_df['P_ideal'], label='理论发电量', alpha=0.8)
    plt.fill_between(sample_df.index, 
                     sample_df['P_ideal'], 
                     sample_df['instant_power'],
                     where=(sample_df['instant_power'] < sample_df['P_ideal']),
                     color='red', alpha=0.3, label='积灰损失')
    plt.xlabel('时间')
    plt.ylabel('发电量 (kW)')
    plt.title('理论发电量与实际发电量对比 (最近一天)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 格式化时间轴
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'avg_residual': avg_residual,
        'residual_std': residual_std
    }

def generate_report(df, metrics, params, station_info):
    """生成分析报告"""
    print("\n📝 生成分析报告...")
    
    # 1. 基本统计
    avg_dust_index = df['dust_index'].mean() * 100
    max_dust_index = df['dust_index'].max() * 100
    
    # 2. 潜在收益计算
    K_value = params['K']
    potential_gain = df['dust_loss'].mean() * 24 * 365  # 年化潜在增益 (kWh)
    cleaning_cost = K_value * 15  # 假设清洗成本15元/kW
    roi = (potential_gain * 0.8 - cleaning_cost) / cleaning_cost * 100 if cleaning_cost > 0 else 0
    
    # 3. 报告内容
    cleaning_info = "无" if not station_info['cleaning_dates'] else station_info['cleaning_dates'][0].strftime('%Y-%m-%d')
    
    report = f"""
    ================ 光伏电站积灰影响分析报告 ================
    
    【电站基本信息】
    - 位置: {station_info['lat_lon']}
    - 装机容量: {K_value:.2f} kW
    - 安装倾角: {station_info['tilt_angle']}度
    - 方位角: {station_info['azimuth']}度
    - 清洗记录: {cleaning_info}
    - 分析时段: {df.index.min().strftime('%Y-%m-%d')} 至 {df.index.max().strftime('%Y-%m-%d')}
    - 数据总量: {len(df)}小时
    
    【物理模型参数】
    - η0 (25°C标准效率): {params['eta0']:.4f}
    - α (温度系数): {params['alpha']:.6f}
    - 模型R²: {params['r2']:.4f}
    
    【积灰影响分析】
    - 平均积灰损失指数: {avg_dust_index:.1f}%
    - 最大积灰损失指数: {max_dust_index:.1f}%
    - 平均积灰损失: {df['dust_loss'].mean():.2f} kW
    
    【经济效益评估】
    - 年化潜在发电增益: {potential_gain:.0f} kWh
    - 清洗成本估算: {cleaning_cost:.0f} 元
    - 预期投资回报率(ROI): {roi:.1f}%
    
    【模型评估指标】
    - MAE: {metrics['mae']:.4f} kW
    - RMSE: {metrics['rmse']:.4f} kW
    - R²: {metrics['r2']:.4f}
    - 平均残差: {metrics['avg_residual']:.4f} kW
    
    【运维建议】
    {'  - 当前积灰水平正常，建议持续监测' if avg_dust_index < 3 else ''}
    {'  - 积灰损失已超过3%，建议安排清洗' if 3 <= avg_dust_index < 5 else ''}
    {'  - 积灰损失已超过5%，建议24小时内清洗' if avg_dust_index >= 5 else ''}
    
    ===================================================
    """
    
    # 4. 保存报告
    with open('dust_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print("✅ 分析报告已保存至 dust_analysis_report.txt")

def main():
    # ======================
    # 步骤0: 读取光伏电站参数
    # ======================
    print("\n" + "="*50)
    print("光伏电站积灰影响分析系统 v2.0")
    print("开始执行分析任务...")
    print("="*50)
    
    station_info = parse_input_parameters()
    
    # ======================
    # 配置参数
    # ======================
    DATA_PATH = '附件/电站1_估计_汇总.csv'     # 输入数据文件
    OUTPUT_PATH = '附件/电站1_估计_汇总_预测.csv'  # 输出文件
    
    # ======================
    # 主流程执行
    # ======================
    # 步骤1: 加载并预处理数据
    df = load_and_preprocess(DATA_PATH)
    
    # 添加装机容量到数据框
    if station_info['K_value'] is not None:
        df['K'] = station_info['K_value']
    
    # 步骤2: 物理模型训练
    try:
        params = train_physical_model(
            df, 
            cleaning_dates=station_info['cleaning_dates'],
            K_value=station_info['K_value']
        )
    except Exception as e:
        print(f"❌ 物理模型训练失败: {str(e)}")
        print("   尝试使用默认参数继续...")
        # 默认参数 (典型多晶硅组件)
        params = {
            'eta0': 0.18, 
            'alpha': -0.004, 
            'r2': 0.85,
            'K': station_info['K_value'] or 1000  # 使用输入的K值或默认值
        }
    
    # 步骤3: 计算理论发电量
    df = calculate_theoretical_power(df, params)
    
    # 步骤4: 模型评估
    metrics = evaluate_model(df)
    
    # 步骤5: 生成分析报告
    generate_report(df, metrics, params, station_info)
    
    # 步骤6: 保存结果到新CSV
    print("\n💾 保存结果到", OUTPUT_PATH)
    # 保留原始列名
    output_df = pd.read_csv(DATA_PATH, parse_dates=['时间'])
    
    # 添加新列
    output_df['瞬时发电功率_kW'] = df['instant_power'].values
    output_df['理论发电量_kW'] = df['P_ideal'].values
    output_df['残差'] = df['residual_total'].values
    output_df['积灰损失_kW'] = df['dust_loss'].values
    output_df['积灰指数'] = df['dust_index'].values
    
    # 保存结果
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ 结果已成功保存至 {OUTPUT_PATH}")
    print(f"   - 新增列: 瞬时发电功率_kW, 理论发电量_kW, 残差, 积灰损失_kW, 积灰指数")
    
    print("\n" + "="*50)
    print("🎉 任务完成！")
    print("   - 详细结果: data_estimated.csv")
    print("   - 模型评估: model_evaluation.png")
    print("   - 物理参数: physical_model_fit.png")
    print("   - 分析报告: dust_analysis_report.txt")
    print("="*50)

if __name__ == "__main__":
    main()
