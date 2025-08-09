import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib
import warnings
from statsmodels.tsa.seasonal import STL
from typing import Callable, Tuple
import seaborn as sns
from sklearn.metrics import r2_score

# 设置字体，能够显示中文
matplotlib.rc("font", family='SimSun', weight="bold")

DIANZHAN_ID: int
TIME_COL: str = '时间'
VALUE_COL: str = '辐照强度w/m2'
ESTIMATED_VALUE_COL: str = '估计辐照强度w/m2'

print("电站【环境检测仪数据】处理程序")
DIANZHAN_ID = int(input("请输入要处理的电站编号："))
if DIANZHAN_ID not in range(1, 5):
    print("输入错误，电站编号为1~4，请重新输入！")
    exit(1)
    
print("将处理电站" + str(DIANZHAN_ID) + "的环境检测仪数据")

warnings.filterwarnings('ignore')  # 抑制一些不重要警告
plt.style.use('ggplot')  # 使用更好的绘图样式

# 加载数据
print("0. 读入原始数据...")
data = pd.read_excel(f"附件/电站{DIANZHAN_ID}环境检测仪数据.xlsx", engine="openpyxl")

print(data.dtypes)

# 假设原始数据已经加载为data DataFrame
# 列名：[TIME_COL, VALUE_COL]
print("原始数据预览:")
print(data.head())

## 1. 数据预处理
print("1. 数据预处理...")
# 去除时间重复的值
data = data.drop_duplicates(subset=[TIME_COL], keep='first', inplace=False).copy()
# 确保时间列是datetime类型并设为索引
data[TIME_COL] = pd.to_datetime(data[TIME_COL])
data.set_index(TIME_COL, inplace=True)
data.sort_index(inplace=True)  # 确保按时间排序
data[VALUE_COL] = data[VALUE_COL].astype(float)  # 确保辐照强度列是float类型

## 2. 数据清洗与异常值检测
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """专门针对辐照强度数据的清洗函数"""
    
    cleaned_df = df.copy()
    original_values = df[VALUE_COL].values
    
    # 计算一天中的小时
    cleaned_df['小时'] = cleaned_df.index.hour + cleaned_df.index.minute / 60
    cleaned_df['周天'] = cleaned_df.index.dayofweek
    cleaned_df['年天'] = cleaned_df.index.dayofyear
    
    # 第一步：处理缺失值
    print(f"数据集原始缺失值数量：{cleaned_df[VALUE_COL].isna().sum()}")
    cleaned_df[VALUE_COL] = cleaned_df[VALUE_COL].interpolate(method='time')
    
    # 第二步：检测异常值 - 基于统计和业务知识
    # 方法1：隔离森林
    print("数据集异常值检测 - 方法1：隔离森林")
    scaler = StandardScaler()
    features = cleaned_df[['小时', '周天', '年天', VALUE_COL]]
    features_scaled = scaler.fit_transform(features)
    
    iso_forest = IsolationForest(
        contamination=0.05,
        random_state=42,
        n_estimators=200
    )
    iso_outliers = iso_forest.fit_predict(features_scaled) == -1
    
    # 方法2：时间序列分解异常检测
    print("数据集异常值检测 - 方法2：时间序列分解异常检测")
    stl = STL(
        cleaned_df[VALUE_COL],
        period=24 * 4, # 4个点/小时 * 24小时/天
        robust=True
    )
    result = stl.fit()
    residuals = result.resid
    resid_std = residuals.std()
    resid_outliers = np.abs(residuals) > 3 * resid_std
    
    # 方法3：基于业务逻辑检测 - 夜间值应为0
    # （根据数据集的特征易知，晚上不可能有辐照值）
    print("数据集异常值检测 - 方法3：基于业务逻辑检测 - 夜间值应为0")
    night_outliers = (
        ((cleaned_df['小时'] < 6) | (cleaned_df['小时'] > 19)) &
        (cleaned_df[VALUE_COL] > 5)
    )
    
    # 综合异常检测结果
    combined_outliers = iso_outliers | resid_outliers | night_outliers.values
    print(f"完毕，所检测到的数据集中的异常值数量：{combined_outliers.sum()}")
    
    # 可视化异常值（通过matplotlib绘图）
    plt.figure(figsize=(14, 6))
    plt.plot(cleaned_df.index, cleaned_df[VALUE_COL], 'b-', label='正常数据')
    plt.scatter(
        cleaned_df.index[combined_outliers],
        cleaned_df.loc[combined_outliers, VALUE_COL],
        c='r',
        marker='x',
        s=100,
        label='异常值'
    )
    plt.title('辐照强度异常值检测')
    plt.ylabel(VALUE_COL)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # 标记并替换异常值
    cleaned_df['原始值'] = original_values
    cleaned_df.loc[combined_outliers, VALUE_COL] = np.nan
    
    # 再次插值填充（使用时间序列感知的方法）
    cleaned_df[VALUE_COL] = cleaned_df[VALUE_COL].interpolate(method='time')
    
    # 确保夜间值在合理范围 (0附近)
    night_mask = (cleaned_df['小时'] < 6) | (cleaned_df['小时'] > 19)
    cleaned_df.loc[night_mask, VALUE_COL] = np.clip(
        cleaned_df.loc[night_mask, VALUE_COL], 0, 5
    )
    
    # 移除辅助列
    cleaned_df.drop(columns=['小时', '周天', '年天', '原始值'], axis=1, inplace=True)
    
    return cleaned_df

# 对原数据集进行数据清洗
print("2. 数据清洗与异常值检测...")
cleaned_data = clean_data(data)

# 3. 构建精准的日夜分离模型
def build_model(df):
    """构建平滑过渡的辐照强度预测模型"""
    
    # 提取重要时间特征
    days_from_start = (df.index - df.index[0]).days.values  # 相对于起始日的天数
    hour_of_day = (df.index.hour + df.index.minute/60).values  # 一天中的小数小时
    values = df[VALUE_COL].values
    
    # 安全处理数据
    safe_days = np.clip(days_from_start, -365, 365)
    safe_hours = np.clip(hour_of_day, 0, 24)
    safe_values = np.nan_to_num(values, nan=0.0, posinf=1000, neginf=0.0)
    safe_values = np.clip(safe_values, 0, 2000)
    
    # 定义更精细的辐照强度模型
    def irradiance_model(x, *params):
        """
        更精细的辐照强度模型
        x: [days_from_start, hour_of_day]
        params: 模型参数
        """
        
        d, h = x[0], x[1]
        
        # 解析参数
        # 日变化参数
        a_h = params[0]    # 二次项系数
        b_h = params[1]    # 一次项系数
        peak_h = params[2] # 高峰小时
        
        # 季节变化参数
        a_d = params[3]    # 季节振幅
        phase_d = params[4] # 季节相位
        offset_d = params[5] # 季节基准值
        
        # 日变化模型 (二次函数)
        h_centered = h - peak_h
        day_component = a_h * h_centered**2 + b_h * h_centered
        
        # 季节变化模型 (正弦函数)
        season_component = a_d * np.sin(2 * np.pi * d/365 + phase_d) + offset_d
        
        # 组合模型 - 使用sigmoid确保正值
        combined = 1 / (1 + np.exp(-(day_component + season_component)))
        
        # 应用日出日落平滑过渡
        # 计算日出日落时间 (8:00-20:00简化模型)
        sunrise = 8.0
        sunset = 18.0
        
        # 日出前过渡 (6:00-8:00)
        if h < sunrise:
            # 日出前2小时开始平滑上升
            transition = max(0, min(1, (h - (sunrise - 2)) / 2))
            combined *= transition
        
        # 日落后过渡 (20:00-22:00)
        elif h > sunset:
            # 日落2小时后完全黑暗
            transition = max(0, min(1, ((sunset + 2) - h) / 2))
            combined *= transition
        
        # 限制最大值在1000 w/m²内
        return np.minimum(combined * 1000, 1000)
    
    # 初始参数估计 - 基于领域知识
    initial_params = [
        -0.5,    # a_h (二次项系数)
        15.0,    # b_h (一次项系数)
        13.0,    # peak_h (高峰小时)
        0.3,     # a_d (季节振幅)
        0.0,     # phase_d (季节相位)
        1.5      # offset_d (季节基准值)
    ]
    
    # 数据点矩阵
    x_data = np.vstack([safe_days, safe_hours])
    
    # 拟合模型 - 使用更健壮的优化方法
    try:
        params, _ = optimize.curve_fit(
            irradiance_model, 
            x_data, 
            safe_values,
            p0=initial_params,
            bounds=(
                [-2.0, 0, 10.0, 0.1, 0, 0.5],   # 下限
                [0, 30.0, 15.0, 1.0, 2*np.pi, 3.0]  # 上限
            ),
            maxfev=20000
        )
        print("模型拟合成功")
    except Exception as e:
        print(f"模型拟合失败: {e}")
        print("使用初始参数作为回退")
        params = initial_params
    
    # 输出拟合参数
    print(f"拟合参数: a_h={params[0]:.4f}, b_h={params[1]:.4f}, peak_h={params[2]:.2f}")
    print(f"          a_d={params[3]:.4f}, phase_d={params[4]:.4f}, offset_d={params[5]:.4f}")
    
    # 评估模型在训练数据上的表现
    try:
        pred_values = irradiance_model(x_data, *params)
        r2 = r2_score(safe_values, pred_values)
        print(f"模型R²分数: {r2:.4f}")
    except Exception as e:
        print(f"评估失败: {e}")
        r2 = 0
    
    # 定义预测函数
    def predict_func(target_time):
        """支持单点/多点预测的精细函数"""
        
        if not isinstance(target_time, pd.DatetimeIndex):
            target_time = pd.DatetimeIndex([target_time])
        
        # 计算相对于起始日的天数
        days_from_start = (target_time - df.index[0]).days.values
        
        # 提取小时数
        hour_of_day_pred = (target_time.hour + 
                            target_time.minute/60 + 
                            target_time.second/3600)
        
        # 准备输入数据
        safe_days_pred = np.clip(days_from_start, -365, 365)
        safe_hours_pred = np.clip(hour_of_day_pred, 0, 24)
        x_pred = np.vstack([safe_days_pred, safe_hours_pred])
        
        # 预测辐照强度
        try:
            pred = irradiance_model(x_pred, *params)
        except Exception as e:
            print(f"预测出错: {e}")
            # 简单回退：基于时间的粗略估计
            # 白天：根据小时估算，夜间：0
            pred = np.where(
                (safe_hours_pred >= 6) & (safe_hours_pred <= 18),
                500 * np.sin(np.pi * (safe_hours_pred - 6) / 12),
                0
            )
        
        # 确保非负值
        pred = np.maximum(pred, 0)
        
        # 返回结果
        if len(pred) == 1:
            return float(pred[0])
        return pred
    
    return predict_func

# 构建预测函数
print("3. 构建周期性函数模型...")
solar_predict_function = build_model(cleaned_data)

## 4. 生成整小时预测数据集
def gen_hourly_estimates(df: pd.DataFrame, predict_func: Callable) -> pd.DataFrame:
    """生成整小时刻度的估计数据集"""
    
    # 确定时间范围（扩展到整小时）
    start_time = df.index[0].floor('H') # 开始时间前推到整点
    end_time = df.index[-1].ceil('H')   # 结束时间后推到整点
    
    # 生成每小时时间序列
    hourly_index = pd.date_range(start=start_time, end=end_time, freq='H')
    
    # 预测每个小时的值（批量预测提高效率）
    estimated_values = predict_func(hourly_index)
    
    # 创建DataFrame
    estimated_data = pd.DataFrame({
        TIME_COL: hourly_index,
        ESTIMATED_VALUE_COL: estimated_values
    })
    estimated_data.set_index(TIME_COL, inplace=True)
    
    return estimated_data

# 生成整小时估计数据集
print("4. 生成整小时预测数据集...")
estimated_data = gen_hourly_estimates(cleaned_data, solar_predict_function)

## 5. 结果可视化与验证
def plot_results(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    estimated_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """绘制完整结果分析图"""
    
    fig, axs = plt.subplots(3, 1, figsize=(16, 18), sharex=True)
    
    # 数据清洗前后对比
    axs[0].plot(original_df.index, original_df[VALUE_COL],
                'b-', alpha=0.5, label='原始数据')
    axs[0].plot(cleaned_df.index, cleaned_df[VALUE_COL],
                'r-', alpha=0.5, label='清洗后数据')
    axs[0].set_title('辐照强度数据清洗前后对比')
    axs[0].set_ylabel(VALUE_COL)
    axs[0].grid(True)
    axs[0].legend()
    
    # 整小时预测结果
    axs[1].plot(cleaned_df.index, cleaned_df[VALUE_COL],
                'bo-', markersize=4, label='清洗后数据点')
    axs[1].plot(estimated_df.index, estimated_df[ESTIMATED_VALUE_COL],
                'g-', linewidth=1.5, label='整小时估计')
    axs[1].set_title('整小时估计结果')
    axs[1].set_ylabel(VALUE_COL)
    axs[1].grid(True)
    axs[1].legend()
    
    # 日周期特征分析
    # 添加小时列
    cleaned_df['小时'] = cleaned_df.index.hour + cleaned_df.index.minute/60
    estimated_df['小时'] = estimated_df.index.hour + estimated_df.index.minute/60
    
    # 绘制箱线图展示小时分布 - 修改了这部分
    # 创建统一的48个时间区间
    bins = np.linspace(0, 24, 49)  # 创建49个边界点定义48个区间
    
    # 分组计算中位数 - 使用相同的时间区间
    actual_medians, _ = np.histogram(
        cleaned_df['小时'], bins=bins, 
        weights=cleaned_df[VALUE_COL],
        density=False
    )
    actual_counts, _ = np.histogram(cleaned_df['小时'], bins=bins)
    actual_medians = np.where(actual_counts > 0, actual_medians / actual_counts, 0)
    
    estimated_medians, _ = np.histogram(
        estimated_df['小时'], bins=bins, 
        weights=estimated_df[ESTIMATED_VALUE_COL],
        density=False
    )
    estimated_counts, _ = np.histogram(estimated_df['小时'], bins=bins)
    estimated_medians = np.where(estimated_counts > 0, estimated_medians / estimated_counts, 0)
    
    # 计算每个时间区间的中值位置
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 绘制中位数的平滑曲线
    axs[2].plot(bin_centers, actual_medians,
                'b^', label='实际中位数', alpha=0.7)
    axs[2].plot(bin_centers, estimated_medians,
                'r^', label='预测中位数', linewidth=2)
    axs[2].set_title('日周期特征分析')
    axs[2].set_xlabel('一天中的时间（小时）')
    axs[2].set_ylabel(VALUE_COL)
    axs[2].legend()
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 返回小时数据以便进一步分析
    return cleaned_df[['小时', VALUE_COL]], estimated_df[['小时', ESTIMATED_VALUE_COL]]

# 绘制数据集的处理与分析结果
print("5. 结果可视化与验证...")
actual_hourly, predicted_hourly = plot_results(data, cleaned_data, estimated_data)

# 输出统计信息
print("\n清洗后数据统计信息:")
print(cleaned_data[VALUE_COL].describe())

print("\n整小时估计数据集统计信息:")
print(estimated_data[ESTIMATED_VALUE_COL].describe())

# 保存估计数据集到csv文件
filename = f"附件/电站{DIANZHAN_ID}环境检测仪数据_估计.csv"
estimated_data.to_csv(filename)
print("整小时估计数据集已保存至 " + filename)

# 模型性能评估
def evaluate_model(actual_df: pd.DataFrame, pred_df: pd.DataFrame) -> Tuple[float, float, float]:
    """评估模型在已知数据点上的性能"""
    
    # 创建合并DataFrame
    merged = actual_df.merge(
        pred_df.reset_index().rename(columns={'index': TIME_COL}),
        on=TIME_COL,
        suffixes=('_actual', '_pred')
    )
    
    # 计算性能指标
    mae = np.mean(np.abs(merged[VALUE_COL] - merged[ESTIMATED_VALUE_COL]))
    rmse = np.sqrt(np.mean((merged[VALUE_COL] - merged[ESTIMATED_VALUE_COL]) ** 2))
    r2 = 1 - np.sum((merged[VALUE_COL] - merged[ESTIMATED_VALUE_COL]) ** 2) / np.sum(
        (merged[VALUE_COL] - merged[VALUE_COL].mean()) ** 2
    )
    
    print("\n模型性能评估:")
    print(f"平均绝对误差 (MAE): {mae:.2f} w/m²")
    print(f"均方根误差 (RMSE): {rmse:.2f} w/m²")
    print(f"R²分数: {r2:.4f}")
    
    # 误差分布图
    plt.figure(figsize=(12, 6))
    errors = merged[VALUE_COL] - merged[ESTIMATED_VALUE_COL]
    sns.histplot(errors, kde=True)
    plt.title('预测误差分布')
    plt.xlabel('误差 (w/m²)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return mae, rmse, r2

# 评估模型
_ = evaluate_model(cleaned_data, estimated_data)
