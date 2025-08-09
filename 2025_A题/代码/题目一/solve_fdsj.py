# -*- coding: utf-8 -*-
"""
光伏电站累计发电量时序分析
- 读取 CSV
- 缺失/异常值清洗
- 基于白天数据构建连续单值函数（样条）
- 按整小时生成预测序列 estimated_data
- 计算 MAE、RMSE、R²
- （可选）输出灰尘影响指数（残差）
"""

import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline, CubicSpline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib

DIANZHAN_ID: int
TIME_COL: str = '时间'
VALUE_COL: str = '累计发电量kwh'
ESTIMATED_VALUE_COL: str = '估计累计发电量kwh'

print("电站【发电数据】处理程序")
DIANZHAN_ID = int(input("请输入要处理的电站编号："))
if DIANZHAN_ID not in range(1, 5):
    print("输入错误，电站编号为1~4，请重新输入！")
    exit(1)
    
print("将处理电站" + str(DIANZHAN_ID) + "的发电数据")

# 设置字体，能够显示中文
matplotlib.rc("font", family='SimSun', weight="bold")

# ----------------------------------------------------------------------
# 1. 读取原始数据
# ----------------------------------------------------------------------
print("1. 读取原始数据")

FILE_PATH = f"附件/电站{DIANZHAN_ID}发电数据.xlsx"

# 读取时直接把时间列解析为 datetime，去掉第一列的索引（若有）
df = pd.read_excel(
    FILE_PATH,
    engine="openpyxl",
    parse_dates=[TIME_COL],
    index_col=TIME_COL,          # 设为索引，后面方便切片
    usecols=lambda c: c not in ["Unnamed: 0", "index"]  # 排除自动产生的列
)

# 确保按时间升序（文件已排好，但保险起见）
df = df.sort_index()

# 重命名列，去掉可能的空格
df.columns = [VALUE_COL]

# ----------------------------------------------------------------------
# 2. 基础缺失值处理
# ----------------------------------------------------------------------
print("2. 基础缺失值处理")

print("原始记录数:", len(df))
df = df.dropna()                     # 删除任何 NaN 行
print("删除缺失值后记录数:", len(df))

# ----------------------------------------------------------------------
# 3. 异常值检测（基于增量功率）
# ----------------------------------------------------------------------
print("3. 异常值检测")

# 计算增量（单位：kWh）
df["增量"] = df[VALUE_COL].diff()

# ① 负增量（累计不应下降） → 直接剔除
neg_mask = df["增量"] < 0

# ② 极端正增量（可能是仪表跳变或数据错误）
# 使用 IQR 判别
q1 = df["增量"].quantile(0.25)
q3 = df["增量"].quantile(0.75)
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
pos_outlier_mask = df["增量"] > upper_fence

# 合并异常掩码
outlier_mask = neg_mask | pos_outlier_mask
print(f"检测到异常点: {outlier_mask.sum()}（负增量 {neg_mask.sum()}, 正异常 {pos_outlier_mask.sum()}）")

# 删除异常点
df_clean = df[~outlier_mask].copy()
df_clean.drop(columns=["增量"], inplace=True)   # 清理临时列
print("清洗后记录数:", len(df_clean))

# ----------------------------------------------------------------------
# 4. 夜间‑白天划分（这里采用 6:00‑20:00 为白天，实际可根据当地日出日落微调）
# ----------------------------------------------------------------------
print("4. 夜间‑白天划分")

df_clean["hour"] = df_clean.index.hour
day_mask = (df_clean["hour"] >= 6) & (df_clean["hour"] <= 20)   # 白天
night_mask = ~day_mask

# 为了让样条在夜间返回 0，单独保存白天数据用于拟合
df_day = df_clean[day_mask]

# ----------------------------------------------------------------------
# 5. 构建连续单值函数模型（样条）
# ----------------------------------------------------------------------
print("5. 构建连续单值函数模型（样条）")

# 将时间转为相对秒数（float），便于数值计算
epoch = pd.Timestamp("1970-01-01")          # Unix epoch，任意固定点均可
t_seconds = (df_day.index - epoch).total_seconds().values
E_day = df_day[VALUE_COL].values

# 采用 UnivariateSpline（可调平滑因子 s），这里让 s 与噪声水平成比例
# s = 0 → 通过所有点（可能过拟合），适当增大 s 可得到更平滑曲线
s_factor = 0.5 * len(df_day)   # 经验值，可自行调参
spline = UnivariateSpline(t_seconds, E_day, s=s_factor)

# 为了在夜间返回 0，定义包装函数
def f_cumulative(t, set_later_half_to_zero=True):
    """
    输入：numpy array / pandas DatetimeIndex（UTC/本地均可，只要对应 epoch）
    输出：对应的累计发电量（kWh），夜间强制为 0
    """
    # 转为秒数
    if isinstance(t, pd.DatetimeIndex) or isinstance(t, pd.Series):
        t_sec = (t - epoch).total_seconds()
    else:
        t_sec = t
    # 计算样条值
    y = spline(t_sec)
    # 夜间强制为 0（依据 hour）
    if isinstance(t, (pd.DatetimeIndex, pd.Series)):
        hour = t.hour
    else:
        # 若是 ndarray，先转为 datetime 再取 hour
        hour = pd.to_datetime(t, unit='s').hour
    print("y type:", type(y))
    if set_later_half_to_zero:
        y[ (hour < 6) | (hour > 20) ] = 0.0
    else:
        y[ hour < 6 ] = 0.0
    # 防止出现负值（数值误差导致）
    y = np.maximum(y, 0.0)
    return y

# ----------------------------------------------------------------------
# 6. 按整小时生成预测数据集（estimated_data）
# ----------------------------------------------------------------------
print("6. 按整小时生成预测数据集")

# 起止时间向下/向上取整到整点
start_time = df_clean.index.min().floor('H')
end_time   = df_clean.index.max().ceil('H')

hourly_index = pd.date_range(start=start_time, end=end_time, freq='H')
hourly_pred  = f_cumulative(hourly_index)

estimated_data = pd.DataFrame({
    TIME_COL: hourly_index,
    ESTIMATED_VALUE_COL: hourly_pred
})
estimated_data.set_index(TIME_COL, inplace=True)

# 修正预测数据集，补全每天晚间时段到第二天00:00:00的数据
# （不应该是0.0，而是维持当天最高发电量）
for i, row in enumerate(estimated_data.itertuples()):
    kwh = row[1]
    if kwh == 0.0 and not str(row[0]).endswith('00:00:00') and i > 0:
        last_kwh = estimated_data.iloc[i - 1, 0]
        if last_kwh > 0.0:
            estimated_data.iloc[i, 0] = last_kwh

print("\n=== 预测数据（前 30 行） ===")
print(estimated_data.head(30))

# ----------------------------------------------------------------------
# 7. 评估模型（在原始（清洗后）时间点上对比）
# ----------------------------------------------------------------------
print("7. 评估模型")

# 取原始清洗后数据对应的时间点的预测值
y_true = df_clean[VALUE_COL]
y_pred = pd.Series(f_cumulative(df_clean.index, set_later_half_to_zero=False), index=df_clean.index)

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)

print("\n=== 模型评估指标 ===")
print(f"MAE  : {mae: .4f} kWh")
print(f"RMSE : {rmse: .4f} kWh")
print(f"R²   : {r2: .6f}")

# ----------------------------------------------------------------------
# 8. 灰尘影响指数 —— 残差序列
# ----------------------------------------------------------------------
print("8. 灰尘影响指数 —— 残差序列")

residual = y_true - y_pred
residual.name = "残差（真实‑预测）"

# 简单绘图（可帮助判断是否存在系统性衰减）
plt.figure(figsize=(12,4))
plt.plot(residual.index, residual.values, label='残差')
plt.axhline(0, color='k', linestyle='--')
plt.title('残差（真实累计发电量 - 预测）')
plt.xlabel('时间')
plt.ylabel('kWh')
plt.legend()
plt.tight_layout()
plt.show()

# 若想得到每日的灰尘指数（例如每日最大残差），可以：
daily_dust_index = residual.resample('D').max()
print("\n每日灰尘影响指数（最大残差）示例：")
print(daily_dust_index.head())

# ----------------------------------------------------------------------
# 9. 保存结果（可选）
# ----------------------------------------------------------------------
print("9. 保存结果")

print("正在将估计和残差数据保存到 CSV 文件中...")
estimated_data.to_csv(f"附件/电站{DIANZHAN_ID}发电数据_估计.csv")
residual.to_csv(f"附件/电站{DIANZHAN_ID}发电数据_残差.csv")
