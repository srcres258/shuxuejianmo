import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import warnings
import matplotlib

warnings.filterwarnings('ignore')

# 设置字体，能够显示中文
matplotlib.rc("font", family='SimSun', weight="bold")

DIANZHAN_ID: int

print("电站【天气数据】处理程序")
DIANZHAN_ID = int(input("请输入要处理的电站编号："))
if DIANZHAN_ID not in range(1, 5):
    print("输入错误，电站编号为1~4，请重新输入！")
    exit(1)
    
print("将处理电站" + str(DIANZHAN_ID) + "的环境检测仪数据")

# 1. 加载数据
print("1. 加载数据")
df = pd.read_excel(f"附件/电站{DIANZHAN_ID}天气数据.xlsx", engine='openpyxl', parse_dates=['时间', '日出时间', '日落时间'])
print("原始数据类型:", df.dtypes)
print("原始数据形状:", df.shape)
print("\n前5行数据:")
print(df.head())
print("\n数据信息:")
print(df.info())

# 2. 数据清洗
print("2. 数据清洗")
# 处理重复数据
df = df.drop_duplicates(subset=['时间'], keep='first')

# 处理缺失值
print("处理缺失值")
numeric_cols = ['当前温度', '最高温度', '最低温度', '风速', '湿度']
imputer = KNNImputer(n_neighbors=5)
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

categorical_cols = ['天气', '风向']
df[categorical_cols] = df[categorical_cols].fillna(method='ffill')

# 处理异常值
print("处理异常值")
temp_range = (-50, 50)
wind_speed_range = (0, 50)
humidity_range = (0, 100)

df['当前温度'] = df['当前温度'].clip(*temp_range)
df['最高温度'] = df['最高温度'].clip(*temp_range)
df['最低温度'] = df['最低温度'].clip(*temp_range)
df['风速'] = df['风速'].clip(*wind_speed_range)
df['湿度'] = df['湿度'].clip(*humidity_range)

df['当前温度'] = np.where(df['当前温度'] > df['最高温度'], df['最高温度'], df['当前温度'])
df['当前温度'] = np.where(df['当前温度'] < df['最低温度'], df['最低温度'], df['当前温度'])

# 编码分类变量
print("编码分类变量")
le_weather = LabelEncoder()
le_wind = LabelEncoder()
df['天气编码'] = le_weather.fit_transform(df['天气'])
df['风向编码'] = le_wind.fit_transform(df['风向'])

# 3. 时间序列处理与重采样
print("3. 时间序列处理与重采样")
df = df.set_index('时间').sort_index()
start_time = df.index.min().floor('H')
end_time = df.index.max().ceil('H')
full_hourly_index = pd.date_range(start=start_time, end=end_time, freq='H')

# 4. 使用气象学模型进行预测
print("4. 使用气象学模型进行预测")
features = ['当前温度', '最高温度', '最低温度', '风速', '湿度', '天气编码', '风向编码']
predictions = pd.DataFrame(index=full_hourly_index)

for feature in features:
    print(f"\n处理特征: {feature}")
    
    y = df[feature].resample('H').mean()
    y = y.interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    
    try:
        model = SARIMAX(y, order=(1,1,1), seasonal_order=(1,1,1,24))
        results = model.fit(disp=False)
        pred = results.get_prediction(start=start_time, end=end_time, dynamic=False)
        predictions[feature] = pred.predicted_mean
        
        if len(y) > 10:
            train, test = train_test_split(y, test_size=0.2, shuffle=False)
            model_eval = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,24))
            results_eval = model_eval.fit(disp=False)
            forecast = results_eval.get_forecast(steps=len(test))
            
            mse = mean_squared_error(test, forecast.predicted_mean)
            mae = mean_absolute_error(test, forecast.predicted_mean)
            print(f"  {feature} 评估 - MSE: {mse:.4f}, MAE: {mae:.4f}")
            
    except Exception as e:
        print(f"  {feature} 建模失败: {str(e)}")
        predictions[feature] = y.reindex(full_hourly_index).interpolate(method='time').fillna(method='bfill').fillna(method='ffill')

# 确保预测值在训练数据的编码范围内
min_weather_code = df['天气编码'].min()
max_weather_code = df['天气编码'].max()
predictions['天气编码'] = predictions['天气编码'].clip(min_weather_code, max_weather_code)

min_wind_code = df['风向编码'].min()
max_wind_code = df['风向编码'].max()
predictions['风向编码'] = predictions['风向编码'].clip(min_wind_code, max_wind_code)

predictions['天气'] = le_weather.inverse_transform(predictions['天气编码'].round().astype(int))
predictions['风向'] = le_wind.inverse_transform(predictions['风向编码'].round().astype(int))

predictions['日出时间'] = df['日出时间'].resample('D').first().reindex(predictions.index.date).values
predictions['日落时间'] = df['日落时间'].resample('D').first().reindex(predictions.index.date).values

final_df = predictions[['当前温度', '最高温度', '最低温度', '天气', '风向', '风速', '湿度', '日出时间', '日落时间']]
final_df.index.name = '时间'

print("\n预测数据前5行:")
print(final_df.head())

# 5. 可视化对比与模型评估
print("5. 可视化对比与模型评估")
plt.figure(figsize=(15, 10))

plt.subplot(2, 1, 1)
plt.plot(df.index, df['当前温度'], 'bo-', label='原始数据', alpha=0.5)
plt.plot(final_df.index, final_df['当前温度'], 'r-', label='预测数据')
plt.title('当前温度对比')
plt.xlabel('时间')
plt.ylabel('温度(℃)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(df.index, df['风速'], 'go-', label='原始数据', alpha=0.5)
plt.plot(final_df.index, final_df['风速'], 'm-', label='预测数据')
plt.title('风速对比')
plt.xlabel('时间')
plt.ylabel('风速(m/s)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

common_index = df.index.intersection(final_df.index)
if len(common_index) > 0:
    print("\n模型评估:")
    for feature in ['当前温度', '风速', '湿度']:
        if feature in df.columns and feature in final_df.columns:
            y_true = df.loc[common_index, feature]
            y_pred = final_df.loc[common_index, feature]
            
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
            
            print(f"{feature}:")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: {np.sqrt(mse):.4f}")

# 保存预测数据集为CSV文件
filename = f"附件/电站{DIANZHAN_ID}天气数据_估计.csv"
final_df.to_csv(filename, index=True)

print("\n预测数据集已保存为", filename)
