import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
import matplotlib

warnings.filterwarnings('ignore')

# 设置字体，能够显示中文
matplotlib.rc("font", family='SimSun', weight="bold")

# 配置绘图风格
# plt.style.use('seaborn-whitegrid')
# sns.set_palette('colorblind')

# 1. 高级数据预处理
def preprocess_data(df):
    # 复制数据避免修改原始数据
    df = df.copy()
    
    # 转换时间列
    df['时间'] = pd.to_datetime(df['时间'])
    
    # 计算小时发电量
    df['小时发电量'] = df.groupby(df['时间'].dt.date)['估计累计发电量kwh'].diff().fillna(df['估计累计发电量kwh'])
    
    # 提取时间特征
    df['年'] = df['时间'].dt.year
    df['月'] = df['时间'].dt.month
    df['日'] = df['时间'].dt.day
    df['时'] = df['时间'].dt.hour
    
    # 计算日出日落时间差
    df['日出时间'] = pd.to_datetime(df['日出时间'])
    df['日落时间'] = pd.to_datetime(df['日落时间'])
    df['日出时差(小时)'] = (df['日出时间'] - df['时间']).dt.total_seconds() / 3600
    df['日落时差(小时)'] = (df['日落时间'] - df['时间']).dt.total_seconds() / 3600
    
    # 添加太阳高度角特征（简单模拟）
    solar_noon = df.groupby(df['时间'].dt.date)['时'].transform(lambda x: x.mean())
    df['相对太阳高度'] = np.sin(np.pi * (df['时'] - solar_noon) / 15)
    
    # 添加季节特征（根据月份）
    season_map = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}
    df['季节'] = df['月'].map(season_map)
    
    # 添加时间序列特征
    df['年序'] = (df['时间'] - df['时间'].min()).dt.days
    df['日序'] = df.groupby(df['时间'].dt.date).cumcount()
    
    # 选择有光照数据（7:00-18:00且辐照>0）
    daylight_mask = (df['时'] >= 7) & (df['时'] <= 18) & (df['估计辐照强度w/m2'] > 0)
    df_daylight = df[daylight_mask].copy()
    
    return df, df_daylight

# 2. 模型构建与评估
def build_and_evaluate_model(train_df, full_df, daylight_mask):
    # 定义特征和特征类型
    numerical_features = ['估计辐照强度w/m2', '当前温度', '湿度', '风速', 
                          '日出时差(小时)', '日落时差(小时)', '相对太阳高度', '年序', '日序']
    
    categorical_features = ['季节', '天气', '风向']
    
    # 创建数据预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )
    
    # 创建模型（使用随机森林处理非线性关系）
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=200, 
            max_depth=10,
            min_samples_split=5,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        ))
    ])
    
    # 准备训练数据（使用清洗后的第一天数据）
    train_mask = (full_df['时间'].dt.date == pd.to_datetime('2024-05-01').date()) & daylight_mask
    X_train = full_df.loc[train_mask]
    y_train = full_df.loc[train_mask, '小时发电量']
    
    # 交叉验证评估模型性能
    cv_scores = cross_val_score(model, X_train, y_train, 
                                cv=5, scoring='neg_root_mean_squared_error',
                                n_jobs=-1)
    print(f"交叉验证RMSE: {-cv_scores.mean():.2f} (±{cv_scores.std():.2f})")
    
    # 模型训练
    model.fit(X_train, y_train)
    
    # 在整个数据集上预测
    X_all = full_df[daylight_mask]
    full_df.loc[daylight_mask, '预测小时发电量'] = model.predict(X_all)
    
    # 按天累计预测发电量
    full_df['预测累计发电量'] = full_df.groupby(full_df['时间'].dt.date)['预测小时发电量'].cumsum()
    
    # 计算残差
    full_df['残差'] = full_df['估计累计发电量kwh'] - full_df['预测累计发电量']
    
    # 计算模型评估指标
    eval_mask = full_df['预测累计发电量'].notna() & (full_df['预测累计发电量'] > 0)
    y_actual = full_df.loc[eval_mask, '估计累计发电量kwh']
    y_pred = full_df.loc[eval_mask, '预测累计发电量']
    
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)
    
    print("\n最终模型评估指标:")
    print(f"MAE = {mae:.2f} kWh")
    print(f"RMSE = {rmse:.2f} kWh")
    print(f"R² = {r2:.4f}")
    
    return full_df, model, mae, rmse, r2

# 3. 高级可视化分析
def visualize_results(full_df, mae, rmse, r2, model):
    # 创建画布
    fig = plt.figure(figsize=(18, 20), constrained_layout=True)
    gs = fig.add_gridspec(4, 3)
    
    # 实际与预测累计发电量对比
    ax1 = fig.add_subplot(gs[0, :])
    for date, group in full_df.groupby(full_df['时间'].dt.date):
        if group['预测累计发电量'].notna().any():
            actual_line, = ax1.plot(group['时间'], group['估计累计发电量kwh'], 'b-', 
                                 alpha=0.7, linewidth=1.5)
            pred_line, = ax1.plot(group['时间'], group['预测累计发电量'], 'r--', 
                                alpha=0.8, linewidth=1.5)
            
            # 填充残差区域（实际低于预测）
            residual_mask = group['估计累计发电量kwh'] < group['预测累计发电量']
            if residual_mask.any():
                ax1.fill_between(group['时间'], group['估计累计发电量kwh'], group['预测累计发电量'],
                             where=residual_mask, color='red', alpha=0.2)
    
    ax1.set_title('每日实际与预测累计发电量对比', fontsize=14, fontweight='bold')
    ax1.set_xlabel('时间')
    ax1.set_ylabel('累计发电量 (kWh)')
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.legend([actual_line, pred_line], ['实际值', '预测值'], loc='upper left')
    
    # 小时发电量对比
    ax2 = fig.add_subplot(gs[1, 0])
    daylight = full_df[full_df['时'].between(7, 18)].copy()
    
    # 计算误差分布
    daylight['误差'] = daylight['小时发电量'] - daylight['预测小时发电量']
    bin_size = 50
    bins = np.arange(daylight['估计辐照强度w/m2'].min(), 
                    daylight['估计辐照强度w/m2'].max() + bin_size, bin_size)
    daylight['辐照强度区间'] = pd.cut(daylight['估计辐照强度w/m2'], bins, labels=bins[:-1] + bin_size/2)
    
    error_stats = daylight.groupby('辐照强度区间')['误差'].agg(['mean', 'std', 'count']).reset_index()
    error_stats = error_stats.dropna()
    
    # 创建散点图
    scatter = ax2.scatter(daylight['估计辐照强度w/m2'], daylight['小时发电量'], 
                       alpha=0.6, s=40, c=daylight['时'], cmap='viridis')
    ax2.scatter(daylight['估计辐照强度w/m2'], daylight['预测小时发电量'], 
             alpha=0.5, s=20, marker='x', c='red')
    
    # 添加误差带
    ax2.errorbar(error_stats['辐照强度区间'], error_stats['mean'], yerr=error_stats['std'],
              fmt='-o', color='purple', alpha=0.8, label='误差带')
    
    ax2.set_title('小时发电量 vs 辐照强度', fontsize=14)
    ax2.set_xlabel('辐照强度 (w/m2)')
    ax2.set_ylabel('小时发电量 (kWh)')
    fig.colorbar(scatter, ax=ax2, label='小时')
    ax2.grid(True, alpha=0.3)
    
    # 残差分析
    ax3 = fig.add_subplot(gs[1, 1])
    residuals = daylight['小时发电量'] - daylight['预测小时发电量']
    
    # 拟合正态分布
    (mu, sigma) = stats.norm.fit(residuals.dropna())
    n_bins = 30
    
    # 绘制直方图和分布曲线
    sns.histplot(residuals, bins=n_bins, kde=False, ax=ax3, stat='density', color='blue', alpha=0.5)
    x = np.linspace(min(residuals), max(residuals), 100)
    ax3.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
             label=f'正态分布\nμ={mu:.2f}, σ={sigma:.2f}')
    
    ax3.set_title(f'残差分布 (MAE={mae:.2f}, RMSE={rmse:.2f})', fontsize=14)
    ax3.set_xlabel('残差 (kWh)')
    ax3.set_ylabel('密度')
    ax3.grid(True, linestyle=':', alpha=0.5)
    ax3.legend()
    
    # 时间维度残差分析
    ax4 = fig.add_subplot(gs[1, 2])
    daily_residual = full_df.groupby(full_df['时间'].dt.date)['残差'].agg(['mean', 'std']).reset_index()
    daily_residual['日期'] = pd.to_datetime(daily_residual['时间'])
    
    # 计算滑动平均值
    daily_residual['7天均值'] = daily_residual['mean'].rolling(7, center=True).mean()
    
    # 绘制残差时间序列
    ax4.plot(daily_residual['日期'], daily_residual['mean'], 'bo-', alpha=0.5, markersize=4, label='每日平均残差')
    ax4.plot(daily_residual['日期'], daily_residual['7天均值'], 'r-', linewidth=2, label='7日移动平均')
    
    # 填充标准差区域
    ax4.fill_between(daily_residual['日期'], 
                  daily_residual['7天均值'] - daily_residual['std'], 
                  daily_residual['7天均值'] + daily_residual['std'], 
                  color='gray', alpha=0.2, label='±标准差')
    
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_title('残差时间序列分析', fontsize=14)
    ax4.set_xlabel('日期')
    ax4.set_ylabel('残差 (kWh)')
    ax4.legend()
    ax4.grid(True, linestyle=':', alpha=0.5)
    
    # 模型评估指标
    ax5 = fig.add_subplot(gs[2, 0])
    metrics = ['MAE', 'RMSE', 'R²']
    values = [mae, rmse, r2]
    colors = ['royalblue', 'orange', 'green']
    
    bars = ax5.bar(metrics, values, color=colors)
    ax5.axhline(y=0, color='k', linewidth=0.8)
    ax5.set_title('模型评估指标', fontsize=14)
    ax5.set_ylabel('数值')
    
    # 添加数值标签
    for bar, v in zip(bars, values):
        height = bar.get_height()
        sign = '+' if v > 0 else ''
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.05*max(values), 
                 f'{sign}{v:.3f}', ha='center', va='bottom', fontsize=12)
    
    # 每小时发电效率分析
    ax6 = fig.add_subplot(gs[2, 1])
    hour_efficiency = daylight.groupby('时')['小时发电量'].agg(['mean', 'std'])
    ax6.errorbar(hour_efficiency.index, hour_efficiency['mean'], yerr=hour_efficiency['std'],
              fmt='o-', capsize=5, color='darkgreen', alpha=0.8)
    
    ax6.set_title('不同小时的平均发电量', fontsize=14)
    ax6.set_xlabel('小时')
    ax6.set_ylabel('小时发电量 (kWh)')
    ax6.set_xticks(range(7, 19))
    ax6.set_xlim(6, 19)
    ax6.grid(True, alpha=0.3)
    
    # 温度-湿度关系
    ax7 = fig.add_subplot(gs[2, 2])
    scatter = ax7.scatter(daylight['当前温度'], daylight['湿度'], 
                       c=daylight['小时发电量'], cmap='viridis', alpha=0.7, s=60)
    
    ax7.set_title('温度-湿度关系与发电量', fontsize=14)
    ax7.set_xlabel('当前温度 (°C)')
    ax7.set_ylabel('湿度 (%)')
    fig.colorbar(scatter, ax=ax7, label='小时发电量')
    ax7.grid(True, alpha=0.3)
    
    # 特征重要性
    ax8 = fig.add_subplot(gs[3, :])
    try:
        feature_importances = model.named_steps['regressor'].feature_importances_
        feature_names = model.named_steps['preprocessor'].get_feature_names_out()
        importance_df = pd.DataFrame({'特征': feature_names, '重要性': feature_importances})
        importance_df = importance_df.sort_values('重要性', ascending=False).head(15)
        
        importance_df.plot(kind='barh', x='特征', y='重要性', 
                        ax=ax8, legend=False, color='royalblue')
        ax8.set_title('特征重要性', fontsize=14, fontweight='bold')
        ax8.set_xlabel('重要性分数')
        ax8.grid(True, alpha=0.3)
    except Exception as e:
        print(f"无法提取特征重要性: {e}")
        ax8.set_visible(False)
    
    # plt.tight_layout()
    plt.savefig('photovoltaic_model_results_advanced.png', dpi=300)
    plt.show()

# 主程序
def main():
    print("===== 光伏发电模型分析 =====")
    
    # 1. 数据读取
    print("读取数据...")
    try:
        df = pd.read_csv('附件/电站1_估计_汇总.csv', parse_dates=['时间'])
        print(f"成功读取数据: {len(df)}行")
    except Exception as e:
        print(f"数据读取失败: {e}")
        return
    
    # 2. 数据预处理
    print("预处理数据...")
    full_df, daylight_df = preprocess_data(df)
    
    # 创建日光时段掩码
    daylight_mask = full_df.index.isin(daylight_df.index)
    
    # 3. 模型构建与评估
    print("构建模型...")
    result_df, model, mae, rmse, r2 = build_and_evaluate_model(
        daylight_df, full_df, daylight_mask
    )
    
    # 4. 可视化分析
    print("生成可视化...")
    visualize_results(result_df, mae, rmse, r2, model)
    
    # 5. 输出结果
    filename = '附件/电站1_估计_汇总_预测.csv'
    print("保存结果...")
    result_df.to_csv(filename, index=False)
    
    print("\n===== 分析完成 =====")
    print(f"保存结果到: {filename}")
    print(f"可视化结果保存为: photovoltaic_model_results_advanced.png")

if __name__ == "__main__":
    main()
    