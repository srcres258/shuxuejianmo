import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import matplotlib

warnings.filterwarnings('ignore')

# 设置字体，能够显示中文
matplotlib.rc("font", family='SimSun', weight="bold")

# 配置环境
plt.style.use('seaborn-whitegrid')
sns.set_palette('coolwarm')
np.random.seed(42)

class DustIndexCalculator:
    def __init__(self, alpha=0.1, reset_threshold=0.3):
        """
        积灰指数计算器
        
        参数:
        alpha: EWMA平滑系数 (0 < alpha < 1)
        reset_threshold: 灰尘指数重置阈值
        """
        self.alpha = alpha
        self.reset_threshold = reset_threshold
        self.max_deviation = 1e-6  # 初始化为小值，避免除以零
        self.last_dust_index = 0
        self.last_cleaning_date = None
        
    def calculate_dust_index(self, df, env_features=['当前温度', '湿度', '风速']):
        """
        计算积灰指数
        
        参数:
        df: 包含预测和实际发电量的DataFrame
        env_features: 用于环境校正的特征列表
        
        返回:
        添加了Dust_Index列的DataFrame
        """
        # 创建副本避免修改原始数据
        df = df.copy()
        
        # 1. 计算相对偏差（避免除以零）
        df['相对偏差'] = np.where(
            df['预测累计发电量'] > 0,
            (df['预测累计发电量'] - df['估计累计发电量kwh']) / df['预测累计发电量'],
            0  # 当预测发电量为零时，偏差设为零
        )
        df['相对偏差'] = df['相对偏差'].clip(lower=0)  # 只考虑负偏差（效率损失）
        
        # 2. 环境因素校正
        if env_features and all(feat in df.columns for feat in env_features):
            # 使用环境因素校正偏差
            df = self._environmental_correction(df, env_features)
        else:
            # 如果没有环境特征，使用原始相对偏差
            df['校正偏差'] = df['相对偏差']
        
        # 3. 计算每日最大偏差（作为归一化基准）
        daily_max = df.groupby(df['时间'].dt.date)['校正偏差'].max().reset_index()
        daily_max.columns = ['日期', '日最大偏差']
        df = pd.merge(df, daily_max, left_on=df['时间'].dt.date, right_on='日期')
        
        # 更新全局最大偏差（避免零值）
        current_max = df['校正偏差'].max()
        if current_max > self.max_deviation:
            self.max_deviation = max(current_max, 1e-6)  # 确保至少为1e-6
        
        # 4. 计算初始灰尘指数（避免除以零）
        df['初始灰尘指数'] = df['校正偏差'] / self.max_deviation
        df['初始灰尘指数'] = df['初始灰尘指数'].clip(0, 1)  # 确保在[0,1]范围内
        
        # 5. 应用时间序列平滑（EWMA）
        df = self._apply_ewma(df)
        
        # 6. 检测清洗事件并重置指数
        df = self._detect_cleaning_events(df)
        
        return df
    
    def _environmental_correction(self, df, features):
        """使用环境因素校正偏差，避免NaN值"""
        # 创建环境校正模型
        # 选择有效数据点（偏差大于零）
        valid_mask = df['相对偏差'] > 0
        
        if valid_mask.sum() > 10:  # 确保有足够的数据点
            X = df.loc[valid_mask, features]
            y = df.loc[valid_mask, '相对偏差']
            
            # 处理缺失值
            X = X.fillna(X.mean())
            y = y.fillna(0)
            
            # 训练校正模型
            model = LinearRegression()
            model.fit(X, y)
            
            # 预测环境因素导致的偏差
            df['环境偏差'] = model.predict(df[features])
            
            # 计算校正后的偏差
            df['校正偏差'] = df['相对偏差'] - df['环境偏差']
            df['校正偏差'] = df['校正偏差'].clip(lower=0)  # 确保非负
        else:
            # 如果没有足够数据，使用原始偏差
            df['校正偏差'] = df['相对偏差']
        
        return df
    
    def _apply_ewma(self, df):
        """应用指数加权移动平均，处理NaN值"""
        # 按时间排序
        df = df.sort_values('时间')
        
        # 初始化灰尘指数列
        df['Dust_Index'] = 0.0
        
        # 应用EWMA
        for i in range(len(df)):
            if i == 0:
                # 初始值
                df.loc[df.index[i], 'Dust_Index'] = df.loc[df.index[i], '初始灰尘指数']
            else:
                prev_index = df.loc[df.index[i-1], 'Dust_Index']
                current_index = df.loc[df.index[i], '初始灰尘指数']
                
                # EWMA公式: DI_t = α * current + (1-α) * DI_{t-1}
                df.loc[df.index[i], 'Dust_Index'] = (
                    self.alpha * current_index + (1 - self.alpha) * prev_index
                )
        
        # 确保指数在[0,1]范围内
        df['Dust_Index'] = df['Dust_Index'].clip(0, 1)
        self.last_dust_index = df['Dust_Index'].iloc[-1]
        
        return df
    
    def _detect_cleaning_events(self, df):
        """检测清洗事件并重置指数，避免NaN值"""
        # 按天分组计算平均灰尘指数
        daily_avg = df.groupby(df['时间'].dt.date)['Dust_Index'].mean().reset_index()
        daily_avg.columns = ['日期', '日均灰尘指数']
        
        # 检测可能的清洗事件（灰尘指数显著下降）
        daily_avg['指数变化'] = daily_avg['日均灰尘指数'].diff()
        
        # 当指数下降超过阈值时，标记为清洗日
        cleaning_dates = daily_avg[
            (daily_avg['指数变化'] < -self.reset_threshold) & 
            (daily_avg['日均灰尘指数'] > 0.5)  # 只考虑高灰尘情况
        ]['日期']
        
        if not cleaning_dates.empty:
            last_cleaning = cleaning_dates.max()
            self.last_cleaning_date = last_cleaning
            print(f"检测到清洗事件于: {last_cleaning}")
            
            # 在清洗日后重置灰尘指数
            for date in cleaning_dates:
                mask = df['时间'].dt.date == date
                df.loc[mask, 'Dust_Index'] = df.loc[mask, 'Dust_Index'] * 0.2  # 重置为较低值
        
        return df
    
    def cleaning_recommendation(self, current_index=None):
        """根据当前灰尘指数给出清洗建议"""
        if current_index is None:
            current_index = self.last_dust_index
        
        if current_index > 0.85:
            return "立即清洗: 灰尘指数超过0.85"
        elif current_index > 0.7:
            return "建议清洗: 灰尘指数超过0.7"
        else:
            return "无需清洗: 灰尘指数在正常范围内"
    
    def plot_dust_index(self, df):
        """可视化灰尘指数"""
        plt.figure(figsize=(15, 8))
        
        # 灰尘指数时间序列
        plt.subplot(2, 1, 1)
        plt.plot(df['时间'], df['Dust_Index'], 'b-', label='灰尘指数')
        plt.axhline(y=0.7, color='orange', linestyle='--', label='预警阈值')
        plt.axhline(y=0.85, color='red', linestyle='--', label='清洗阈值')
        
        # 标记清洗事件
        if self.last_cleaning_date:
            plt.axvline(x=pd.to_datetime(self.last_cleaning_date), color='green', 
                        linestyle='--', alpha=0.7, label='上次清洗')
        
        plt.title('光伏板积灰指数时间序列', fontsize=14)
        plt.xlabel('时间')
        plt.ylabel('灰尘指数')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        
        # 灰尘指数分布
        plt.subplot(2, 1, 2)
        sns.histplot(df['Dust_Index'], bins=20, kde=True)
        plt.axvline(x=0.7, color='orange', linestyle='--')
        plt.axvline(x=0.85, color='red', linestyle='--')
        plt.title('灰尘指数分布', fontsize=14)
        plt.xlabel('灰尘指数')
        plt.ylabel('频率')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('dust_index_analysis.png', dpi=300)
        plt.close()

# 主程序
def main():
    print("===== 光伏电站积灰指数分析 =====")
    
    # 1. 读取数据
    print("步骤1/4: 读取数据...")
    try:
        df = pd.read_csv('附件/电站1_估计_汇总_预测.csv', parse_dates=['时间'])
        print(f"数据读取成功: {len(df)}行记录")
        
        # 检查必要列是否存在
        required_cols = ['时间', '估计累计发电量kwh', '预测累计发电量']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"错误: 缺少必要列 {missing}")
            return
    except Exception as e:
        print(f"数据读取失败: {e}")
        return
    
    # 2. 预处理数据
    print("步骤2/4: 数据预处理...")
    # 确保没有NaN值
    df['预测累计发电量'] = df['预测累计发电量'].fillna(0)
    df['估计累计发电量kwh'] = df['估计累计发电量kwh'].fillna(0)
    
    # 3. 计算积灰指数
    print("步骤3/4: 计算积灰指数...")
    dust_calculator = DustIndexCalculator(alpha=0.05, reset_threshold=0.3)
    df = dust_calculator.calculate_dust_index(df)
    
    # 4. 分析清洗建议
    print("步骤4/4: 分析清洗建议...")
    current_index = df['Dust_Index'].iloc[-1] if 'Dust_Index' in df.columns else 0
    recommendation = dust_calculator.cleaning_recommendation(current_index)
    print(f"\n当前灰尘指数: {current_index:.4f}")
    print(f"清洗建议: {recommendation}")
    
    if dust_calculator.last_cleaning_date:
        print(f"上次检测到清洗时间: {dust_calculator.last_cleaning_date}")
    
    # 5. 可视化与结果保存
    dust_calculator.plot_dust_index(df)
    
    # 保存结果
    filename = '附件/电站1_估计_汇总_预测_灰尘指数.csv'
    df.to_csv(filename, index=False)
    print("\n结果已保存至: " + filename)
    print("可视化已保存为: dust_index_analysis.png")
    
    # 预警分析
    if 'Dust_Index' in df.columns:
        high_dust_mask = df['Dust_Index'] > 0.7
        if any(high_dust_mask):
            high_dust_days = df[high_dust_mask]['时间'].dt.date.nunique()
            print(f"\n预警: 有 {high_dust_days} 天灰尘指数超过0.7")
            
            # 检测连续高灰尘指数天数
            df['高灰尘日'] = df['Dust_Index'] > 0.7
            df['连续高灰尘'] = df['高灰尘日'].groupby((~df['高灰尘日']).cumsum()).cumsum()
            
            max_consecutive = df['连续高灰尘'].max()
            if max_consecutive >= 3:
                print(f"严重预警: 灰尘指数连续 {max_consecutive} 天超过0.7")
        else:
            print("灰尘指数正常，无需预警")
    else:
        print("警告: 未能计算灰尘指数")

if __name__ == "__main__":
    main()
