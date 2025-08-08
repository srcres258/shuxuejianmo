import pandas as pd

DIANZHAN_ID: int

print("电站估计数据合并程序")
DIANZHAN_ID = int(input("请输入要处理的电站编号："))
if DIANZHAN_ID not in range(1, 5):
    print("输入错误，电站编号为1~4，请重新输入！")
    exit(1)
    
print("将合并电站" + str(DIANZHAN_ID) + "的所有估计数据")

# 1. 读取数据
print("1. 读取数据")

df1 = pd.read_csv(f"附件/电站{DIANZHAN_ID}发电数据_估计.csv", parse_dates=['时间'])
df2 = pd.read_csv(f"附件/电站{DIANZHAN_ID}环境检测仪数据_估计.csv", parse_dates=['时间'])
df3 = pd.read_csv(f"附件/电站{DIANZHAN_ID}天气数据_估计.csv", parse_dates=['时间'])

# 2. 合并数据
print("2. 合并数据")

merged_df = df1
for df in [df2, df3]:
    merged_df = pd.merge(merged_df, df, on='时间', how='outer')
    
# 3. 清洗合并后的数据
print("3. 清洗合并后的数据")

merged_df = merged_df.drop_duplicates()
merged_df = merged_df.dropna()

# 4. 保存合并后的数据
print("4. 保存合并后的数据")

merged_df.to_csv(f"附件/电站{DIANZHAN_ID}_估计_汇总.csv", index=False)
