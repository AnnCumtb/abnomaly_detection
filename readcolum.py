import pandas as pd

# 读取CSV文件
df = pd.read_csv('train.csv')

# 获取列名
columns = df.columns.tolist()

# 打印列名
print(columns)
