# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取Excel文件
# file_path = 'test.csv'  # 你可以替换为实际的文件路径
# df = pd.read_csv(file_path)
#
# # 选择你想要展示的列，比如第一列
# column_index = 16  # 如果你想要第一列，设置为0；第二列为1，依此类推
#
# # 创建图表
# plt.figure(figsize=(10, 6))
# plt.plot(df.iloc[:, column_index])  # 绘制所选列的数据
# plt.title(f'Column {column_index + 1}')  # 设置标题
# plt.xlabel('Index')  # x轴标签
# plt.ylabel('Value')  # y轴标签
#
# # 显示图表
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# # 读取Excel文件
# file_path = 'test.csv'  # 你可以替换为实际的文件路径
# df = pd.read_csv(file_path)
#
# # 选择你想要展示的列，比如第一列
# column_index = 2  # 设置为你要展示的列索引，这里假设展示第一列
# normal_column = 'normal2'  # 'normal'列名，你可以替换为实际的列名
#
# # 获取 'normal' 列值为 1 的区间
# normal_indices = df[df[normal_column] == 1].index  # 找到值为 1 的索引
#
# # 创建图表
# plt.figure(figsize=(10, 6))
# plt.plot(df.iloc[:, column_index], label=f'canshu 9')  # 绘制所选列的数据
# plt.title(f'canshu 9 with Normal Indicator')  # 设置标题
# plt.xlabel('Index')  # x轴标签
# plt.ylabel('Value')  # y轴标签
#
# in_normal_period = False
# start_index = None
# for i in range(len(df)):
#     if df[normal_column].iloc[i] == 1 and not in_normal_period:
#         # 找到一个新的区间开始
#         start_index = i
#         in_normal_period = True
#     elif df[normal_column].iloc[i] == 0 and in_normal_period:
#         # 找到一个区间结束
#         plt.axvline(x=start_index, color='r', linestyle='--', label='Normal Start')  # 竖线起始位置
#         plt.axvline(x=i, color='r', linestyle='--', label='Normal End')  # 竖线结束位置
#         in_normal_period = False
#
# # 处理最后一个区间，如果最后一个区间以1结束
# if in_normal_period:
#     plt.axvline(x=start_index, color='r', linestyle='--', label='Normal Start')  # 竖线起始位置
#     plt.axvline(x=len(df)-1, color='r', linestyle='--', label='Normal End')  # 竖线结束位置
#
# # 显示图表
# plt.legend()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
#
# # 读取Excel文件
# file_path = 'test.csv'  # 你可以替换为实际的文件路径
# df = pd.read_csv(file_path)
#
# # 检查数据
# print(df.head())  # 打印前几行查看数据结构
#
# # 绘制每一列的图表
# fig, axes = plt.subplots(4, 4, figsize=(20, 16))  # 创建4x4的子图
# axes = axes.flatten()  # 将2D数组转换为1D数组，方便索引
#
# for i in range(14):
#     axes[i].plot(df.iloc[:, i])  # 绘制每列的数据
#     axes[i].set_title(f'Column {i+1}')  # 设置标题
#     axes[i].set_xlabel('Index')  # x轴标签
#     axes[i].set_ylabel(f'Value')  # y轴标签
#
# plt.tight_layout()  # 自动调整子图间的间距
# plt.savefig('./save3.png')
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
rcParams['axes.unicode_minus'] = False  # 正常显示负号
# 读取CSV文件
df = pd.read_csv("data2.csv")

# 假设 CSV 文件的列名是 ['param1', 'param2', ..., 'param12', 'status']
params = df.columns[:-1]  # 获取所有参数列（除最后一列 'status'）
status = df['normal']  # 获取 'status' 列

total_data_points = len(df)  # 总的数据点数
abnormal_data_points = len(df[status == 1])  # 异常数据点数
abnormal_ratio = abnormal_data_points / total_data_points * 100  # 异常数据点占比（百分比）

print(f"异常区域占比：{abnormal_ratio:.2f}%")

# 创建一个子图，显示12个参数的变化趋势
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 16))  # 创建6行2列的子图
axes = axes.flatten()  # 将 axes 变成一维数组

for i, param in enumerate(params):
    axes[i].plot(df[param], label=param, color='blue')
    axes[i].set_title(param)
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Value')

    # 标记异常状态区间
    abnormal_indices = status[status == 1].index  # 找出异常的索引
    axes[i].axvspan(abnormal_indices.min(), abnormal_indices.max(), color='red', alpha=0.3, label="Anomaly")

    axes[i].legend()

# 调整布局
plt.tight_layout()
plt.show()
