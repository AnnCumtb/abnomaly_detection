import torch.optim as optim
import torch.autograd as autograd
import torch.utils.data as Data
from torch.utils.data import TensorDataset, DataLoader
import scipy.io as scio
import random
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
import sys
import json
import traceback
import copy
from arch.unitroot import ADF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minepy import MINE
import base64
import codecs
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
use_cuda=True




index=np.array([1])
win_len=index.__len__()
readpath_tmean='gan_20_300_2_64_64_biaozhun_tmean.npy'
readpath_tstd='gan_20_300_2_64_64_biaozhun_tstd.npy'
TIME=[['2022-03-01 13:46:39','2022-03-19 03:33:19']]
datamodel='biaozhun'


# 生成器
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

# 判别器
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 滑动窗口获取数据
def find_true_segments(mask):
    segments = []
    start = None
    for i, value in enumerate(mask):
        if value == 1 and start is None:
            start = i
        elif value == 0 and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


# 生成采样数据
def make_samples(data, label, step=1, window=200):
    """
    :param data: 原始数据
    :param label: 标签
    :param step: 每次跳跃的步数
    :param window: 窗口大小，默认300
    :return:
    """
    samples = []
    labels = []
    start = 0
    for i in range(data.shape[0]):
        if start + window > data.shape[0]:
            break
        # 滑动窗口取数据并打标签
        for j in range(1):
            temp = np.copy(data[start:start + window, index])#裁取窗口区域的数据
            samples.append(temp)#是不包含labels的
            labels.append(label)#传入过来的
            start += step #每次滑动窗口的步长
    use_feature = index.shape[0]
    # 新建三维数组，且初始值为0。用于将 list 存储的数据保存成一个三维 ndarray，方便后续训练和使用
    res = np.zeros((samples.__len__(), window, use_feature), dtype='float32')
    i = 0
    for s in samples:
        res[i, :, :] = s
        i += 1
    # 返回三元组和labels的numpy数组。
    return res, np.array(labels, dtype=int)

def find_true_segments0(mask_df, zhengchang='normal', shijian='年月日时分秒'):
    segments = []
    start_time = None
    for index, row in mask_df.iterrows():
        value = row[zhengchang]  # 请将 '你的mask列名称' 替换为实际的列名称
        time = str(row[shijian])  # 请将 '时间坐标列名称' 替换为实际的列名称

        if value == 1 and start_time is None:
            start_time = time
        elif value == 0 and start_time is not None:
            segments.append([start_time, time])
            start_time = None

    if start_time is not None:
        segments.append([start_time, str(mask_df.iloc[-1][shijian])])  # 使用最后一行的时间作为结束时间

    return segments


if __name__ == '__main__':
    #获取JSON数据
    input_data = '{"batchSize":64,"beginTime":"2022-01-01 00:00:00",' \
                 '"ceshiFilePath":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZVx0ZXN0LmNzdg==","ceshiTimeFilePath":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZVxjZXNoaV90aW1lLmpzb24=",' \
                  '"cols":[0,1,2],' \
                 '"dataFun":"biaozhun",' \
                 '"endTime":"2023-03-15 00:00:00",' \
                 '"epoch":20,' \
                 '"idn":["0"],' \
                 '"idnname":["参数9"],' \
                 '"jinduPath":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZVxhLmpzb24=",' \
                 '"model":"gan","modelFileName":"gan_20_300_2_64_64_biaozhun.txt",' \
                 '"modelFileNameInfo":"gan_20_300_2_64_64_biaozhun",' \
                  '"read_path":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZVx0cmFpbi5jc3Y=",' \
                 '"sampling":1000,' \
                 '"save_single_path":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZQ==",' \
                 '"stt":2,' \
                '"trainDateInfos":[{"beginTime":"2022-01-16 00:00:00","endTime":"2022-03-01 00:00:00"},' \
                '{"beginTime":"2022-2-12 00:00:00","endTime":"2022-03-15 00:00:00"}],' \
                 '"wd":50,"xlType":"2","ycc":64}'


    try:
        json_data = json.loads(input_data)
        print(json_data.get('idnname'))

    except ValueError as e:
        print("Invalid JSON data")
        sys.exit(8)

    try:
        #定义一些参数
        WIN = json_data.get('wd') if json_data.get('wd') != None else 200
        STEP = json_data.get('stt') if json_data.get('stt') != None else 2
        EPOCH = json_data.get('epoch') if json_data.get('epoch') != None else 10
        BATCH = json_data.get('batchSize') if json_data.get('batchSize') != None else 64
        HIDDEN = json_data.get('ycc') if json_data.get('ycc') != None else 64
        index = np.array(json_data.get('idn')).astype('int') + 2
        win_len = index.__len__()


        # 采用了base64加密方法不直接暴露文件路径
        #获取文件路径
        readpath = './train2.csv'
        savepath_pt = './gan_20_300_2_64_64_biaozhun_model.pt'
        savepath_loss_tr = './gan_20_300_2_64_64_biaozhun_loss_tr.npy'
        savepath_tmax = './gan_20_300_2_64_64_biaozhun_tmax.npy'
        savepath_tmin = './gan_20_300_2_64_64_biaozhun_tmin.npy'
        savepath_tmean = './gan_20_300_2_64_64_biaozhun_tmean.npy'
        savepath_tstd = './gan_20_300_2_64_64_biaozhun_tstd.npy'
        savejsonpath = 'a.json'
        ceshicsvpath = './train2fake.csv'
        ceshilabelpath = './ceshi_time2.json'

        trainDateInfos = json_data.get('trainDateInfos')



        #重新定义一个time，又从输入里面拿一次
        TIME = []
        timeInfo = []
        timeInfo.append(json_data.get('beginTime'))
        timeInfo.append(json_data.get('endTime'))
        TIME.append(timeInfo)

        #获取数据处理的函数
        datamodel = json_data.get('dataFun')

        #定义参数
        jindu = 0
        save_json = {}
        save_json['model'] = 'danwei'
        save_json['zhuangtai'] = 0
        save_json['type'] = 'xunlian'
        save_json['canshu'] = json_data.get('idn')
        save_json['canshu_mingcheng'] = json_data.get('idnname')
        save_json['jindu'] = jindu

        # 读入训练数据集csv文件
        df = pd.read_csv(ceshicsvpath, encoding="utf-8")
        # 将 "年月日时分秒" 列转换为日期时间类型
        df['年月日时分秒'] = pd.to_datetime(df['年月日时分秒'])
        # 指定起始时间和结束时间
        # 初始化时间段列表
        time_periods = []

        # 迭代 TIME 列表，将每个时间段的起始时间和结束时间添加到 time_periods 列表
        for start_time, end_time in TIME:
            time_periods.append((pd.to_datetime(start_time), pd.to_datetime(end_time)))

        # 创建布尔条件，选择在多个时间段内的行
        mask = False
        for start_time, end_time in time_periods:
            # 检查数据框中 '年月日时分秒' 列的值是否在当前的 start_time 和 end_time 之间。如果在这个范围内，则将对应位置的 mask 设置为 True
            mask |= (df['年月日时分秒'] >= start_time) & (df['年月日时分秒'] <= end_time)

        # 把符合条件的数据给存放到ceshi_data
        ceshi_data = df[mask]

        # 转换为numpy数组
        freetest = np.array(ceshi_data)

        # 根据数据类型做不同的处理
        if datamodel == 'biaozhun':
            tMean = np.load(readpath_tmean)
            tStd = np.load(readpath_tstd)
            freetest[:, 2:] = (freetest[:, 2:] - tMean) / tStd

        elif datamodel == 'guiyi':
            # 计算最小值和最大值
            tMin = np.min(freetest[:, 2:].astype('float'), axis=0)
            tMax = np.max(freetest[:, 2:].astype('float'), axis=0)
            np.save(savepath_tmin, tMin)
            np.save(savepath_tmax, tMax)
            # 归一化数据
            freetest[:, 2:] = (freetest[:, 2:].astype('float') - tMin) / (tMax - tMin)

        # 滑动窗口构造样本
        freetest1, freelabel1 = make_samples(freetest, label=0, step=STEP, window=WIN)
        freetest1_flat = freetest1.reshape(freetest1.shape[0], -1)

        # 创建了TensorDataset实例
        Test_DS = TensorDataset(torch.from_numpy(freetest1).float(), torch.from_numpy(freelabel1).long())
        # 创建了TensorLoader实例
        Test_DL = DataLoader(Test_DS, shuffle=False, batch_size=64)

        # 初始化模型和优化器
        latent_dim = 100

        generator = Generator(latent_dim, win_len*WIN)
        generator.load_state_dict(torch.load('gan_generator.pth'))
        generator.eval()

        discriminator = Discriminator(win_len*WIN)
        discriminator.load_state_dict(torch.load('gan_discriminator.pth'))
        discriminator.eval()

        X_test_tensor = torch.from_numpy(freetest1_flat).float()

        with torch.no_grad():
            generated_samples = generator(torch.randn(X_test_tensor.size(0), latent_dim))
            generated_samples_np = generated_samples.numpy()

        # 通过判别器进行预测
        with torch.no_grad():
            real_predictions = discriminator(X_test_tensor).numpy()
            generated_predictions = discriminator(torch.from_numpy(generated_samples_np)).numpy()

        # 计算异常分数（越低越正常）
        anomaly_scores = 1 - real_predictions.flatten()

        threshold = 0.93  # 可以根据训练数据设置合适的阈值
        label = np.zeros(X_test_tensor.shape[0])

        exceeds_threshold = anomaly_scores > threshold
        #anomalies = np.any(exceeds_threshold)  # 形状为 (n_samples,)
        label[exceeds_threshold] = 1
        window_labels = label
        original_labels = np.zeros(freetest.__len__(), dtype=int)  # 假设所有标签初始为0，表示正常

        # 将异常标签映射回原始数据
        for i, l in enumerate(window_labels):
            # 计算当前窗口的起始和结束位置
            start = int(i * STEP + 0.5 * WIN)
            end = start + WIN

            # 将当前窗口的异常标签映射到原始数据
            original_labels[start:end] = l

        print(ceshi_data.shape[1])
        ceshi_data.insert(loc=ceshi_data.shape[1], column='normal4', value=original_labels)
        seg = find_true_segments0(ceshi_data)
        ceshi_data.to_csv(ceshicsvpath, encoding="utf_8", index=False)
        ceshilabel = {}
        ceshilabel['label'] = seg
        f = open(ceshilabelpath, 'w', encoding='utf-8')
        f.write(json.dumps(ceshilabel, ensure_ascii=False))
        f.close()


    except BaseException as e:
        print(e.with_traceback())
        traceback.print_exc()
        sys.exit(3)

    except (EOFError, IOError, OSError, ValueError, TypeError) as e:
        print(e.with_traceback())
        traceback.print_exc()
        sys.exit(3)
