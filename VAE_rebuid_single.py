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

from new.VAE_rebuid import savepath_pt

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
use_cuda=True


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, latent_dim=2):
        super(VAE, self).__init__()
        # 编码器
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # 解码器
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # 如果数据已标准化，可使用线性激活函数

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc_mu(h1), self.fc_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h2 = self.relu(self.fc2(z))
        return self.sigmoid(self.fc3(h2))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

#损失函数
def loss_function(recon_x, x, mu, logvar):
    # 重构误差
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    # KL 散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD



index=np.array([1])
win_len=index.__len__()
TIME0=[['2022-01-01 03:33:19','2022-02-19 03:33:19'],
      ['2022-02-20 03:33:19','2022-02-25 03:33:19']]
TIME=[['2022-03-01 13:46:39','2022-03-19 03:33:19']]
TIME0.extend(TIME)
datamodel='biaozhun'



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

# 获得训练数据的误差序列
def trainLoss(trainSet, lstm, loss_func, use_cuda):
    loss_tr = []
    k = 0
    for xval, label in trainSet:
        k = k + 1
        jindu = round(50 + k / trainSet.__len__() * 100 / 2, 2)
        save_json['jindu'] = jindu
        f = open(savejsonpath, 'w')
        f.write(json.dumps(save_json, ensure_ascii=False))
        f.close()
        if use_cuda:
            xval = xval.cuda()
        val_output = lstm(xval)
        loss_val = loss_func(val_output, xval) #计算模型输出值和groundtruth的均方差
        loss_tr.append(loss_val.cpu().data.numpy())
    loss_tr = np.array(loss_tr, dtype=float)
    print('train Set loss save success!')
    return loss_tr


if __name__ == '__main__':
    #获取JSON数据
    input_data = '{"batchSize":64,"beginTime":"2022-01-01 00:00:00",' \
                 '"ceshiFilePath":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZVx0ZXN0LmNzdg==","ceshiTimeFilePath":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZVxjZXNoaV90aW1lLmpzb24=",' \
                 '"cols":[0,1,2],' \
                 '"dataFun":"biaozhun",' \
                 '"endTime":"2022-03-01 00:00:00",' \
                 '"epoch":20,' \
                 '"idn":["0"],' \
                 '"idnname":["参数1"],' \
                 '"jinduPath":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZVxhLmpzb24=",' \
                 '"model":"vae","modelFileName":"vae_20_300_2_64_64_biaozhun.txt",' \
                 '"modelFileNameInfo":"vae_20_300_2_64_64_biaozhun",' \
                 '"read_path":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZVx0cmFpbi5jc3Y=",' \
                 '"sampling":1000,' \
                 '"save_single_path":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZQ==",' \
                 '"stt":2,' \
                 '"trainDateInfos":[{"beginTime":"2022-01-16 00:00:00","endTime":"2022-03-01 00:00:00"},' \
                 '{"beginTime":"2022-2-12 00:00:00","endTime":"2022-03-02 00:00:00"}],' \
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
        readpath = './train5.csv'
        savepath_pt = './vae_20_300_2_64_64_biaozhun_model.pt'
        savepath_loss_tr = './vae_20_300_2_64_64_biaozhun_loss_tr.npy'
        savepath_tmax = './vae_20_300_2_64_64_biaozhun_tmax.npy'
        savepath_tmin = './vae_20_300_2_64_64_biaozhun_tmin.npy'
        savepath_tmean = './vae_20_300_2_64_64_biaozhun_tmean.npy'
        savepath_tstd = './vae_20_300_2_64_64_biaozhun_tstd.npy'
        savejsonpath = 'a.json'
        trainDateInfos = json_data.get('trainDateInfos')

        #定义TIME0，是模型训练时候输入给的时间，actually dont know what it for
        TIME0 = []
        for time in trainDateInfos:
            timeInfo = []
            timeInfo.append(time['beginTime'])
            timeInfo.append(time['endTime'])
            TIME0.append(timeInfo)
        print(TIME0)

        #重新定义一个time，又从输入里面拿一次，still dont know what it for
        TIME = []
        timeInfo = []
        timeInfo.append(json_data.get('beginTime'))
        timeInfo.append(json_data.get('endTime'))
        TIME.append(timeInfo)
        TIME0.extend(TIME)

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


        #把上面这些参数重新存到json文件中
        f = open(savejsonpath, 'w')
        f.write(json.dumps(save_json, ensure_ascii=False))
        f.close()

        #读入训练数据集csv文件
        df = pd.read_csv(readpath, encoding="utf_8")
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

        #把符合条件的数据给存放到xunlian_data
        xunlian_data = df[mask]

        #转换为numpy数组
        freetrain = np.array(xunlian_data)


        # 根据数据类型做不同的处理
        if datamodel == 'biaozhun':
            tMean = np.mean(freetrain[:, 2:].astype('float'), axis=0)
            tStd = np.std(freetrain[:, 2:].astype('float'), axis=0)
            tStd[tStd == 0] = 1
            np.save(savepath_tmean, tMean)
            np.save(savepath_tstd, tStd)
            freetrain[:, 2:] = (freetrain[:, 2:] - tMean) / tStd #做了个归一化
            print(tMean)
        elif datamodel == 'guiyi':
            # 计算最小值和最大值
            tMin = np.min(freetrain[:, 2:].astype('float'), axis=0)
            tMax = np.max(freetrain[:, 2:].astype('float'), axis=0)
            np.save(savepath_tmin, tMin)
            np.save(savepath_tmax, tMax)
            # 归一化数据
            freetrain[:, 2:] = (freetrain[:, 2:].astype('float') - tMin) / (tMax - tMin)


        # 滑动窗口构造样本
        freetrain0, freeTrLabel0 = make_samples(freetrain, label=0, step=STEP,window=WIN)

        # 创建了TensorDataset实例
        Train_DS = TensorDataset(torch.from_numpy(freetrain0).float(), torch.from_numpy(freeTrLabel0).long())
        # 创建了TensorLoader实例
        Train_DL = DataLoader(Train_DS, shuffle=True, batch_size=BATCH)

        vae = VAE(input_dim=win_len)
        if use_cuda:
            vae = vae.cuda()

        #声明优化器
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        #存储每个epoch损失
        for epoch in range(EPOCH):
            vae.train()
            total_loss = 0
            for data in Train_DL:
                inputs = data[0]
                if use_cuda:
                    inputs = inputs.cuda()
                optimizer.zero_grad()
                recon_batch, mu, logvar = vae(inputs)
                loss = loss_function(recon_batch, inputs, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(Train_DL.dataset)
            print(f'Epoch [{epoch + 1}/{EPOCH}], Loss: {avg_loss:.4f}')

        print('over')
        torch.save(vae.state_dict(), 'vae_model.pth')
        print('VAE 模型已保存。')

        vae.eval()
        vae.cpu()
        X_train_tensor = torch.from_numpy(freetrain0).float()
        # 设置异常检测阈值
        with torch.no_grad():
            reconstructions_train, _, _ = vae(X_train_tensor)
            train_errors = torch.mean((reconstructions_train - X_train_tensor) ** 2, dim=1).numpy()
        threshold = np.max(train_errors, axis=0)+3*np.std(train_errors, axis=0)  # 设定阈值位
        np.save("vae_threshold.npy",threshold)


    except BaseException as e:
        print(e.with_traceback())
        traceback.print_exc()
        sys.exit(3)

    except (EOFError, IOError, OSError, ValueError, TypeError) as e:
        print(e.with_traceback())
        traceback.print_exc()
        sys.exit(3)

    finally:
        save_json['zhuangtai'] = 1
        save_json['jindu'] = 100
        f = open(savejsonpath, 'w')
        f.write(json.dumps(save_json, ensure_ascii=False))
        f.close()



