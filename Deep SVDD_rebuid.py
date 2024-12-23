
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
use_cuda=False


class DeepSVDD(nn.Module):
    def __init__(self, input_dim, representation_dim=32):
        super(DeepSVDD, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, representation_dim)
        )

    def forward(self, x):
        return self.net(x)

# 计算初始中心 c
def init_center_c(model, dataloader):
    n_samples = 0
    representation_dim = model.net[-1].out_features
    c = torch.zeros(representation_dim, device="cpu")
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            inputs = data[0]
            outputs = model(inputs)
            outputs = outputs.mean(dim=1)
            n_samples += outputs.size(0)
            c += torch.sum(outputs, dim=0)
    c /= n_samples
    np.save("svdd_c.npy",c)
    if use_cuda:
        c = c.cuda()
    return c



index=np.array([5])
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




if __name__ == '__main__':
    #获取JSON数据
    input_data = '{"batchSize":64,"beginTime":"2022-01-01 00:00:00",' \
                 '"ceshiFilePath":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZVx0ZXN0LmNzdg==","ceshiTimeFilePath":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZVxjZXNoaV90aW1lLmpzb24=",' \
                 '"cols":[0,1,2,3,4,5,6,7,8,9,10,11,12],' \
                 '"dataFun":"biaozhun",' \
                 '"endTime":"2022-03-01 00:00:00",' \
                 '"epoch":20,"fileName":"31813.txt",' \
                 '"idn":["0","2","3","6","9","10","12"],\
            "idnname":["参数12"],' \
                 '"jinduPath":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZVxhLmpzb24=",' \
                 '"model":"svdd","modelFileName":"svdd_20_300_2_64_64_biaozhun.txt",' \
                 '"modelFileNameInfo":"svdd_20_300_2_64_64_biaozhun",' \
                 '"read_path":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZVx0cmFpbi5jc3Y=",' \
                 '"sampling":1000,' \
                 '"save_single_path":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZQ==",' \
                 '"stt":2,' \
                 '"trainDateInfos":[{"beginTime":"2022-01-16 00:00:00","endTime":"2022-03-01 00:00:00"},' \
                 '{"beginTime":"2022-2-12 00:00:00","endTime":"2022-03-02 00:00:00"}],' \
                 '"wd":300,"xlType":"2","ycc":64}'

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
        readpath = base64.b64decode(json_data.get("read_path")).decode("utf-8")
        savepath_pt = base64.b64decode(json_data.get("save_single_path")).decode("utf-8") + r'/' + json_data.get(
            "modelFileNameInfo") + '_model.pt'
        savepath_loss_tr = base64.b64decode(json_data.get("save_single_path")).decode("utf-8") + r'/' + json_data.get(
            "modelFileNameInfo") + '_loss_tr.npy'
        savepath_tmax = base64.b64decode(json_data.get("save_single_path")).decode("utf-8") + r'/' + json_data.get(
            "modelFileNameInfo") + '_tmax.npy'
        savepath_tmin = base64.b64decode(json_data.get("save_single_path")).decode("utf-8") + r'/' + json_data.get(
            "modelFileNameInfo") + '_tmin.npy'
        savepath_tmean = base64.b64decode(json_data.get("save_single_path")).decode("utf-8") + r'/' + json_data.get(
            "modelFileNameInfo") + '_tmean.npy'
        savepath_tstd = base64.b64decode(json_data.get("save_single_path")).decode("utf-8") + r'/' + json_data.get(
            "modelFileNameInfo") + '_tstd.npy'
        savejsonpath = base64.b64decode(json_data.get("jinduPath")).decode("utf-8")
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
        save_json['model'] = 'biaozhun'
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
        freetrain0, freeLabel0 = make_samples(freetrain, label=0, step=STEP,window=WIN)

        # 创建了TensorDataset实例
        Train_DS = TensorDataset(torch.from_numpy(freetrain0).float(), torch.from_numpy(freeLabel0).long())
        # 创建了TensorLoader实例
        Train_DL = DataLoader(Train_DS, shuffle=True, batch_size=BATCH)


        svdd = DeepSVDD(input_dim=win_len)
        c = init_center_c(svdd, Train_DL)
        if use_cuda:
            svdd = svdd.cuda()

        #声明优化器
        optimizer = torch.optim.Adam(svdd.parameters(), lr=1e-3)

        #存储每个epoch损失
        for epoch in range(EPOCH):
            for data in Train_DL:
                inputs = data[0]
                if use_cuda:
                    inputs = inputs.cuda()
                outputs = svdd(inputs)
                outputs = outputs.mean(dim=1)  # 聚合
                if use_cuda:
                    outputs = outputs.cuda()
                dist = torch.sum((outputs - c) ** 2, dim=1)
                loss = torch.mean(dist)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch + 1}/{EPOCH}], Loss: {loss.item():.4f}')

        # 保存模型
        torch.save(svdd.state_dict(), 'deep_svdd_model.pth')
        print('Deep SVDD模型已保存。')

        print('over')
        svdd.eval()

        X_train_tensor = torch.from_numpy(freetrain0).float()
        svdd = svdd.cpu()

        # 设置异常检测阈值
        with torch.no_grad():
            outputs_train = svdd(X_train_tensor)
            c = c.cpu()
            train_distances = torch.sum((outputs_train - c) ** 2, dim=1).cpu().numpy()
        threshold = np.max(train_distances, axis=0)+3*np.std(train_distances, axis=0) # 设定阈值为训练集距离的95百分位
        np.save("svdd_threshold.npy",threshold)

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
