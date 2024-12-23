import pandas as pd
import numpy as np
import base64
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import sys
import json
import traceback
import time

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


WIN=200
STEP=2
EPOCH=10
BATCH=64
HIDDEN=64
index=np.array([5])
win_len=index.__len__()
savepath_loss_tr='31813_loss_tr.npy'
readpath_tmax='31813_tmax.npy'
readpath_tmin='31813_tmin.npy'
readpath_tmean='31813_tmean.npy'
readpath_tstd='31813_tstd.npy'


TIME=[['2022-04-01 13:46:39','2022-07-19 03:33:19']]
datamodel='biaozhun'


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Autoencoder, self).__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        s0, s1, s2 = x.shape[0], x.shape[1], x.shape[2]
        x = x.reshape(s0, s1 * s2)
        # 编码
        x = self.encoder(x)

        # 解码
        x = self.decoder(x)
        x = x.reshape(s0, s1, s2)
        return x




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


def find_true_segments0(mask_df, zhengchang='normal', shijian='年月日时分秒'):
    segments = []
    start_time = None
    for index, row in mask_df.iterrows():
        value = row[zhengchang]
        time = str(row[shijian])

        if value == 1 and start_time is None:
            start_time = time
        elif value == 0 and start_time is not None:
            segments.append([start_time, time])
            start_time = None

    if start_time is not None:
        segments.append([start_time, str(mask_df.iloc[-1][shijian])])  # 使用最后一行的时间作为结束时间

    return segments


def trainLoss(trainSet, lstm, loss_func, use_cuda):
    loss_tr = []
    k=0
    for xval,label in trainSet:
        k=k+1
        jindu=round(k/trainSet.__len__()*100,2)
        save_json['jindu'] = jindu
        f = open(savejsonpath, 'w')
        f.write(json.dumps(save_json, ensure_ascii=False))
        f.close()
        if use_cuda:
            xval = xval.cuda()
        start_time = time.time()
        val_output = lstm(xval)
        end_time = time.time()
        inference_time_seconds = end_time - start_time
        inference_time_milliseconds = inference_time_seconds * 1000
        print(f"Forward inference time: {inference_time_milliseconds:.4f} ms")
        print(val_output)
        loss_val = loss_func(val_output, xval)
        loss_tr.append(loss_val.cpu().data.numpy())
    loss_tr = np.array(loss_tr, dtype=float)
    return loss_tr


if __name__ == '__main__':
    # input_data = sys.stdin.read()
    input_data ='{"batchSize":1,"beginTime":"2022-01-01 00:00:00",' \
                '"ceshiFilePath":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZVx0ZXN0LmNzdg==","ceshiTimeFilePath":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZVxjZXNoaV90aW1lLmpzb24=",' \
                '"cols":[0,1,2,3,4,5,6,7,8,9,10,11,12],' \
                '"dataFun":"biaozhun",' \
                '"endTime":"2022-03-01 00:00:00",' \
                '"epoch":20,"fileName":"31813.txt",' \
                '"idn":["0","2","3","6","9","10","12"],\
            "idnname":["参数12"],' \
                '"jinduPath":"QzpcVXNlcnNcQWRtaW5pc3RyYXRvclxEZXNrdG9wXGhlbHBccHl0aG9uRmlsZVxhLmpzb24=",' \
                '"model":"LSTM-prediction","modelFileName":"LSTM-prediction_20_300_2_64_64_biaozhun.txt",' \
                '"modelFileNameInfo":"LSTM-prediction_20_300_2_64_64_biaozhun",' \
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
        WIN = json_data.get('wd') if json_data.get('wd') != None else 200
        STEP = json_data.get('stt') if json_data.get('stt') != None else 2
        EPOCH = json_data.get('epoch') if json_data.get('epoch') != None else 10
        BATCH = json_data.get('batchSize') if json_data.get('batchSize') != None else 64
        HIDDEN = json_data.get('ycc') if json_data.get('ycc') != None else 64
        index = np.array(json_data.get('idn')).astype('int')+2
        win_len = index.__len__()
        use_cuda = False
        readpath = base64.b64decode(json_data.get("read_path")).decode("utf-8")
        savepath_pt = base64.b64decode(json_data.get("save_single_path")).decode("utf-8")+r'/'+json_data.get("modelFileNameInfo")+'_model.pt'
        savepath_loss_tr = base64.b64decode(json_data.get("save_single_path")).decode("utf-8")+r'/'+json_data.get("modelFileNameInfo")+'_loss_tr.npy'
        readpath_tmax = base64.b64decode(json_data.get("save_single_path")).decode("utf-8")+r'/'+json_data.get("modelFileNameInfo")+'_tmax.npy'
        readpath_tmin = base64.b64decode(json_data.get("save_single_path")).decode("utf-8")+r'/'+json_data.get("modelFileNameInfo")+'_tmin.npy'
        readpath_tmean = base64.b64decode(json_data.get("save_single_path")).decode("utf-8")+r'/'+json_data.get("modelFileNameInfo")+'_tmean.npy'
        readpath_tstd = base64.b64decode(json_data.get("save_single_path")).decode("utf-8")+r'/'+json_data.get("modelFileNameInfo")+'_tstd.npy'
        savejsonpath=base64.b64decode(json_data.get("jinduPath")).decode("utf-8")
        ceshicsvpath = base64.b64decode(json_data.get("ceshiFilePath")).decode("utf-8")
        ceshilabelpath = base64.b64decode(json_data.get("ceshiTimeFilePath")).decode("utf-8")
        loss_func = nn.MSELoss()
        BATCH = 1

        trainDateInfos = json_data.get('trainDateInfos')
        TIME=[]
        timeInfo = []
        timeInfo.append(json_data.get('beginTime'))
        timeInfo.append(json_data.get('endTime'))
        TIME.append(timeInfo)

        datamodel =  json_data.get('dataFun')
        jindu=0
        xunlian_flag=0
        save_json = {}
        save_json['model']='danwei'
        save_json['zhuangtai'] = 0
        save_json['type'] = 'ceshi'
        save_json['canshu'] =json_data.get('idn')
        save_json['canshu_mingcheng'] = json_data.get('idnname')
        save_json['jindu'] = jindu
        f = open(savejsonpath, 'w')
        f.write(json.dumps(save_json,ensure_ascii=False))
        f.close()

        normalSeg_name = pd.read_csv(readpath, encoding="utf_8")

        df = normalSeg_name
        df['年月日时分秒'] = pd.to_datetime(df['年月日时分秒'])
        time_periods = []

        for start_time, end_time in TIME:
            time_periods.append((pd.to_datetime(start_time), pd.to_datetime(end_time)))

        mask = False
        for start_time, end_time in time_periods:
            mask |= (df['年月日时分秒'] >= start_time) & (df['年月日时分秒'] <= end_time)

        xunlian_data = df[mask]
        freetrain = np.array(xunlian_data)


        if datamodel=='biaozhun':
            tMean=np.load(readpath_tmean)
            tStd = np.load(readpath_tstd)
            freetrain[:, 2:] = (freetrain[:, 2:] - tMean) / tStd
        elif datamodel=='guiyi':
            tMin=np.load(readpath_tmin)
            tMax=np.load(readpath_tmax)
            freetrain[:, 2:] = (freetrain[:, 2:].astype('float') - tMin) / (tMax - tMin)

        freetrain1, freeTrLabel1 = make_samples(freetrain, label=0, step=STEP, window=WIN)
        Test_DS = TensorDataset(torch.from_numpy(freetrain1).float(), torch.from_numpy(freeTrLabel1).long())
        Test_DL = DataLoader(Test_DS, shuffle=False, batch_size=1)

        lstm = Autoencoder(input_dim=WIN*win_len, hidden_dim=HIDDEN)
        lstm.load_state_dict(torch.load(savepath_pt))
        if use_cuda:
            lstm = lstm.cuda()

        loss_tr = trainLoss(Test_DL, lstm, loss_func, use_cuda)

        TH = np.array(loss_tr)
        LOSS_TR = np.load(savepath_loss_tr)
        TH[:] = LOSS_TR
        label = np.zeros(loss_tr.shape)
        label[loss_tr > LOSS_TR] = 1
        window_labels = label
        original_labels = np.zeros(freetrain.__len__(), dtype=int)  # 假设所有标签初始为0，表示正常

        for i, l in enumerate(window_labels):
            # 计算当前窗口的起始和结束位置
            start = int(i * STEP + 0.5 * WIN)
            end = start + WIN

            original_labels[start:end] = l
        print(original_labels)

        # xunlian_data.insert(loc=xunlian_data.shape[1], column='normal2', value=original_labels)
        # seg = find_true_segments0(xunlian_data)
        # xunlian_data.to_csv(ceshicsvpath, encoding="utf_8",index=False)
        # ceshilabel = {}
        # ceshilabel['label'] = seg
        # f=open(ceshilabelpath,'w',encoding='utf-8')
        # f.write(json.dumps(ceshilabel,ensure_ascii=False))
        # f.close()


    except BaseException  as e:

        print(e.with_traceback())
        traceback.print_exc()
        sys.exit(3)
    except (EOFError,IOError,OSError,ValueError,TypeError)  as e:

        print(e.with_traceback())
        traceback.print_exc()
        sys.exit(3)
    finally:
        save_json['zhuangtai'] = 1
        save_json['jindu'] = 100
        f = open(savejsonpath, 'w')
        f.write(json.dumps(save_json, ensure_ascii=False))
        f.close()