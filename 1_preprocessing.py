# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-19 11:01:14
@LastEditTime: 2019-09-22 09:36:50
@Update: 
'''
import os
import pywt
import time, datetime
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


from params import GAPTHRESH, IDLETHRESH, MAXSPEEDTHRESH, MAXPOSACC, MAXNEGACC, MINTIME, MINRUNTIME
from dwt_signal_decomposition import plot_signal_decomp, dwtDecompose

def timestamp2unix(timestamp):
    """
    Params:
        timestamp: {str} e.g. `2017/12/18 13:42:17.000.`
    Returns:
        unixTime:  {int}
    Notes:
        将字符串时间戳转换为unix时间戳
    """
    timestamp = timestamp.split('.')[0]
    d = datetime.datetime.strptime(timestamp, "%Y/%m/%d %H:%M:%S")
    unixTime = time.mktime(d.timetuple())
    return unixTime

def readData(filename):
    """
    Params:
        filename: {str} `.xlsx`
    """
    data = np.array(pd.read_excel(filename))
    data[:, 0] = np.array(list(map(timestamp2unix, data[:, 0])))

    return data.astype(np.float)

def getSpeedSequenceFromLongitudeAndLatitude(longitude, latitude):
    """
    Params:
        longitude, latitude: {ndarray(N)}
    Returns:
        speedSeq: {ndarray(N)}
    """
    locations = np.r_[longitude, latitude].reshape(2, -1).T.astype(np.float)
    movement  = locations[1:] - locations[:-1]
    a = np.sum(movement**2, axis=1)
    speedSeq  = np.sqrt(np.sum(movement**2, axis=1))
    speedSeq  = np.r_[0, speedSeq]
    return speedSeq

def cutSpeedSequences(timestamp, speedSeq, gapThresh=GAPTHRESH, speedThresh=IDLETHRESH):
    """
    Params:
        timestamp: {ndarray(N)}
        speedSeq:  {ndarray(N)}
        speedThresh: {float} 怠速阈值
    Returns:
        speedSequences: {list[ndarray(n_length)]}
    """
    ## 类型1：时间不连续
    gapPeriod = np.r_[0, timestamp[1:] - timestamp[:-1] - 1.]
    notContiguousIndex = np.r_[0., np.where(gapPeriod > gapThresh)[0], timestamp.shape[0]].astype(np.int)
    
    speedSequences = []; timestamps = []
    for i in range(notContiguousIndex.shape[0] - 1):

        notContiguousSpeedSeq = speedSeq[notContiguousIndex[i]: notContiguousIndex[i + 1] - 1]

        ## 类型3：长期停车；类型4：长时间堵车、断断续续低速行驶
        isIdle = (notContiguousSpeedSeq < speedThresh).astype(np.int)
        index = np.r_[1, isIdle[1:] - isIdle[:-1]]                                      # 负跳变(-1)：开始加速
        cutIndex = np.where(index == 1)[0]
        cutIndex = np.r_[cutIndex[:-1], cutIndex[1:] + 1].reshape(2, -1).T
        
        ## 切出运动学片段
        timestamps     += list(map(lambda x: timestamp[x[0]: x[1]], cutIndex))
        speedSequences += list(map(lambda x: notContiguousSpeedSeq[x[0]: x[1]],  cutIndex))

    return np.array(speedSequences), np.array(timestamps)

def calFeaturesOfSequence(speedSeq, timeSeq, speedThresh=IDLETHRESH, maxIdle=180, dwtTime=1):
    """
    Params:
        speedSeq: {ndarray(n_length)}
        timeSeq: {ndarray(n_length)}
    """
    # speedSeq = dwtDecompose(speedSeq, dwtTime)
    # --------------------------------------------
    isIdle = (speedSeq < speedThresh).astype(np.int)    # 是否怠速
    index = np.r_[1, isIdle[1:] - isIdle[:-1]]          # 负跳变(-1)：从怠速起步
    idxStart = np.where(index == -1)[0]                 # 起步时间
    if idxStart.shape[0] == 0: 
        return np.zeros(15)
    else:
        idxStart = idxStart[0]

    # --------------------------------------------
    if idxStart > maxIdle:                              # 怠速时间超过180s，剪裁速度值序列
        speedSeq = speedSeq[maxIdle: ]       
        timeSeq  = timeSeq [maxIdle: ]     
        idxStart -= maxIdle
    
    # --------------------------------------------
    speedSeq = speedSeq / 3.6                           # m/s
    # --------------------------------------------
    # speedSeqDec = dwtDecompose(speedSeqDec, dwtTime)
    # speedSeqDec = np.stack([np.r_[speedSeq[0], speedSeq[:-1]], speedSeq, np.r_[speedSeq[1:], speedSeq[-1]]], axis=0).mean(axis=0)   # 均值滤波
    speedSeq = signal.medfilt(speedSeq, (3,))

    # --------------------------------------------
    temp = speedSeq[1:] - speedSeq[:-1]
    timeGap  = np.r_[1., timeSeq[1:] - timeSeq[:-1]]    # s
    accelerate = np.r_[temp[0], temp] / timeGap
    n_sec  = timeSeq.max() - timeSeq.min()
    n_dist = np.sum(speedSeq * timeGap)

    ## ------------------------------------------------------------------------------------
    feature = []
    
    feature += [speedSeq.max()]                         # 0, 峰值速度(m/s)
    feature += [n_dist / (n_sec - idxStart)]            # 1, 平均速度(m/s)
    feature += [speedSeq.std()]                         # 2, 速度标准差
    
    feature += [n_sec - idxStart]                       # 3, 运行时间(s)
    feature += [n_sec]                                  # 4, 时长(s)

    feature += [np.where(accelerate > 0.)[0].shape[0] / n_sec]  # 5, 加速时间百分比(%)
    feature += [np.where(accelerate < 0.)[0].shape[0] / n_sec]  # 6, 减速时间百分比(%)

    feature += [accelerate.max()]                       # 7, 峰值加速度
    feature += [accelerate.min()]                       # 8, 峰值减速度

    temp = accelerate[accelerate > 0.]
    if temp.shape[0] == 0:
        feature += [0]                                  # 9, 平均加速度
        feature += [0]                                  # 10, 加速度标准差
    else:
        feature += [temp.mean()]                        # 9, 平均加速度
        feature += [temp.std()]                         # 10, 加速度标准差
        
    temp = accelerate[accelerate < 0.]
    if temp.shape[0] == 0:
        feature += [0]                                  # 11, 平均减速度
        feature += [0]                                  # 12, 减速度标准差
    else:
        feature += [temp.mean()]                        # 11, 平均减速度
        feature += [temp.std()]                         # 12, 减速度标准差
        
    feature += [idxStart / n_sec]                       # 13, 怠速时间比(%)
    feature += [n_dist   / n_sec]                       # -1, 平均运行速度(m/s)

    # --------------------------------------------
    feature[ 0] *= 3.6   # km/h
    feature[ 1] *= 3.6   # km/h
    feature[-1] *= 3.6   # km/h
    
    feature = np.array(feature)
    return feature

def padSpeedSequences(speedSequences, timestamps):
    """
    Params:
        timestamps, speedSequences: {ndarray(ndarray(n_sequences))}
    """
    ## 插值补齐
    padValue = []; 
    n_sequence = len(speedSequences)
    for i in range(n_sequence):
        _timestamp = timestamps[i]; _speedSequences = speedSequences[i]
        
        _minTimestamp, _maxTimestamp = _timestamp.min(), _timestamp.max()
        x = (_timestamp - _minTimestamp).astype(np.float); y = _speedSequences.astype(np.float)
        f = interp1d(x, y, kind='linear')
        x_ = np.arange(_maxTimestamp - _minTimestamp + 1); y_ = f(x_)

        index = list(map(lambda _x: _x not in x, x_))
        padValue += [y_[index]]

        _speedSequences = y_; _timestamp = x_ + _minTimestamp
        speedSequences[i] = _speedSequences; timestamps[i] = _timestamp
        
    padValue = np.concatenate(padValue)
    return speedSequences, timestamps, padValue

####################################################################################################

gpsSpeedSeqs = []; gpsSpeedFeats = []
for i in range(1, 4):
    
    # filename = 'data/temp.xlsx'
    filename = 'data/file%d.xlsx' % i
    data = readData(filename)
    print(i)
    print("------------------")
    
    timestamp, gpsSpeed = data[:, 0], data[:, 1]

    plt.figure(0)
    plt.title("Histogram of Speed")
    plt.subplot(3, 1, i)
    plt.xlabel("km/h")
    plt.ylabel("Number")
    plt.hist(gpsSpeed, bins=50, edgecolor='white')

    # gpsSpeed = dwtDecompose(gpsSpeed, 2)

    # --------------------------------------
    ## 切分序列
    n_origin = gpsSpeed.shape[0]
    gpsSpeedSeq, timestampSeqs = cutSpeedSequences(timestamp, gpsSpeed)

    # --------------------------------------
    ## 计算特征
    print("Calculating features...")
    gpsSpeedFeat = np.stack(list(map(lambda x: calFeaturesOfSequence(x[0], x[1]), zip(gpsSpeedSeq, timestampSeqs))), 0)
    # --------------------------------------
    n_detele = 0
    
    plt.figure(i + 3, figsize=(8, 12))
    ## 删除峰值速度小于阈值的序列
    plt.subplot(3, 2, 1)
    plt.title("Max Speed"); plt.xlabel("km/h"); plt.ylabel("number")
    plt.hist(gpsSpeedFeat[:, 0], bins=100)
    
    n_samples = gpsSpeedFeat.shape[0]
    print("Deleting some sequences...")
    index = gpsSpeedFeat[:, 0] > MAXSPEEDTHRESH
    gpsSpeedFeat  = gpsSpeedFeat [index]
    gpsSpeedSeq   = gpsSpeedSeq  [index]
    timestampSeqs = timestampSeqs[index]
    print("Delete %d samples | speed" % (n_samples - gpsSpeedFeat.shape[0]))
    n_detele += n_samples - gpsSpeedFeat.shape[0]

    ## 删除运行时长小于阈值的序列
    plt.subplot(3, 2, 2)
    plt.title("Moving Time"); plt.xlabel("s"); plt.ylabel("number")
    plt.hist(gpsSpeedFeat[:, 3], bins=100)

    n_samples = gpsSpeedFeat.shape[0]
    index = gpsSpeedFeat[:, 3] > MINRUNTIME
    gpsSpeedFeat  = gpsSpeedFeat [index]
    gpsSpeedSeq   = gpsSpeedSeq  [index]
    timestampSeqs = timestampSeqs[index]
    print("Delete %d samples | runing time" % (n_samples - gpsSpeedFeat.shape[0]))
    n_detele += n_samples - gpsSpeedFeat.shape[0]

    ## 删除总时长小于阈值的序列
    plt.subplot(3, 2, 3)
    plt.title("Total Time"); plt.xlabel("s"); plt.ylabel("number")
    plt.hist(gpsSpeedFeat[:, 4], bins=100)

    n_samples = gpsSpeedFeat.shape[0]
    index = gpsSpeedFeat[:, 4] > MINTIME
    gpsSpeedFeat  = gpsSpeedFeat [index]
    gpsSpeedSeq   = gpsSpeedSeq  [index]
    timestampSeqs = timestampSeqs[index]
    print("Delete %d samples | total time" % (n_samples - gpsSpeedFeat.shape[0]))
    n_detele += n_samples - gpsSpeedFeat.shape[0]

    ## 删除加速度大于阈值的序列
    plt.subplot(3, 2, 4)
    plt.title("Accelerate(+)"); plt.xlabel("m/s2"); plt.ylabel("number")
    plt.hist(gpsSpeedFeat[:, 7], bins=100)
    
    n_samples = gpsSpeedFeat.shape[0]
    index = gpsSpeedFeat[:, 7] < MAXPOSACC
    gpsSpeedFeat  = gpsSpeedFeat [index]
    gpsSpeedSeq   = gpsSpeedSeq  [index]
    timestampSeqs = timestampSeqs[index]
    print("Delete %d samples | accelerate(+)" % (n_samples - gpsSpeedFeat.shape[0]))
    n_detele += n_samples - gpsSpeedFeat.shape[0]

    ## 删除减速度大于阈值的序列
    plt.subplot(3, 2, 5)
    plt.title("Accelerate(-)"); plt.xlabel("m/s2"); plt.ylabel("number")
    plt.hist(gpsSpeedFeat[:, 8], bins=100)
    plt.savefig("images/1_gpsFeat_chosen_feature_file%d.png" % i)

    n_samples = gpsSpeedFeat.shape[0]
    index = gpsSpeedFeat[:, 8] > MAXNEGACC
    gpsSpeedFeat  = gpsSpeedFeat [index]
    gpsSpeedSeq   = gpsSpeedSeq  [index]
    timestampSeqs = timestampSeqs[index]
    print("Delete %d samples | accelerate(-)" % (n_samples - gpsSpeedFeat.shape[0]))
    n_detele += n_samples - gpsSpeedFeat.shape[0]
    
    # --------------------------------------
    ## 统计剩余的速度点数目
    n_remain = np.array(list(map(lambda x: x.shape[0], gpsSpeedSeq))).sum()

    # --------------------------------------
    ## 补齐
    gpsSpeedSeq, timestampSeqs, padValue = padSpeedSequences(gpsSpeedSeq, timestampSeqs)

    n_padded = padValue.shape[0]
    plt.figure(1)
    plt.title("Histogram of Padded Values")
    plt.subplot(3, 1, i)
    plt.xlabel("km/h")
    plt.ylabel("Number")
    plt.hist(padValue, bins=100, edgecolor='white')

    # --------------------------------------
    ## 保存
    print("Number of sequences: %d(%d deteled), Mean of sequences' length: %f" 
            % (len(gpsSpeedSeq), n_detele, gpsSpeed.shape[0] / len(gpsSpeedSeq)))
    print("Origin number of lines: %d, remaining number of lines: %d, padded number of lines: %d" 
            % (n_origin, n_remain, n_padded))
    print("-------------------------------------------------------------------------------------------")
    gpsSpeedSeqs += [gpsSpeedSeq]; gpsSpeedFeats += [gpsSpeedFeat]


plt.figure(0)
plt.savefig("images/1_gpsSpeed.png")
plt.figure(1)
plt.savefig("images/1_padValue.png")

gpsSpeedSeqs  = np.concatenate(gpsSpeedSeqs,  axis=0)
gpsSpeedFeats = np.concatenate(gpsSpeedFeats, axis=0)
np.save('output/1_gpsSpeedSequences.npy', gpsSpeedSeqs )
np.save('output/1_gpsSpeedFeatures.npy',  gpsSpeedFeats)
# -----------------------------------------
maxSpeed = gpsSpeedFeats[:, 0]
# print(maxSpeed[maxSpeed > 100].shape)

plt.figure()
plt.title("Maximum Speed(km/h)")
plt.xlabel("km/h"); plt.ylabel("Number")
plt.hist(maxSpeed, bins=30, facecolor='blue', edgecolor='white')
plt.savefig("images/1_maxSpeed_feat.png")

# -----------------------------------------
movingTime = gpsSpeedFeats[:, 3]
plt.figure()
plt.title("Moving Time(s)")
plt.xlabel("s"); plt.ylabel("Number")
plt.hist(movingTime, bins=30, facecolor='blue', edgecolor='white')
plt.savefig("images/1_movingTime_feat.png")

# plt.show()

## 小波变换去噪
# speedSeqDwt = dwtDecompose(speedSeq, dwtThresh)
# plt.figure(); plt.plot(speedSeq); plt.figure(); plt.plot(speedSeqDwt); plt.show()
# plot_signal_decomp(gpsSpeedSeqs, 'sym5', "DWT: GPS speed"); plt.show()
