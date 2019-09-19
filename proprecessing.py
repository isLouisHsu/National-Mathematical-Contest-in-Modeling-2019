# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-19 11:01:14
@LastEditTime: 2019-09-19 17:55:58
@Update: 
'''
import os
import time, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt

from dwt_signal_decomposition import plot_signal_decomp, dwtDecompose

SPEEDTHRESH = 2.

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

    return data

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

def cutSpeedSequences(timestamp, speedSeq, speedThresh=SPEEDTHRESH):
    """
    Params:
        timestamp: {ndarray(N)}
        speedSeq:  {ndarray(N)}
        speedThresh: {float} 怠速阈值
    Returns:
        speedSequences: {list[ndarray(n_length)]}
    """
    ## 类型1：时间不连续
    isContiguous = np.r_[True, ((timestamp[1:] - timestamp[:-1]) == 1).astype(np.bool)]
    notContiguousIndex = np.sort(np.where(isContiguous == False)[0])                # 不连续的开始时间点
    for index in notContiguousIndex[::-1]:                                          # 按不连续两端的数据的均值进行补全
        numCycle  = int(timestamp[index] - timestamp[index - 1])
        padArray  = timestamp[index - 1] + np.arange(numCycle - 1, dtype=np.float) + 1
        timestamp = np.r_[timestamp[:index], padArray, timestamp[index:]]           # 补全时间戳
        padArray  = np.full(numCycle, 0.5 * (speedSeq[index - 1] + speedSeq[index]))
        speedSeq  = np.r_[speedSeq[:index], padArray, speedSeq[index:]]             # 补全速度序列

    ## 类型2：加减速异常数据 TODO:
    
    ## 类型3：长期停车；类型4：长时间堵车、断断续续低速行驶
    isIdle = (speedSeq < speedThresh).astype(np.int)
    index = np.r_[1, isIdle[1:] - isIdle[:-1]]                                      # 负跳变(-1)：开始加速
    cutIndex = np.where(index == 1)[0]
    cutIndex = np.r_[cutIndex[:-1], cutIndex[1:] + 1].reshape(2, -1).T
    
    ## 切出运动学片段
    # speedSeq[speedSeq < speedThresh] = 0.                                           # 小于阈值作怠速处理
    speedSequences = list(map(lambda x: speedSeq[x[0]: x[1]], cutIndex))

    return speedSequences, cutIndex

def calFeaturesOfSequence(seq, speedThresh=SPEEDTHRESH, maxIdle=180):
    """
    Params:
        seq: {ndarray(n_length)}
    Notes:
    -   
    """
    # plt.figure(); plt.plot(seq); plt.show()
    
    isIdle = (seq < speedThresh).astype(np.int) # 是否怠速
    index = np.r_[1, isIdle[1:] - isIdle[:-1]]  # 负跳变(-1)：从怠速起步
    idxStart = np.where(index == -1)[0][0]      # 起步时间
    accelerate = np.r_[0, seq[1:] - seq[:-1]]   # 加速度(m/s)
    accelerate = accelerate[idxStart:]

    seq_   = seq / 3.6                          # m/s
    n_sec  = seq.shape[0]
    n_dist = seq.sum()

    feature = []
    feature += [n_sec]                          # 时长(s)
    feature += [n_dist]                         # 距离(m)
    
    feature += [idxStart / n_sec]               # 怠速时间比(%)

    feature += [np.where(accelerate > 0.)[0].shape[0]]  # 加速时间(s)
    feature += [np.where(accelerate < 0.)[0].shape[0]]  # 减速时间(s)
    
    feature += [seq.max()]                      # 峰值速度(m/s)
    feature += [n_dist / n_sec]                 # 平均运行速度(m/s)
    feature += [n_dist / (n_sec - idxStart)]    # 平均速度(m/s)
    feature += [seq.std()]                      # 速度标准差

    feature += [accelerate.max()]               # 峰值加速度
    feature += [accelerate.min()]               # 峰值减速度

    temp = accelerate[accelerate > 0.]
    feature += [temp.mean()]                    # 平均加速度
    feature += [temp.std()]                     # 加速度标准差
    temp = accelerate[accelerate < 0.]
    feature += [temp.mean()]                    # 平均减速度
    feature += [temp.std()]                     # 减速度标准差
    
    feature = np.array(feature)
    return feature

if __name__ == "__main__":

    # filename = 'data/temp.xlsx'

    for i in range(1, 4):
        filename = 'data/file%d.xlsx' % i
        data = readData(filename)
        timestamp, gpsSpeed = data[:, 0], data[:, 1]
        gpsSpeedSeq, cutIndex = cutSpeedSequences(timestamp, gpsSpeed)
        np.save('output/gpsSpeedCutIndex_file%d.npy' % i, cutIndex)
        gpsSpeedFeat = np.stack(list(map(calFeaturesOfSequence, gpsSpeedSeq)), 0)
        np.save('output/gpsSpeedFeat_file%d.npy' % i, gpsSpeedFeat)
    
    ## 小波变换去噪
    # speedSeqDwt = dwtDecompose(speedSeq, dwtThresh)
    # plt.figure(); plt.plot(speedSeq); plt.figure(); plt.plot(speedSeqDwt); plt.show()
    # plot_signal_decomp(gpsSpeedSeq, 'sym5', "DWT: GPS speed"); plt.show()
    