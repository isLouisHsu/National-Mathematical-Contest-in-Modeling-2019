# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-19 11:01:14
@LastEditTime: 2019-09-20 22:36:42
@Update: 
'''
import os
import pywt
import time, datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


from params import IDLETHRESH, MAXSPEEDTHRESH, MAXACCABS, MINTIME, MINRUNTIME
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

def cutSpeedSequences(timestamp, speedSeq, speedThresh=IDLETHRESH):
    """
    Params:
        timestamp: {ndarray(N)}
        speedSeq:  {ndarray(N)}
        speedThresh: {float} 怠速阈值
    Returns:
        speedSequences: {list[ndarray(n_length)]}
    """
    ## 类型1：时间不连续
    # isContiguous = np.r_[True, ((timestamp[1:] - timestamp[:-1]) == 1).astype(np.bool)]
    # notContiguousIndex = np.sort(np.where(isContiguous == False)[0])                # 不连续的开始时间点
    # for index in notContiguousIndex[::-1]:                                          # 按不连续两端的数据的均值进行补全
    #     numCycle  = int(timestamp[index] - timestamp[index - 1])
    #     padArray  = timestamp[index - 1] + np.arange(numCycle - 1, dtype=np.float) + 1
    #     timestamp = np.r_[timestamp[:index], padArray, timestamp[index:]]           # 补全时间戳
    #     # padArray  = np.full(numCycle, 0.5 * (speedSeq[index - 1] + speedSeq[index]))  # 填充两端均值
    #     padArray  = np.full(numCycle, 0)                                            # 填充`0`
    #     speedSeq  = np.r_[speedSeq[:index], padArray, speedSeq[index:]]             # 补全速度序列

    minTimestamp, maxTimestamp = timestamp.min(), timestamp.max()
    x = (timestamp - minTimestamp).astype(np.float); y = speedSeq.astype(np.float)
    f = interp1d(x, y, kind='linear')
    x = np.arange(maxTimestamp - minTimestamp + 1) 
    speedSeq = f(x); timestamp = x + minTimestamp

    ## 类型2：加减速异常数据 TODO:
    
    ## 类型3：长期停车；类型4：长时间堵车、断断续续低速行驶
    isIdle = (speedSeq < speedThresh).astype(np.int)
    index = np.r_[1, isIdle[1:] - isIdle[:-1]]                                      # 负跳变(-1)：开始加速
    cutIndex = np.where(index == 1)[0]
    cutIndex = np.r_[cutIndex[:-1], cutIndex[1:] + 1].reshape(2, -1).T
    
    ## 切出运动学片段
    # speedSeq[speedSeq < speedThresh] = 0.                                           # 小于阈值作怠速处理
    speedSequences = list(map(lambda x: speedSeq[x[0]: x[1]], cutIndex))

    return np.array(speedSequences)

def calFeaturesOfSequence(seq, speedThresh=IDLETHRESH, maxIdle=180, dwtTime=1):
    """
    Params:
        seq: {ndarray(n_length)}
    Notes:
    -   
    """
    isIdle = (seq < speedThresh).astype(np.int)     # 是否怠速
    index = np.r_[1, isIdle[1:] - isIdle[:-1]]      # 负跳变(-1)：从怠速起步
    idxStart = np.where(index == -1)[0][0]          # 起步时间

    if idxStart > maxIdle:                          # 怠速时间超过180s，剪裁速度值序列
        seq = seq[maxIdle: ]             
        idxStart -= maxIdle

    seq    = seq / 3.6                              # m/s
    n_sec  = seq.shape[0]
    n_dist = seq.sum()

    n_ = 2
    temp = np.r_[np.ones(n_)*seq[0], seq]
    accelerate = temp[n_:] - temp[:-n_]       # 加速度(m/s)
    accelerate = accelerate[idxStart:]

    feature = []
    
    feature += [seq.max()]                              # 0, 峰值速度(m/s)
    feature += [n_dist / (n_sec - idxStart)]            # 1, 平均速度(m/s)
    feature += [seq.std()]                              # 2, 速度标准差
    
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
        
    # feature += [n_dist]                                 # 13, 距离(m)
    feature += [idxStart / n_sec]                       # 14, 怠速时间比(%)
    feature += [n_dist / n_sec]                         # 15, 平均运行速度(m/s)
    
    feature = np.array(feature)
    return feature

if __name__ == "__main__":

    plt.figure()
    plt.title("Histogram of Speed")
    
    for i in range(1, 4):
        
        # filename = 'data/temp.xlsx'
        filename = 'data/file%d.xlsx' % i
        data = readData(filename)
        print(i, data.shape)
        
        timestamp, gpsSpeed = data[:, 0], data[:, 1]

        plt.subplot(3, 1, i)
        plt.xlabel("km/h")
        plt.ylabel("Number")
        plt.hist(gpsSpeed, edgecolor='white')

        # gpsSpeed = dwtDecompose(gpsSpeed, 2)

        # --------------------------------------
        ## 切分序列
        print("Cutting sequences...")
        if os.path.exists('output/gpsSpeedSequences_file%d.npy' % i):
            gpsSpeedSeq = np.load('output/gpsSpeedSequences_file%d.npy' % i)
        else:
            gpsSpeedSeq = cutSpeedSequences(timestamp, gpsSpeed)

        # --------------------------------------
        ## 计算特征
        print("Calculating features...")
        gpsSpeedFeat = np.stack(list(map(calFeaturesOfSequence, gpsSpeedSeq)), 0)
        
        # --------------------------------------
        # ## 删除峰值速度小于阈值的序列
        # n_samples = gpsSpeedFeat.shape[0]
        # print("Deleting some sequences...")
        # index = gpsSpeedFeat[:, 0] > MAXSPEEDTHRESH
        # gpsSpeedFeat = gpsSpeedFeat[index]
        # gpsSpeedSeq  = gpsSpeedSeq [index]
        # print("Delete %d samples" % (n_samples - gpsSpeedFeat.shape[0]))

        ## 删除加速度大于阈值的序列
        n_samples = gpsSpeedFeat.shape[0]
        index = np.bitwise_and(
            gpsSpeedFeat[:, 9] < MAXACCABS, 
            gpsSpeedFeat[:, 11] > - MAXACCABS)
        gpsSpeedFeat = gpsSpeedFeat[index]
        gpsSpeedSeq  = gpsSpeedSeq [index]
        print("Delete %d samples" % (n_samples - gpsSpeedFeat.shape[0]))

        ## 删除总时长小于阈值的序列
        n_samples = gpsSpeedFeat.shape[0]
        index = gpsSpeedFeat[:, 4] > MINTIME
        gpsSpeedFeat = gpsSpeedFeat[index]
        gpsSpeedSeq  = gpsSpeedSeq [index]
        print("Delete %d samples" % (n_samples - gpsSpeedFeat.shape[0]))

        ## 删除运行时长小于阈值的序列
        n_samples = gpsSpeedFeat.shape[0]
        index = gpsSpeedFeat[:, 3] > MINRUNTIME
        gpsSpeedFeat = gpsSpeedFeat[index]
        gpsSpeedSeq  = gpsSpeedSeq [index]
        print("Delete %d samples" % (n_samples - gpsSpeedFeat.shape[0]))

        # --------------------------------------
        ## 保存
        print("Number of sequences: %d, Mean of sequences' length: %f" 
                % (len(gpsSpeedSeq), gpsSpeed.shape[0] / len(gpsSpeedSeq)))

        np.save('output/gpsSpeedSequences_file%d.npy' % i, gpsSpeedSeq)
        np.save('output/gpsSpeedFeat_file%d.npy' % i, gpsSpeedFeat)

    plt.savefig("images/1_gpsSpeed.png")
    plt.show()
    ## 小波变换去噪
    # speedSeqDwt = dwtDecompose(speedSeq, dwtThresh)
    # plt.figure(); plt.plot(speedSeq); plt.figure(); plt.plot(speedSeqDwt); plt.show()
    # plot_signal_decomp(gpsSpeedSeq, 'sym5', "DWT: GPS speed"); plt.show()
    