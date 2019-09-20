# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-20 11:07:51
@LastEditTime: 2019-09-20 15:56:00
@Update: 
'''
import os
import numpy as np
from matplotlib import pyplot as plt

sequences = []
features = []
for i in range(1, 4):
    sequences += [np.load('output/gpsSpeedSequences_file%d.npy' % i)]
    features  += [np.load('output/gpsSpeedFeat_file%d.npy' % i)]
sequences = np.concatenate(sequences, axis=0)
features  = np.concatenate(features,  axis=0)

# ------------------------------------------------------------------------------------
index = 0
feat      = features[:, index]                              # 最大速度(m/s)
index     = np.argsort(feat)
sequences = sequences[index]
features  = features [index]

# ------------------------------------------------------------------------------------
totalTime = 1200.
featLandmarks = [5, 10, 20, ]
sumRunTime = np.sum(features[:, 3])                         # 运行时间(s)

## 删除运行时间大于`totalTime`的序列
index = features[:, 3] <= totalTime
sequences = sequences[index]
features  = features [index]

# ------------------------------------------------------------------------------------
fig = plt.figure(figsize=(12, 12)); plt.title("Runing Time(s)")

T = []; t = []; N = []
for i in range(len(featLandmarks) + 1):

    ## 当前速度区间
    if i == 0:
        index = feat < featLandmarks[i]
    elif i == len(featLandmarks):
        index = feat > featLandmarks[i - 1]
    elif i > 0 and i < len(feat) - 1:
        index1 = feat > featLandmarks[i - 1]
        index2 = feat < featLandmarks[i]
        index  = np.bitwise_and(index1, index2)
    subSequence = sequences[index]
    subFeature  = features [index]
    
    Tk = subFeature[:, 3].sum() / sumRunTime * totalTime    # 当前速度区间，运动片段运行时长之和 / 所有运动片段运行时长之和
    tk = subFeature[:, 3].mean()                            # 当前速度区间，平均运行时长
    Nk = int(np.round(Tk / tk))                             # 当前速度区间，划分的累积频率区间数目
    T += [Tk]; t += [tk]; N += [Nk]

    n, bins = np.histogram(subFeature[:, 3], bins=70)
    # print(bins)
    freq = n / n.sum()
    cumFreq = [0] + [np.sum(freq[:i+1]) for i in range(freq.shape[0])]
    
    ax = fig.add_subplot(len(featLandmarks) + 1, 2, i * 2 + 1)
    ax.set_xlabel("index")
    ax.set_ylabel("frequency(%)")
    ax.bar(list(range(freq.shape[0])), freq)
    ax = fig.add_subplot(len(featLandmarks) + 1, 2, i * 2 + 2)
    ax.set_xlabel("time(s)")
    ax.set_ylabel("cumulative frequency(%)")
    for i in range(Nk): ax.hlines(i / Nk, 0, subFeature[:, 3].max(), 'r', 'dashed')
    # ax.vlines(tk, 0, 1, 'r', 'dotted')
    ax.plot(bins, cumFreq)
    ax.grid()

print(T, t, N)

plt.savefig("images/4_running_time.png")
plt.show()