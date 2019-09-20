# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-20 11:07:51
@LastEditTime: 2019-09-20 21:28:57
@Update: 
'''
import os
import numpy as np
from matplotlib import mlab
from matplotlib import pyplot as plt
from sklearn.externals import joblib

from params import deleteClassIndex, featLandmarks

sequences = []
features = []
for i in range(1, 4):
    sequences += [np.load('output/gpsSpeedSequences_file%d.npy' % i)]
    features  += [np.load('output/gpsSpeedFeat_file%d.npy' % i)]
sequences = np.concatenate(sequences, axis=0)
features  = np.concatenate(features,  axis=0)

# ------------------------------------------------------------------------------------
## 删除异常类别的样本
n_components = input("Please enter the number of components(default 6): ")
n_components = 6 if n_components == '' else int(n_components)
n_clusters   = input("Please enter the number of clusters  (default 8): ")
n_clusters   = 8 if n_clusters == '' else int(n_clusters)
pipeline = joblib.load("output/model_pca%d_kmeans%d.pkl" % (n_components, n_clusters))
y = pipeline.predict(features)
index = np.ones(features.shape[0], dtype=np.bool)
for idx in deleteClassIndex:
    idx = y != idx
    index = np.bitwise_and(index, idx)
sequences = sequences[index]
features  = features [index]

# ------------------------------------------------------------------------------------
index = 0
feat      = features[:, index]                              # 最大速度(m/s)
totalTime = 1200.
sumRunTime = np.sum(features[:, 3])                         # 运行时间(s)

# ------------------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 12)); plt.title("Runing Time(s)")

T = []; t = []; N = []
for i in range(len(featLandmarks) + 1):

    ## 速度区间
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

    n, bins = np.histogram(subFeature[:, 3], bins=100)
    bins = (bins[1:] + bins[:-1]) / 2
    freq = n / n.sum()
    cumFreq = [np.sum(freq[:i+1]) for i in range(freq.shape[0])]
    
    Tk = subFeature[:, 3].sum() / sumRunTime * totalTime    # 当前速度区间，运动片段运行时长之和 / 所有运动片段运行时长之和
    tk = subFeature[:, 3].mean()                            # 当前速度区间，平均运行时长
    Nk = int(np.round(Tk / tk))                             # 当前速度区间，划分的累积频率区间数目
    T += [Tk]; t += [tk]; N += [Nk]
    
    ax = fig.add_subplot(len(featLandmarks) + 1, 2, i * 2 + 1)
    ax.set_xlabel("time(s)")
    ax.set_xticks(bins[::10])
    ax.set_ylabel("frequency(%)")
    ax.bar(bins, freq)
    # ax.bar(bins, n)
    # ax.set_xlim(0, 20)
    ax = fig.add_subplot(len(featLandmarks) + 1, 2, i * 2 + 2)
    ax.set_xlabel("time(s)")
    ax.set_xticks(bins[::10])
    ax.set_ylabel("cumulative frequency(%)")
    ax.plot(np.r_[0., bins], np.r_[0, cumFreq])
    ax.grid()
    
    if Nk == 0: continue
    g = 1. / Nk; r = np.array([i*g for i in range(Nk)]); r = np.r_[r, 1.]; r = (r[1:] + r[:-1]) / 2
    for i in range(r.shape[0]): ax.hlines(r[i], 0, bins.max(), 'r', 'dashed')
    ax.vlines(tk, 0, 1, 'r', 'dashed')
    
print(T, t, N)

plt.savefig("images/4_running_time.png")
plt.show()