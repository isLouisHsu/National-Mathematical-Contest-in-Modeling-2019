# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-20 11:07:51
@LastEditTime: 2019-09-21 16:04:50
@Update: 
'''
import os
import numpy as np
from matplotlib import mlab
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.neighbors import KernelDensity

from params import n_components_default, n_clusters_default
from params import deleteClassIndex, featLandmarks, totalTime

sequences = np.load('output/1_gpsSpeedSequences.npy')
features  = np.load('output/1_gpsSpeedFeatures.npy')

# ------------------------------------------------------------------------------------
## 删除异常类别的样本
n_components = input("Please enter the number of components(default %d): " % n_components_default)
n_components = n_components_default if n_components == '' else int(n_components)
n_clusters   = input("Please enter the number of clusters  (default %d): " % n_clusters_default)
n_clusters   = n_clusters_default if n_clusters == '' else int(n_clusters)
pipeline = joblib.load("output/2_2_model_pca%d_gmm%d.pkl" % (n_components, n_clusters))
y = pipeline.predict(features)
index = np.ones(features.shape[0], dtype=np.bool)
for idx in deleteClassIndex:
    idx = y != idx
    index = np.bitwise_and(index, idx)
sequences = sequences[index]
features  = features [index]

# ------------------------------------------------------------------------------------
feat      = features[:, 0]                                  # 最大速度(m/s)
sumRunTime = np.sum(features[:, 3])                         # 运行时间(s)

# ------------------------------------------------------------------------------------
fig = plt.figure(figsize=(6, 12)); plt.title("Runing Time(s)")

chosenSequences = []; chosenFeatures = []

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

    ## 直方图统计，各速度的直方图
    n, bins = np.histogram(subFeature[:, 3], bins=70)
    freq = n / n.sum()                                                      # 速度频率分布
    cumFreq = np.array([np.sum(freq[:i+1]) for i in range(freq.shape[0])])  # 累积频率
    
    ## 根据频率，指定每个序列的采样概率
    p = np.zeros_like(subFeature[:, 3])
    for j in range(n.shape[0]):
        index = np.bitwise_and(subFeature[:, 3] > bins[j], subFeature[:, 3] < bins[j + 1])
        p[index] = freq[j]
    
    ## 采样个数
    Tk = subFeature[:, 3].sum() / sumRunTime * totalTime    # 当前速度区间，运动片段运行时长之和 / 所有运动片段运行时长之和
    tk = subFeature[:, 3].mean()                            # 当前速度区间，平均运行时长
    Nk = int(np.round(Tk / tk))                             # 当前速度区间，划分的累积频率区间数目

    ## 删除运行时长大于Tk的序列
    index = subFeature[:, 3] <= Tk
    subSequence = subSequence[index]
    subFeature  = subFeature [index]
    p = p[index]

    ## 依概率采样
    index = np.random.choice(n.shape[0], size=Nk, replace=False, p=freq)
    chosenSequence = subSequence[index]
    chosenFeature  = subFeature [index]

    ## 按峰值速度排序
    index = np.argsort(chosenFeature[:, 0])
    chosenSequence = chosenSequence[index]
    chosenFeature  = chosenFeature [index]

    ## 保存序列
    chosenSequences += [chosenSequence]
    chosenFeatures  += [chosenFeature ]
    
    ## 绘制直方图
    bins = (bins[1:] + bins[:-1]) / 2
    xlim_ = bins[np.array(cumFreq) < 0.75][-1]
    ax = fig.add_subplot(len(featLandmarks) + 1, 1, i + 1)
    ax.set_xlabel("time(s)")
    ax.set_ylabel("frequency(%)")
    ax.set_xticks(bins[::10])
    ax.set_xlim(0, xlim_)
    ax.bar(np.arange(freq.shape[0]), freq)

plt.savefig("images/4_2_2_running_time_cluster%d.png" % n_clusters)

# ------------------------------------------------------------------------
## 组合序列
combineSequences = []
for i in range(len(chosenFeatures)):
    chosenSequence = chosenSequences[i]
    chosenFeature  = chosenFeatures[i]
    for j in range(chosenFeature.shape[0]):
        startIdx = int(chosenFeature[j][4] - chosenFeature[j][3])
        combineSequence  = chosenSequence[j][startIdx - 1:]
        combineSequences += [np.r_[np.zeros(10), combineSequence]]
combineSequences = np.concatenate(combineSequences)

fig = plt.figure(figsize=(12, 6))
plt.title("Output Sequence")
plt.xlabel("time(s)")
plt.ylabel("speed(km/h)")
plt.plot(combineSequences)
plt.grid()
plt.savefig("images/4_2_2_output_sequence_cluster%d.png" % n_clusters)

plt.show()