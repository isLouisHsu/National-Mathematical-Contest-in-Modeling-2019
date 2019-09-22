# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-20 11:07:51
@LastEditTime: 2019-09-22 12:15:25
@Update: 
'''
import os
import numpy as np
from matplotlib import mlab
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.neighbors import KernelDensity

from params import n_components_default, n_clusters_default
from params import deleteClassIndex, featLandmarks, totalTime, sampleTime

sequences = np.load('output/1_gpsSpeedSequences.npy')
features  = np.load('output/1_gpsSpeedFeatures.npy')

# ------------------------------------------------------------------------------------
## 删除异常类别的样本
# n_components = input("Please enter the number of components(default %d): " % n_components_default)
# n_components = n_components_default if n_components == '' else int(n_components)
n_components = n_components_default
# n_clusters = input("Please enter the number of clusters  (default %d): " % n_clusters_default)
# n_clusters = n_clusters_default if n_clusters == '' else int(n_clusters)
n_clusters = n_clusters_default
pipeline = joblib.load("output/2_1_model_pca%d_kmeans%d.pkl" % (n_components, n_clusters))
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

    # ---------------------------------------------------------------------------------
    ## 统计该速度区间，峰值速度与平均加速度（包含正负）的联合概率分布
    featIdx0 = 0; featIdx1 = 7
    n0, bins0 = np.histogram(subFeature[:, featIdx0], bins=3)
    n1, bins1 = np.histogram(subFeature[:, featIdx1], bins=3)
    gap0 = bins0[1] - bins0[0]; gap1 = bins1[1] - bins1[0]

    pdf = np.zeros((n0.shape[0], n1.shape[0]))
    for j in range(n0.shape[0]):
        for k in range(n1.shape[0]):
            ### 峰值速度
            _idx1 = subFeature[:, featIdx0] > bins0[j]
            _idx2 = subFeature[:, featIdx0] < bins0[j + 1]
            idx1  = np.bitwise_and(_idx1, _idx2)
            ### 平均加速度
            _idx1 = subFeature[:, featIdx1] > bins1[k]
            _idx2 = subFeature[:, featIdx1] < bins1[k + 1]
            idx2  = np.bitwise_and(_idx1, _idx2)
            ### 统计数目
            idx   = np.bitwise_and(idx1, idx2)
            pdf[j, k] = idx.sum()
    pdf = pdf / pdf.sum()

    plt.figure()
    plt.imshow(pdf, cmap=plt.cm.hot)
    plt.title("PDF")
    plt.xlabel("max speed")
    plt.ylabel("max accelerate")
    plt.colorbar()
    # plt.xticks(bins0, rotation=45)
    # plt.yticks(bins1)
    plt.savefig("images/4_2_1_pdf_%d.png" % i)
    # plt.show()

    # ---------------------------------------------------------------------------------
    ## 直方图统计，运行时长的直方图
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
    print(i, Tk, tk, Nk)
    
    ## 删除运行时长大于Tk的序列
    index = subFeature[:, 3] <= Tk
    subSequence = subSequence[index]
    subFeature  = subFeature [index]
    p = p[index]

    # ---------------------------------------------------------------------------------
    ## 多次采样
    subChosenSequences = []; subChosenFeatures = []
    for j in range(sampleTime):
        index = np.random.choice(n.shape[0], size=Nk, replace=False, p=freq)
        subSequence_ = subSequence[index]
        subFeature_  = subFeature [index]
        subChosenSequences += [subSequence_]
        subChosenFeatures  += [subFeature_ ]
    subChosenSequences = np.array(subChosenSequences)
    subChosenFeatures  = np.array(subChosenFeatures )

    ## 滤除序列运行时间长度之和偏离Tk较多的序列
    lengths = np.array(list(map(lambda x: x[:, 3].sum(), subChosenFeatures)))
    t = 0.2
    index = np.bitwise_and(lengths > Tk*(1 - t), lengths < Tk*(1 + t))
    subChosenSequences = subChosenSequences[index]
    subChosenFeatures  = subChosenFeatures [index]

    # ---------------------------------------------------------------------------------
    ## 卡方检验
    K = []
    n_combine = subChosenSequences.shape[0]
    for j in range(n_combine):
    
        cnt = np.zeros_like(pdf, dtype=np.int)
        for  k in range(Nk):
            _f = subChosenFeatures[j, k]
            a, b = _f[featIdx0], _f[featIdx1]
            ia = int((a - bins0[0]) // gap0)
            ib = int((b - bins1[0]) // gap1)
            ia = ia - 1 if a == bins0[-1] else ia
            ib = ib - 1 if b == bins1[-1] else ib
            cnt[ia, ib] += 1
        
        k = ((cnt - Nk * pdf)**2 / ((Nk * pdf)**2 + np.finfo(np.float).eps))
        K += [k.sum()]
    K = np.array(K)

    index = np.argsort(K)
    K = K[index]
    subChosenSequences = subChosenSequences[index]
    subChosenFeatures = subChosenFeatures[index]
    
    ## 选择卡方值最小的
    n_seq = subChosenSequences.shape[0]
    size = n_seq if n_seq < 10 else 10
    chosenSequences += [subChosenSequences[0: size]]
    chosenFeatures  += [subChosenFeatures [0: size]]

# ------------------------------------------------------------------------
for k in range(size):
    ## 组合序列
    gap = 10
    combineSequences = []; combineFeatures = []
    for i in range(len(chosenFeatures)):
        chosenSequence = chosenSequences[i][k]
        chosenFeature  = chosenFeatures [i][k]
        for j in range(chosenFeature.shape[0]):
            startIdx = int(chosenFeature[j][4] - chosenFeature[j][3])
            combineSequence = chosenSequence[j][startIdx - 1:]
            combineFeature  = chosenFeature[j]
            combineSequences += [combineSequence]
            combineFeatures  += [combineFeature ]

    ## 根据峰值速度排序
    combineSequences = np.array(combineSequences)
    combineFeatures  = np.array(combineFeatures )
    index = np.argsort(combineFeatures[:, 0])
    combineSequences = combineSequences[index]
    combineFeatures  = combineFeatures [index]
    np.save("output/4_2_2_combineSequences_%d.npy" % k, combineSequences)
    np.save("output/4_2_2_combineFeatures_%d.npy"  % k, combineFeatures )

    ## 显示序列
    combineSequences = list(map(lambda x: np.r_[np.zeros(10), x], combineSequences))
    combineSequences = np.concatenate(combineSequences)
    fig = plt.figure(figsize=(12, 6))
    plt.title("Output Sequence")
    plt.xlabel("time(s)")
    plt.ylabel("speed(km/h)")
    plt.plot(combineSequences)
    plt.grid()
    plt.savefig("images/4_2_1_output_sequence_cluster%d_%d.png" % (n_clusters, k))

    # plt.show()