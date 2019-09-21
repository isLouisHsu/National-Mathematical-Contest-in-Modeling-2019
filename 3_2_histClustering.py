# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-19 21:22:25
@LastEditTime: 2019-09-21 16:48:55
@Update: 
'''
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.externals import joblib

from params import deleteClassIndex
from params import n_components_default, n_clusters_default

n_components = input("Please enter the number of components(default %d): " % n_components_default)
n_components = n_components_default if n_components == '' else int(n_components)
n_clusters = input("Please enter the number of clusters  (default %d): " % n_clusters_default)
n_clusters   = n_clusters_default if n_clusters == '' else int(n_clusters)
pipeline = joblib.load("output/2_2_model_pca%d_gmm%d.pkl" % (n_components, n_clusters))

sequences = np.load('output/1_gpsSpeedSequences.npy')
features  = np.load('output/1_gpsSpeedFeatures.npy')
# ------------------------------------------------------------
## 查看各类的运动学片段样例
y = pipeline.predict(features)
n_classes, n_sequences = len(set(y)), 5
fig = plt.figure(figsize=(n_classes*3, n_sequences*2))
for i in range(n_classes):
    subseq = sequences[y == i]
    for j in range(n_sequences):
        ax = fig.add_subplot(n_sequences, n_classes, n_classes*j + i + 1)
        if j < subseq.shape[0]:
            ax.plot(subseq[j])
        if i == 0:
            ax.set_ylabel("km/h")
        if j == 0:
            ax.set_title("class %d" % (i))
        if j == n_sequences - 1:
            ax.set_xlabel("time(s)")
plt.savefig("images/3_2_sequences_gmm_cluster%d.png" % n_clusters)

# ------------------------------------------------------------
## 查看峰值速度大于100的运动学片段样例
t = 100
index = features[:, 0] > t
subseq = sequences[index]; subY = y[index]
print("Number of sequences(>%d): " % t, subseq.shape[0])
if subseq.shape[0] > 16:
   subseq = subseq[:16]; subY = subY[:16]
n_sequences = subseq.shape[0]
nh = int(np.ceil(np.sqrt(n_sequences))); nw = n_sequences // nh + 1
fig = plt.figure(figsize=(nw * 3, nh * 2))
for i in range(n_sequences):
    ax = fig.add_subplot(nw, nh, i // nh * nw + i % nh + 1)
    if i < subseq.shape[0]:
        ax.plot(subseq[i], label = 'class %d' % subY[i])
    ax.legend()
plt.savefig("images/3_2_maxSpeed_geq_%d_sequences_cluster%d.png" % (t, n_clusters))

# ------------------------------------------------------------
## 构造统计量，统计其长度直方图、与各类别的长度直方图
def getStatistic(features):
    return features[:, 0]

statistic = getStatistic(features)
plt.figure()
plt.title("Chosen Feature - all")
plt.xlabel("feature value")
plt.ylabel("Number")
plt.xlim(0, statistic.max())
# plt.ylim(0, 350)
n, bins, patches = plt.hist(statistic, bins=int(statistic.max() - statistic.min()) + 1, facecolor='blue', edgecolor='white')
plt.savefig("images/3_2_feat_hist_cluster%d.png" % n_clusters)

plt.figure(figsize=(5, 10))
plt.title("Chosen Feature - classes")
for i in range(n_classes):
    subStatistic = getStatistic(features[y == i])
    plt.subplot(n_classes, 1, i + 1)
    if i == n_classes - 1:
        plt.xlabel("feature value")
    plt.ylabel("Number - class %d" % i)
    plt.xlim(0, statistic.max())
    # plt.ylim(0, 250)
    n, bins, patches = plt.hist(subStatistic, bins=int(subStatistic.max() - subStatistic.min()) // 2 + 1, facecolor='blue', edgecolor='white')
plt.savefig("images/3_2_feat_hist_subseq_cluster%d.png" % n_clusters)

# ------------------------------------------------------------
## 删除部分类别的运动学片段，重新计算标签
index = np.ones(features.shape[0], dtype=np.bool)
for idx in deleteClassIndex:
    idx = y != idx
    index = np.bitwise_and(index, idx)
sequences = sequences[index]
features  = features [index]
pipeline.set_params(gmm__n_components=n_clusters - len(deleteClassIndex))
y = pipeline.fit_predict(features)
n_classes = len(set(y))

## 绘制高斯混合模型曲线
gmm = GaussianMixture(n_components=n_classes)
gmm.fit(statistic.reshape(-1, 1))
n, bins = np.histogram(statistic, bins=int(statistic.max() - statistic.min()) + 1)
bins = np.r_[0, bins]
plt.figure()
plt.title("Chosen Feature - GMM")
plt.xlabel("feature value")
# plt.ylim(0, 0.02)
plt.ylabel("p(x)")
for i in range(n_classes):
    mu, sigma2 = gmm.means_[i, 0], gmm.covariances_[i, 0]
    y_ = np.exp(- (bins - mu)**2 / (2 * sigma2)) / (2 * np.pi * sigma2)**0.5
    plt.plot(bins, y_ * gmm.weights_[i], label="class%d, mu=%.2f, sigma2=%.2f" % (i, mu, sigma2))
plt.grid(); plt.legend()
np.save("output/3_2_gmm_params_.npy", [gmm.means_.reshape(-1), gmm.covariances_.reshape(-1), gmm.weights_.reshape(-1)])
plt.savefig("images/3_2_feat_hist_GMM_cluster%d.png" % n_clusters)

plt.show()
