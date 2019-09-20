# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-19 21:22:25
@LastEditTime: 2019-09-20 18:06:48
@Update: 
'''
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture

n_components = input("Please enter the number of components(default 6): ")
n_components = 6 if n_components == '' else int(n_components)
n_clusters   = input("Please enter the number of clusters  (default 5): ")
n_clusters   = 5 if n_clusters == '' else int(n_clusters)
pipeline = joblib.load("output/model_pca%d_kmeans%d.pkl" % (n_components, n_clusters))

sequences = []
features = []
for i in range(1, 4):
    sequences += [np.load('output/gpsSpeedSequences_file%d.npy' % i)]
    features  += [np.load('output/gpsSpeedFeat_file%d.npy' % i)]
sequences = np.concatenate(sequences, axis=0); features  = np.concatenate(features,  axis=0)

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
plt.savefig("images/3_sequences_kmeans.png")

# ------------------------------------------------------------
## 统计总的序列长度直方图、各类别序列长度直方图
index = 0
chosenFeat = features[:, index]
plt.figure()
plt.title("Chosen Feature - all")
plt.xlabel("speed(km/h)")
plt.ylabel("Number")
plt.xlim(0, chosenFeat.max())
# plt.ylim(0, 350)
n, bins, patches = plt.hist(chosenFeat, bins=int(chosenFeat.max() - chosenFeat.min()) + 1, facecolor='blue', edgecolor='white')
plt.savefig("images/3_feat_hist.png")

plt.figure(figsize=(5, 10))
plt.title("Chosen Feature - classes")
for i in range(n_classes):
    subChosenFeat = features[y == i][:, index]
    plt.subplot(n_classes, 1, i + 1)
    if i == n_classes - 1:
        plt.xlabel("speed(km/h)")
    plt.ylabel("Number - class %d" % i)
    plt.xlim(0, chosenFeat.max())
    plt.ylim(0, 400)
    n, bins, patches = plt.hist(subChosenFeat, bins=int(subChosenFeat.max() - subChosenFeat.min()) + 1, facecolor='blue', edgecolor='white')
plt.savefig("images/3_feat_hist_subseq.png")

# ------------------------------------------------------------
## 删除两种多余的运动学片段，重新计算标签
deleteClassIndex = [1, 2]   # TODO:
index = np.ones(features.shape[0], dtype=np.bool)
for idx in deleteClassIndex:
    idx = y != idx
    index = np.bitwise_and(index, idx)
sequences = sequences[index]
features  = features [index]
pipeline.set_params(kmeans__n_clusters=n_clusters - len(deleteClassIndex))
y = pipeline.fit_predict(features)
n_classes = len(set(y))

# ------------------------------------------------------------
## 绘制高斯混合模型曲线
gmm = GaussianMixture(n_components=n_classes)
gmm.fit(chosenFeat.reshape(-1, 1))
n, bins = np.histogram(chosenFeat, bins=int(chosenFeat.max() - chosenFeat.min()) + 1)
plt.figure()
plt.title("Chosen Feature - GMM")
plt.xlabel("speed(km/h)")
# plt.ylim(0, 0.02)
plt.ylabel("P")
for i in range(n_classes):
    mu, sigma = gmm.means_[i, 0], gmm.covariances_[i, 0]
    y_ = np.exp(-0.5*np.square((bins-mu)/sigma))/(np.sqrt(2*np.pi)*sigma)
    plt.plot(bins, y_ * gmm.weights_[i], label="mu=%.2f sigma=%.2f" % (mu, sigma))
    plt.grid(); plt.legend()
plt.savefig("images/3_feat_hist_GMM.png")
plt.show()
