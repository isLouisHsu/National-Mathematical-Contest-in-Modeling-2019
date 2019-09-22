# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-22 14:29:48
@LastEditTime: 2019-09-22 17:42:42
@Update: 
'''
import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import signal
from sklearn.externals import joblib

from params import n_components_default, n_clusters_default, deleteClassIndex, k

def calFeature(seq):
    """
    Params:
        seq: {ndarray(n_sequence)}
    """
    seq = signal.medfilt(seq, (3,))
    _speedSeq = seq / 3.6                              # m/s
    accelerate = np.r_[0, _speedSeq[1:] - _speedSeq[:-1]]   # m/s2

    feature = []

    # 峰值速度
    feature += [seq.max()]

    # 平均速度
    feature += [seq.mean()]

    # 速度标准差
    feature += [seq.std()]

    # 加速时间百分比
    _acc = accelerate[accelerate > 0.]
    feature += [_acc.shape[0] / accelerate.shape[0]]

    # 峰值加速度
    feature += [_acc.max()]

    # 平均加速度(+)
    feature += [_acc.mean()]

    # 加速度标准差
    feature += [_acc.std()]

    # 减速时间百分比
    _acc = accelerate[accelerate < 0.]
    feature += [_acc.shape[0] / accelerate.shape[0]]

    # 峰值减速度
    feature += [_acc.min()]

    # 平均减速度(-)
    feature += [_acc.mean()]

    # 减速度标准差
    feature += [_acc.std()]

    return feature

def evalSpeedAndAccelerate(seq):
    """
    Params:
        seq: {ndarray(n_sequence)}
    """
    seq = signal.medfilt(seq, (3,))
    _speedSeq = seq / 3.6                                   # m/s
    acc = np.r_[0, _speedSeq[1:] - _speedSeq[:-1]]          # m/s2

    speedLandmark      = [-np.inf] + [10 * (i + 1) for i in range(6)] + [np.inf]
    accelerateLandmark = [-np.inf] + [i - 4 for i in range(8)]        + [np.inf]

    rows, cols = len(speedLandmark) - 1, len(accelerateLandmark) - 1
    pdf = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            s, e = speedLandmark[i: i + 2]
            index1 = np.bitwise_and(seq > s, seq < e)
            s, e = accelerateLandmark[j: j + 2]
            index2 = np.bitwise_and(acc > s, acc < e)

            index = np.bitwise_and(index1, index2)
            pdf[i, j] = np.where(index)[0].shape[0]
    pdf = pdf / pdf.sum()

    return pdf

# ------------------------------------------------------------------------
speedSeq = np.load("output/4_2_2_combineSequences_%d.npy" % k)
speedSeq = np.concatenate(speedSeq)
featOutput = calFeature(speedSeq)

# ------------------------------------------------------------------------
sequences = np.load('output/1_gpsSpeedSequences.npy')
features  = np.load('output/1_gpsSpeedFeatures.npy')

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
featOrigin = calFeature(np.concatenate(sequences))

np.savetxt("output/5_1_featOutput_%d.txt" % k, featOutput)
np.savetxt("output/5_1_featOrigin_%d.txt" % k, featOrigin)

# ========================================================================
pdfOrigin = evalSpeedAndAccelerate(np.concatenate(sequences))
pdfOutput = evalSpeedAndAccelerate(speedSeq)
pdfError  = np.abs(pdfOrigin - pdfOutput)
# print(pdfOrigin); print(pdfOutput); print(pdfError)

np.savetxt("output/5_1_pdfError_max_%d.txt" % k, [pdfError.max()])

X = [i - 4 for i in range(9)]
Y = [10 * (i + 1) for i in range(7)]
X, Y = np.meshgrid(X, Y)

fig = plt.figure()
# plt.title("所有序列速度(km/h)与加速度(m/s2)联合分布")
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, pdfOrigin, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
ax.set_zlim(0., 0.2)
ax.set_xlabel('accelerate(m/s2)')
ax.set_ylabel('speed(km/h)')
ax.set_zlabel('joint probability distribution')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("images/5_1_orgin_pdf_3d_%d.png" % k)

fig = plt.figure()
# plt.title("合成序列速度(km/h)与加速度(m/s2)联合分布")
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, pdfOutput, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
ax.set_zlim(0., 0.2)
ax.set_xlabel('accelerate(m/s2)')
ax.set_ylabel('speed(km/h)')
ax.set_zlabel('joint probability distribution')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("images/5_1_output_pdf_3d_%d.png" % k)

fig = plt.figure()
# plt.title("联合分布差异")
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, pdfError, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
ax.set_zlim(0., 0.2)
ax.set_xlabel('accelerate(m/s2)')
ax.set_ylabel('speed(km/h)')
ax.set_zlabel('joint probability distribution')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("images/5_1_error_pdf_3d_%d.png" % k)

# plt.show()