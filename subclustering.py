# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-19 21:22:25
@LastEditTime: 2019-09-19 22:23:35
@Update: 
'''
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.neighbors import KernelDensity

n_components, n_clusters = 6, 5
pipeline = joblib.load("output/model_pca%d_kmeans%d.pkl" % (n_components, n_clusters))

sequences = []
features = []
for i in range(1, 4):
    sequences += [np.load('output/gpsSpeedSequences_file%d.npy' % i)]
    features  += [np.load('output/gpsSpeedFeat_file%d.npy' % i)]
sequences = np.concatenate(sequences, axis=0)
features  = np.concatenate(features,  axis=0)
# ------------------------------------------------------------

y = pipeline.predict(features)

n_classes, n_sequences = 5, 5
fig = plt.figure(figsize=(n_classes*2, n_sequences*2))
for i in range(n_classes):
    subseq = sequences[y == i]
    for j in range(n_sequences):
        ax = fig.add_subplot(n_sequences, n_classes, n_classes*j + i + 1)
        ax.plot(subseq[j * 2])
        if i == 0:
            ax.set_ylabel("km/h")
        if j == 0:
            ax.set_title("class %d" % (i))
plt.savefig("output/sequences_kmeans.png")
plt.show()

# ------------------------------------------------------------
maxSpeed = features[:, 5]
plt.figure()
plt.xlabel("Speed(km/h)")
plt.ylabel("Number")
plt.hist(maxSpeed, facecolor='lightblue', edgecolor='white')
plt.savefig("output/speed_hist.png")
plt.show()