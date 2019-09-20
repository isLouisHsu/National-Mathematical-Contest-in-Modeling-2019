# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-19 21:22:25
@LastEditTime: 2019-09-20 11:06:03
@Update: 
'''
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.mixture import GaussianMixture

n_components = input("Please enter the number of components(default 6): ")
n_components = 6 if n_components == '' else int(n_components)
n_clusters = input("Please enter the number of clusters(default 5): ")
n_clusters = 5 if n_clusters == '' else int(n_clusters)
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
n_classes, n_sequences = len(set(y)), 5
fig = plt.figure(figsize=(n_classes*2, n_sequences*2))
for i in range(n_classes):
    subseq = sequences[y == i]
    if subseq.shape[0] < n_sequences:
        continue
    for j in range(n_sequences):
        ax = fig.add_subplot(n_sequences, n_classes, n_classes*j + i + 1)
        ax.plot(subseq[j * 2])
        if i == 0:
            ax.set_ylabel("km/h")
        if j == 0:
            ax.set_title("class %d" % (i))
plt.savefig("images/sequences_kmeans.png")

# ------------------------------------------------------------
index = 0
maxSpeed = features[:, index]
plt.figure()
plt.title("Maximum speed(km/s) - all")
plt.xlabel("km/h")
plt.ylabel("Number")
plt.xlim(0, maxSpeed.max())
plt.ylim(0, 400)
n, bins, patches = plt.hist(maxSpeed, bins=int(maxSpeed.max() - maxSpeed.min()) + 1, edgecolor='white')
plt.savefig("images/speed_hist.png")

plt.figure(figsize=(5, 10))
plt.title("Maximum speed(km/s) - classes")
for i in range(n_classes):
    subMaxSpeed = features[y == i][:, index]
    plt.subplot(n_classes, 1, i + 1)
    plt.xlabel("km/h")
    plt.ylabel("Number")
    plt.xlim(0, maxSpeed.max())
    plt.ylim(0, 400)
    n, bins, patches = plt.hist(subMaxSpeed, bins=int(subMaxSpeed.max() - subMaxSpeed.min()) + 1, edgecolor='white')
plt.savefig("images/speed_hist_subseq.png")

# ------------------------------------------------------------
gmm = GaussianMixture(n_components=n_components)
gmm.fit(maxSpeed.reshape(-1, 1))
n, bins, patches = plt.hist(maxSpeed, bins=int(maxSpeed.max() - maxSpeed.min()) + 1, edgecolor='white')
plt.figure()
plt.title("Maximum speed(km/s) - GMM")
plt.xlabel("km/h")
plt.ylabel("P")
for i in range(n_components):
    mu, sigma = gmm.means_[i, 0], gmm.covariances_[i, 0]
    y_ = np.exp(-0.5*np.square((bins-mu)/sigma))/(np.sqrt(2*np.pi)*sigma)
    plt.plot(bins, y_)
    plt.grid()
plt.savefig("images/speed_hist_GMM.png")
plt.show()
