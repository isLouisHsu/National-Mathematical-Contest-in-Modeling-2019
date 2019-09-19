# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-19 17:58:02
@LastEditTime: 2019-09-19 19:13:33
@Update: 
'''
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

speedFeats = []
for i in range(1, 4):
    speedFeat = np.load('output/gpsSpeedFeat_file%d.npy')
    speedFeats += [speedFeat]
speedFeats = np.concatenate(speedFeats, axis=0)

steps = [('scaler', StandardScaler()), ('pca', PCA(n_components=speedFeat.shape[1]))]
pipline = Pipeline(steps)
speedFeatsDecomposed = pipline.fit(speedFeats)

plt.figure()
plt.title('Explained Variance(Eigen Values) Ratio of PCA')
plt.plot(pipline['pca'].explained_variance_ratio_)
plt.xlabel("Number of Components")
plt.ylabel("Ratio of Explained Variance")
plt.show()