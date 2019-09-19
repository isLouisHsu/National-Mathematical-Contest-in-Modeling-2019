# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-19 17:58:02
@LastEditTime: 2019-09-19 20:02:09
@Update: 
'''
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV

speedFeats = []
for i in range(1, 4):
    speedFeat = np.load('output/gpsSpeedFeat_file%d.npy' % i)
    speedFeats += [speedFeat]
speedFeats = np.concatenate(speedFeats, axis=0)
n_samples, n_features = speedFeats.shape

# ------------------------------------------------------------------
steps = [('scaler', StandardScaler()), ('pca', PCA(n_components=n_features))]
pipline = Pipeline(steps)
pipline.fit(speedFeats)

plt.figure(figsize=(8, 4))
plt.title('Explained Variance(Eigen Values) Ratio of PCA')
plt.subplot(121)
plt.plot(pipline.named_steps.pca.explained_variance_ratio_)
plt.xlabel("Number of Components")
plt.ylabel("Ratio of Explained Variance")
plt.grid()
plt.subplot(122)
plt.plot([pipline.named_steps.pca.explained_variance_ratio_[:i+1].sum() for i in range(n_features)])
plt.xlabel("Number of Components")
plt.ylabel("Sum of Ratio of Explained Variance")
plt.grid()
plt.savefig("images/pca.png")
plt.show()

# ------------------------------------------------------------------
# steps = [
#         ('scaler', StandardScaler()), 
#         ('pca', PCA()), 
#         ('kmeans', KMeans()),
#     ]
# pipline = Pipeline(steps)

# parameters = {
#     'pca__n_components': [_ for _ in range(6, n_features)], 
#     'kmeans__n_clusters': [_ for _ in range(3, 12)]
#     }
# searcher = GridSearchCV(pipline, parameters, cv=5)
# searcher.fit(speedFeats)

# print(searcher.cv_results_)
# print(searcher.best_estimator_)
# print(searcher.best_params_)

# ------------------------------------------------------------------
n_components = int(input("Please enter the number of components: "))
steps = [
        ('scaler', StandardScaler()), 
        ('pca', PCA(n_components=n_components)), 
        ('kmeans', KMeans()),
    ]
pipline = Pipeline(steps)

scores = []
for n_clusters in range(3, 15):
    pipline.set_params(kmeans__n_clusters=n_clusters)
    pipline.fit(speedFeats)
    scores += [pipline.score(speedFeats)]

plt.figure()
plt.title("K-means' Scores(n_components=%d)" % n_components)
plt.plot(scores)
plt.xlabel("Number of Clusters")
plt.ylabel("Score")
plt.grid()
plt.savefig("images/kmeans.png")
plt.show()