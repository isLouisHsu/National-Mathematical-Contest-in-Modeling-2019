# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-19 17:58:02
@LastEditTime: 2019-09-19 21:21:44
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
from sklearn.externals import joblib

speedFeats = []
for i in range(1, 4):
    speedFeat = np.load('output/gpsSpeedFeat_file%d.npy' % i)
    speedFeats += [speedFeat]
speedFeats = np.concatenate(speedFeats, axis=0)
n_samples, n_features = speedFeats.shape
print("Shape of samples: ", speedFeats.shape)

# ------------------------------------------------------------------
steps = [('scaler', StandardScaler()), ('pca', PCA(n_components=n_features))]
pipline = Pipeline(steps)
pipline.fit(speedFeats)

plt.figure(figsize=(8, 4))
plt.title('Explained Variance(Eigen Values) Ratio of PCA')
plt.subplot(121)
plt.bar(list(range(n_features)), pipline.named_steps.pca.explained_variance_ratio_)
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
n_clusters_list = list(range(3, 7))
for n_clusters in n_clusters_list:
    pipline.set_params(kmeans__n_clusters=n_clusters)
    pipline.fit(speedFeats)
    scores += [pipline.score(speedFeats)]

plt.figure()
plt.title("K-means' Scores(n_components=%d)" % n_components)
plt.plot(n_clusters_list, scores)
plt.xlabel("Number of Clusters")
plt.ylabel("Score")
plt.grid()
plt.savefig("images/kmeans.png")
plt.show()

# ------------------------------------------------------------------
n_clusters = int(input("Please enter the number of clusters: "))
steps = [
        ('scaler', StandardScaler()), 
        ('pca', PCA(n_components=n_components)), 
        ('kmeans', KMeans(n_clusters=n_clusters)),
    ]
pipline = Pipeline(steps)
pipline.fit(speedFeats)
y = pipline.predict(speedFeats)
np.save("output/seqLabels_pca%d_kmeans%d.npy" % (n_components, n_clusters), y)

joblib.dump(pipline, "output/model_pca%d_kmeans%d.pkl" % (n_components, n_clusters))
