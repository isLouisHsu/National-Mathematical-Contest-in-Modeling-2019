# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-19 17:58:02
@LastEditTime: 2019-09-21 14:32:55
@Update: 
'''
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from params import n_components_default, n_clusters_default

SCALAR = StandardScaler

features = np.load('output/gpsSpeedFeatures.npy')
n_samples, n_features = features.shape
print("Shape of samples: ", features.shape)

# ------------------------------------------------------------------
steps = [('scaler', SCALAR()), ('pca', PCA(n_components=n_features))]
pipline = Pipeline(steps)
pipline.fit(features)

plt.figure(figsize=(8, 4))
plt.title('Explained Variance(Eigen Values) Ratio of PCA')
plt.subplot(121)
plt.bar(list(range(n_features)), pipline.named_steps.pca.explained_variance_ratio_, edgecolor='white')
plt.xlabel("Number of Components")
plt.ylabel("Ratio of Explained Variance")
plt.grid()
plt.subplot(122)
plt.plot([pipline.named_steps.pca.explained_variance_ratio_[:i+1].sum() for i in range(n_features)])
plt.xlabel("Number of Components")
plt.ylabel("Sum of Ratio of Explained Variance")
plt.grid()
plt.savefig("images/2_pca.png")
plt.show()

# ------------------------------------------------------------------
n_components = input("Please enter the number of components(default %d): " % n_components_default)
n_components = n_components_default if n_components == '' else int(n_components)
steps = [
        ('scaler', SCALAR()), 
        ('pca', PCA(n_components=n_components)), 
        ('kmeans', KMeans()),
    ]
pipline = Pipeline(steps)

scores = []
n_clusters_list = list(range(3, 15))
for n_clusters in n_clusters_list:
    pipline.set_params(kmeans__n_clusters=n_clusters)
    pipline.fit(features)
    scores += [pipline.score(features)]

plt.figure()
plt.title("K-means' Scores(n_components=%d)" % n_components)
plt.plot(n_clusters_list, scores)
plt.xlabel("Number of Clusters")
plt.ylabel("Score")
plt.grid()
plt.savefig("images/2_kmeans.png")
plt.show()

# ------------------------------------------------------------------
n_clusters = input("Please enter the number of clusters  (default %d): " % n_clusters_default)
n_clusters = n_clusters_default if n_clusters == '' else int(n_clusters)
steps = [
        ('scaler', SCALAR()), 
        ('pca', PCA(n_components=n_components)), 
        ('kmeans', KMeans(n_clusters=n_clusters)),
    ]
pipline = Pipeline(steps)
pipline.fit(features)
y = pipline.predict(features)
np.savetxt("output/2_seqLabels_pca%d_kmeans%d.txt" % (n_components, n_clusters), y.astype(np.int))
joblib.dump(pipline, "output/model_pca%d_kmeans%d.pkl" % (n_components, n_clusters))

# -------------------------------------------------------------------
cluster_centers_ = pipline.named_steps.kmeans.cluster_centers_
components_      = pipline.named_steps.pca.components_
mean_            = pipline.named_steps.pca.mean_
clusterCenters  = cluster_centers_.dot(components_) + mean_
clusterCenters  = pipline.named_steps.scaler.inverse_transform(clusterCenters)
np.savetxt("output/2_pca_explained_variance_.txt", pipline.named_steps.pca.explained_variance_)
np.savetxt("output/2_pca__components_.txt", components_)
np.savetxt("output/2_pca__mean_.txt", mean_)
np.savetxt("output/2_kmeans__cluster_centers_.txt", cluster_centers_)
np.savetxt("output/2_kmeans__cluster_centers_inv.txt", clusterCenters)
print(clusterCenters)