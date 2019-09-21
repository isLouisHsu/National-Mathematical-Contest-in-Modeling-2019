# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-21 13:55:31
@LastEditTime: 2019-09-21 13:57:58
@Update: 
'''
import os
import numpy as np
from matplotlib import mlab
from matplotlib import pyplot as plt
from sklearn.externals import joblib

from params import n_components_default, n_clusters_default
from params import deleteClassIndex, featLandmarks, totalTime

sequences = np.load('output/gpsSpeedSequences.npy')
features  = np.load('output/gpsSpeedFeatures.npy')

plt.figure()
plt.title("Sequence sample")
plt.xlabel("time(s)")
plt.ylabel("speed(km/h)")
plt.plot(sequences[np.random.randint(sequences.shape[0])])
plt.grid()
plt.savefig("images/0_sequence_sample.png")
plt.show()