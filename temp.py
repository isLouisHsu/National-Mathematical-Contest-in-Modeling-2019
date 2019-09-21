# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-21 13:55:31
@LastEditTime: 2019-09-21 17:01:25
@Update: 
'''
import os
import numpy as np
import pandas as pd
from matplotlib import mlab
from matplotlib import pyplot as plt
from sklearn.externals import joblib

from params import n_components_default, n_clusters_default
from params import deleteClassIndex, featLandmarks, totalTime

def readData(filename):
    """
    Params:
        filename: {str} `.xlsx`
    """
    data = np.array(pd.read_excel(filename))
    # data[:, 0] = np.array(list(map(timestamp2unix, data[:, 0])))

    return data

def getSpeedSequenceFromLongitudeAndLatitude(longitude, latitude):
    """
    Params:
        longitude, latitude: {ndarray(N)}
    Returns:
        speedSeq: {ndarray(N)}
    """
    locations = np.r_[longitude, latitude].reshape(2, -1).T.astype(np.float)
    movement  = locations[1:] - locations[:-1]
    a = np.sum(movement**2, axis=1)
    speedSeq  = np.sqrt(np.sum(movement**2, axis=1))
    speedSeq  = np.r_[0, speedSeq]
    return speedSeq


####################################################################################
sequences = np.load('output/1_gpsSpeedSequences.npy')
features  = np.load('output/1_gpsSpeedFeatures.npy')

# -----------------------------------------------------------------------------------

# plt.figure()
# plt.title("Sequence sample")
# plt.xlabel("time(s)")
# plt.ylabel("speed(km/h)")
# plt.plot(sequences[np.random.randint(sequences.shape[0])])
# plt.grid()
# plt.savefig("images/0_sequence_sample.png")

# -----------------------------------------------------------------------------------
# filename = 'data/temp.xlsx'
# data = readData(filename)
# time = data[: 500, 0]
# gpsSpeed = data[: 500, 1]
# calSpeed = getSpeedSequenceFromLongitudeAndLatitude(data[:, 5], data[:, 6])[: 500]

# plt.figure(figsize=(6, 12))
# plt.subplot(211)
# plt.title("GPS Speed")
# plt.xlabel("time(s)")
# plt.ylabel("speed(km/h)")
# plt.plot(time, gpsSpeed)
# plt.subplot(212)
# plt.title("Calculated Speed")
# plt.plot(time, calSpeed)
# plt.xlabel("time(s)")
# plt.ylabel("speed")
# plt.savefig("images/0_gps&cal_speed.png")

# -----------------------------------------------------------------------------------

plt.show()