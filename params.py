# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-20 21:22:41
@LastEditTime: 2019-09-22 17:43:12
@Update: 
'''
# step 1
GAPTHRESH  = 15 * 60  # s
IDLETHRESH = 1.         # km/h，切分序列阈值
MAXSPEEDTHRESH = 4.     # km/h，峰值速度小于该值的序列被删除
MAXPOSACC  =  4         # m/s2，最大加速度大于该值的序列被删除
MAXNEGACC  = -8         # m/s2，最大减速度小于该值的序列被删除
MINTIME    = 10         # s，总时长小于该值的序列被删除
MINRUNTIME = 4          # s，运行时间小于该值的序列被删除

# step 2
n_components_default = 6
n_clusters_default   = 8

# step 3
deleteClassIndex = [0, 3, 7]

# step 4
featLandmarks = [25, 55]
totalTime  = 1300.
sampleTime = 2000

# step 5
k = 16