# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-20 21:22:41
@LastEditTime: 2019-09-21 22:49:11
@Update: 
'''
# step 1
GAPTHRESH  = 15 * 60  # s
IDLETHRESH = 1.
MAXSPEEDTHRESH = 4.
MAXPOSACC  = 6
MAXNEGACC  = 8
MINTIME    = 10
MINRUNTIME = 4

# step 2
n_components_default = 6
n_clusters_default   = 8

# step 3
deleteClassIndex = [3, 4, ]

# step 4
featLandmarks = [25, 60, 100]
totalTime = 1200.