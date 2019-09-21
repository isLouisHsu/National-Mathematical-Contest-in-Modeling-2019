# -*- coding: utf-8 -*-
'''
@Description: 
@Version: 1.0.0
@Author: louishsu
@Github: https://github.com/isLouisHsu
@E-mail: is.louishsu@foxmail.com
@Date: 2019-09-19 14:14:34
@LastEditTime: 2019-09-21 21:54:59
@Update: 
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import pywt

def plot_signal_decomp(data, w, title):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    
    https://github.com/PyWavelets/pywt/blob/master/demo/dwt_signal_decomposition.py
    """
    mode = pywt.Modes.smooth
    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []
    for i in range(5):
        (a, d) = pywt.dwt(a, w, mode)
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(pywt.waverec(coeff_list, w))

    fig = plt.figure()
    ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    ax_main.set_title(title)
    ax_main.plot(data)
    ax_main.set_xlim(0, len(data) - 1)

    for i, y in enumerate(rec_a):
        ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
        ax.plot(y, 'r')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("A%d" % (i + 1))

    for i, y in enumerate(rec_d):
        ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
        ax.plot(y, 'g')
        ax.set_xlim(0, len(y) - 1)
        ax.set_ylabel("D%d" % (i + 1))

def dwtDecompose(x, n):
    """
    Params:
        x: {ndarray(N)}
    """
    mode = pywt.Modes.smooth
    w = pywt.Wavelet('sym5')
    
    a = x.copy()
    for i in range(n):
        a, _ = pywt.dwt(a, w, mode)
    reca = pywt.waverec([a] + [None] * n, w)
    return reca

if __name__ == "__main__":

    data1 = np.concatenate((np.arange(1, 400),
                            np.arange(398, 600),
                            np.arange(601, 1024)))
    plot_signal_decomp(data1, 'coif5', "DWT: Signal irregularity")

    x = np.linspace(0.082, 2.128, num=1024)[::-1]
    data2 = np.sin(40 * np.log(x)) * np.sign((np.log(x)))
    plot_signal_decomp(data2, 'sym5',
                    "DWT: Frequency and phase change - Symmlets5")

    import pywt.data
    ecg = pywt.data.ecg()
    plot_signal_decomp(ecg, 'sym5', "DWT: Ecg sample - Symmlets5")
    plt.show()