import numpy as np
# import time
# from GaussElimination import GaussElimination1D_2
# from matplotlib import pyplot as plt
# import cv2
from numba import jit
from numba import prange
# import os


@jit(nopython=True, parallel=True)
def steady(size_z, size_x, size_y, t, a_p, a_e, a_w, a_n, a_s, a_t, a_b, b, a_p0, t_p0, t_pm1):
    res1 = np.zeros((size_z, size_x, size_y), dtype=np.float64)
    # a_p[:,:,:]=a_p[:,:,:]-a_p0[:,:,:]
    # b[:,:,:]=0

    eps = 0.001  #誤差限界
    omega = 1.7
    # t[i,j,k]=Tp

    for j in prange(1, size_x - 1):
        for k in range(1, size_y - 1):
            for i in range(2, size_z - 1):#最初は実験値なので2から
                t[i, j, k] = (1 - omega) * t[i, j, k] + omega * (
                        a_e[i, j, k] * t[i, j + 1, k] + a_w[i, j, k] * t[i, j - 1, k] + a_n[i, j, k] * t[
                    i, j, k - 1] + a_s[i, j, k] * t[i, j, k + 1] + a_t[i, j, k] * t[i - 1, j, k] + a_b[i, j, k] * t[
                            i + 1, j, k] + b[i, j, k]) / a_p[i, j, k]#三次元の離散化方程式に基づいてTpを導出する

    for j in prange(1, size_x - 1):
        for k in range(1, size_y - 1):
            for i in range(2, size_z - 1):
                res1[i, j, k] = (b[i, j, k] - a_p[i, j, k] * t[i, j, k] + a_e[i, j, k] * t[i, j + 1, k] + a_w[i, j, k] *
                                 t[i, j - 1, k] + a_n[i, j, k] * t[i, j, k - 1] + a_s[i, j, k] * t[i, j, k + 1] + a_t[
                                     i, j, k] * t[i - 1, j, k] + a_b[i, j, k] * t[i + 1, j, k]) ** 2
    # 式をすべて右辺に移して、残差を計算
    res = np.sqrt(np.sum(res1))
    print(res)

    while res >= eps:
        for j in prange(1, size_x - 1):
            for k in range(1, size_y - 1):
                for i in range(1, size_z - 1):
                    t[i, j, k] = (1 - omega) * t[i, j, k] + omega * (
                            a_e[i, j, k] * t[i, j + 1, k] + a_w[i, j, k] * t[i, j - 1, k] + a_n[i, j, k] * t[
                        i, j, k - 1] + a_s[i, j, k] * t[i, j, k + 1] + a_t[i, j, k] * t[i - 1, j, k] + a_b[i, j, k] * t[
                                i + 1, j, k] + b[i, j, k]) / a_p[i, j, k]

        for j in prange(1, size_x - 1):
            for k in range(1, size_y - 1):
                for i in range(1, size_z - 1):
                    res1[i, j, k] = (b[i, j, k] - a_p[i, j, k] * t[i, j, k] + a_e[i, j, k] * t[i, j + 1, k] + a_w[
                        i, j, k] * t[i, j - 1, k] + a_n[i, j, k] * t[
                                         i, j, k - 1] + a_s[i, j, k] * t[i, j, k + 1] + a_t[i, j, k] * t[i - 1, j, k] +
                                     a_b[i, j, k] * t[i + 1, j, k]) ** 2

        res = np.sqrt(np.sum(res1))
        print(res)
    return t


@jit(nopython=True, parallel=True)
def gauss_seidel(size_z, size_x, size_y, t, a_p, a_e, a_w, a_n, a_s, a_t, a_b, b, a_p0, t_p0, t_pm1):
    res1 = np.zeros((size_z, size_x, size_y), dtype=np.float64)
    res2 = np.zeros((size_z, size_x, size_y), dtype=np.float64)
    eps = 0.001

    for j in prange(1, size_x - 1):
        for k in range(1, size_y - 1):
            for i in range(2, size_z - 1):
                b[i, j, k] = a_p0[i, j, k] * t_p0[i, j, k]
                t[i, j, k] = (a_e[i, j, k] * t[i, j + 1, k] + a_w[i, j, k] * t[i, j - 1, k] + a_n[i, j, k] * t[
                    i, j, k - 1] + a_s[i, j, k] * t[i, j, k + 1] + a_t[i, j, k] * t[i - 1, j, k] + a_b[i, j, k] * t[
                                  i + 1, j, k] + b[i, j, k]) / a_p[i, j, k]

    for j in prange(1, size_x - 1):
        for k in range(1, size_y - 1):
            for i in range(2, size_z - 1):
                res1[i, j, k] = (b[i, j, k] - a_p[i, j, k] * t[i, j, k] + a_e[i, j, k] * t[i, j + 1, k] + a_w[i, j, k] *
                                 t[i, j - 1, k] + a_n[i, j, k] * t[i, j, k - 1] + a_s[i, j, k] * t[i, j, k + 1] + a_t[
                                     i, j, k] * t[i - 1, j, k] + a_b[i, j, k] * t[i + 1, j, k]) ** 2
                res2[i, j, k] = b[i, j, k] ** 2

    res = np.sqrt(np.sum(res1) / np.sum(res2))
    #print(res)

    while res >= eps:
        for j in prange(1, size_x - 1):
            for k in range(1, size_y - 1):
                for i in range(2, size_z - 1):
                    t[i, j, k] = (a_e[i, j, k] * t[i, j + 1, k] + a_w[i, j, k] * t[i, j - 1, k] + a_n[i, j, k] * t[
                        i, j, k - 1] + a_s[i, j, k] * t[
                                      i, j, k + 1] + a_t[i, j, k] * t[i - 1, j, k] + a_b[i, j, k] * t[i + 1, j, k] + b[
                                      i, j, k]) / a_p[i, j, k]

        for j in prange(1, size_x - 1):
            for k in range(1, size_y - 1):
                for i in range(2, size_z - 1):
                    res1[i, j, k] = (b[i, j, k] - a_p[i, j, k] * t[i, j, k] + a_e[i, j, k] * t[i, j + 1, k] + a_w[
                        i, j, k] * t[i, j - 1, k] + a_n[i, j, k] * t[
                                         i, j, k - 1] + a_s[i, j, k] * t[i, j, k + 1] + a_t[i, j, k] * t[i - 1, j, k] +
                                     a_b[i, j, k] * t[i + 1, j, k]) ** 2
                    res2[i, j, k] = b[i, j, k] ** 2

        res = np.sqrt(np.sum(res1) / np.sum(res2))
        #print(res)

    return t


#sor法
@jit(nopython=True, parallel=True)
def sor(size_z, size_x, size_y, t, a_p, a_e, a_w, a_n, a_s, a_t, a_b, b, a_p0, t_p0, t_pm1):
    res1 = np.zeros((size_z, size_x, size_y), dtype=np.float64)
    res2 = np.zeros((size_z, size_x, size_y), dtype=np.float64)
    eps = 10**(-5)
    omega = 1.7
    res = eps + 1

    while res >= eps:
        for j in prange(1, size_x - 1):
            for k in range(1, size_y - 1):
                for i in range(1, size_z - 1):
                    t[i, j, k] = (1 - omega) * t[i, j, k] + omega * (
                            a_e[i, j, k] * t[i, j + 1, k] + a_w[i, j, k] * t[i, j - 1, k] + a_n[i, j, k] * t[
                        i, j, k - 1] + a_s[i, j, k] * t[i, j, k + 1] + a_t[i, j, k] * t[i - 1, j, k] + a_b[i, j, k] * t[
                                i + 1, j, k] + b[i, j, k]) / a_p[i, j, k]

        for j in prange(1, size_x - 1):
            for k in range(1, size_y - 1):
                for i in range(1, size_z - 1):
                    '''
                    res1[i, j, k] = (b[i, j, k] - a_p[i, j, k] * t[i, j, k] + a_e[i, j, k] * t[i, j + 1, k] + a_w[
                        i, j, k] * t[i, j - 1, k] + a_n[i, j, k] * t[
                                         i, j, k - 1] + a_s[i, j, k] * t[i, j, k + 1] + a_t[i, j, k] * t[i - 1, j, k] +
                                     a_b[i, j, k] * t[i + 1, j, k]) ** 2
                    res2[i, j, k] = b[i, j, k] ** 2
                    '''
                    #2乗誤差を求める
                    res1[i, j, k] = (b[i, j, k] - a_p[i, j, k] * t[i, j, k] + a_e[i, j, k] * t[i, j + 1, k] + a_w[
                        i, j, k] * t[i, j - 1, k] + a_n[i, j, k] * t[
                                         i, j, k - 1] + a_s[i, j, k] * t[i, j, k + 1] + a_t[i, j, k] * t[i - 1, j, k] +
                                     a_b[i, j, k] * t[i + 1, j, k]) ** 2
                    #
                    res2[i, j, k] = (a_p[i, j, k] * t[i, j, k]) ** 2

        # res = np.sqrt(np.sum(res1) / np.sum(res2))

        #すべての誤差で一番大きい値がepsより小さかったら
        res = np.amax(np.sqrt(res1[1:size_z - 1, 1:size_x - 1, 1:size_y - 1] / res2[1:size_z - 1, 1:size_x - 1, 1:size_y - 1]))
        print(res)

    return t


@jit(nopython=True, parallel=True)
def sor_old(size_z, size_x, size_y, t, a_p, a_e, a_w, a_n, a_s, a_t, a_b, b, a_p0, t_p0, t_pm1):
    res1 = np.zeros((size_z, size_x, size_y), dtype=np.float64)
    res2 = np.zeros((size_z, size_x, size_y), dtype=np.float64)
    eps = 0.001
    omega = 1.7

    res = eps + 1

    while res >= eps:
        for j in prange(2, size_x - 2):
            for k in range(2, size_y - 2):
                for i in range(2, size_z - 2):
                    t[i, j, k] = (1 - omega) * t[i, j, k] + omega * (
                            a_e[i, j, k] * t[i, j + 1, k] + a_w[i, j, k] * t[i, j - 1, k] + a_n[i, j, k] * t[
                        i, j, k - 1] + a_s[i, j, k] * t[i, j, k + 1] + a_t[i, j, k] * t[i - 1, j, k] + a_b[i, j, k] * t[
                                i + 1, j, k] + b[i, j, k]) / a_p[i, j, k]

        for j in prange(2, size_x - 2):
            for k in range(2, size_y - 2):
                for i in range(2, size_z - 2):
                    res1[i, j, k] = (b[i, j, k] - a_p[i, j, k] * t[i, j, k] + a_e[i, j, k] * t[i, j + 1, k] + a_w[
                        i, j, k] * t[i, j - 1, k] + a_n[i, j, k] * t[
                                         i, j, k - 1] + a_s[i, j, k] * t[i, j, k + 1] + a_t[i, j, k] * t[i - 1, j, k] +
                                     a_b[i, j, k] * t[i + 1, j, k]) ** 2
                    res2[i, j, k] = b[i, j, k] ** 2

        res = np.sqrt(np.sum(res1) / np.sum(res2))
        # print(res)

    return t


#共役勾配法
def conjugate_gradient(n, size_x, size_y, t, a_p, a_e, a_w, a_n, a_s, a_t, a_b, b, a_p0, t_p0, t_pm1):
    r = np.zeros((n + 1, size_x, size_y), dtype=np.float64)

    for j in prange(1, size_x - 1):
        for k in range(1, size_y - 1):
            for i in range(1, n):
                b[i, j, k] = a_p0[i, j, k] * t_p0[i, j, k]
                r[i, j, k] = (a_e[i, j, k] * t[i, j + 1, k] + a_w[i, j, k] * t[i, j - 1, k] + a_n[i, j, k] * t[
                    i, j, k - 1] + a_s[i, j, k] * t[i, j, k + 1] + a_t[i, j, k] * t[i - 1, j, k] + a_b[i, j, k] * t[
                                  i + 1, j, k] + b[i, j, k])
