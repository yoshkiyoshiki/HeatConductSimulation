import math

import cv2
import os
import numpy as np
import glob

import pandas as pd
import seaborn as sns
from numba import jit
from numba import prange
from matplotlib import pyplot as plt
from statistics import stdev, median, mean
import scipy.optimize
from scipy.stats import boxcox
from scipy.special import inv_boxcox


PURPLE = '\033[95m'
CYAN = '\033[96m'
DARKCYAN = '\033[36m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
END = '\033[0m'

k = np.zeros(3)

#95%信頼区間の最小値と最大値
def confidence_interval(k1, k2, k3, avg_boxcox, std_boxcox, lambda_boxcox, confidence_interval_percent):

    if confidence_interval_percent == 68:
        sigma = 1
    elif confidence_interval_percent == 95:
        sigma = 2
    elif confidence_interval_percent == 99:
        sigma = 3

    max_luminance = 10 ** ((np.log10((avg_boxcox + sigma * std_boxcox) * lambda_boxcox + 1)) / lambda_boxcox)
    min_luminance = 10 ** ((np.log10((avg_boxcox - sigma * std_boxcox) * lambda_boxcox + 1)) / lambda_boxcox)
    x1_max = (-k2 + (np.sqrt(k2 * k2 - 4 * (k1 - max_luminance) * k3))) / (2 * k3)
    x2_max = (-k2 - (np.sqrt(k2 * k2 - 4 * (k1 - max_luminance) * k3))) / (2 * k3)

    x1_min = (-k2 + (np.sqrt(k2 * k2 - 4 * (k1 - min_luminance) * k3))) / (2 * k3)
    x2_min = (-k2 - (np.sqrt(k2 * k2 - 4 * (k1 - min_luminance) * k3))) / (2 * k3)

    if np.all(x1_max < x2_max):
        x_max = x1_max
    else:
        x_max = x2_max

    if np.all(x1_min < x2_min):
        x_min = x1_min
    else:
        x_min = x2_min

    return x_max, x_min

def func(x, k1, k2, k3):
    return k1 + k2 * x + k3 * x * x

def trim(image, trim_size):
    trim1 = image.shape[0] // 2 - trim_size // 2
    trim2 = image.shape[0] // 2 + trim_size // 2
    trim3 = image.shape[1] // 2 - trim_size // 2
    trim4 = image.shape[1] // 2 + trim_size // 2
    trimmed = image[trim1:trim2, trim3:trim4]
    return trimmed


# 1次関数近似
def calibrate_1(calibrate_path, cali_ext, calibrate_temp, trim_size):
    print("Start calibration")
    if os.path.isfile("{}/{}".format(calibrate_path, "calibrate_1d.npz")):
        npz = np.load("{}/{}".format(calibrate_path, "calibrate_1d.npz"))
        a = npz["arr_0"]
        b = npz["arr_1"]
    else:
        files = glob.glob("{}/*.{}".format(calibrate_path, cali_ext))
        cali = np.empty((0, trim_size, trim_size))
        for f in files:
            img = trim(cv2.imread(f, 2), trim_size)
            cali = np.vstack((cali, [img]))
        a = np.zeros([trim_size, trim_size])
        b = np.zeros([trim_size, trim_size])
        for i in range(trim_size):
            for j in range(trim_size):
                k = np.polyfit(cali[:, i, j], calibrate_temp, 1)
                a[i][j] = k[0]
                b[i][j] = k[1]
        np.savez("{}/{}".format(calibrate_path, "calibrate_1d"), a, b)
    print("Calibration finished")
    return a, b


# 2次関数近似
def calibrate_2(calibrate_path, cali_ext, calibrate_temp, trim_size):
    print("Start calibration")
    if os.path.isfile("{}/{}".format(calibrate_path, "calibrate_2d.npz")):
        npz = np.load("{}/{}".format(calibrate_path, "calibrate_2d.npz"))
        a = npz["arr_0"]
        b = npz["arr_1"]
        c = npz["arr_2"]
    else:
        files = glob.glob("{}/*.{}".format(calibrate_path, cali_ext))
        cali = np.empty((0, trim_size, trim_size))
        for f in files:
            img = trim(cv2.imread(f, 2), trim_size)
            cali = np.vstack((cali, [img]))
        a = np.zeros([trim_size, trim_size])
        b = np.zeros([trim_size, trim_size])
        c = np.zeros([trim_size, trim_size])
        for i in range(trim_size):
            for j in range(trim_size):
                k = np.polyfit(cali[:, i, j], calibrate_temp, 2)
                a[i][j] = k[0]
                b[i][j] = k[1]
                c[i][j] = k[2]
        np.savez("{}/{}".format(calibrate_path, "calibrate_2d"), a, b, c)
    print("Calibration finished")
    return a, b, c


# 2次関数近似　複数画像
def calibrate_2_2(result_ext, calibrate_path, calibrate_folder, cali_ext, calibrate_temp, trim_size, confidence_interval_percent):
    print("Start calibration")
    if os.path.isfile("{}/{}".format(calibrate_path, "calibrate_2d.npz")):
        npz = np.load("{}/{}".format(calibrate_path, "calibrate_2d.npz"))
        a = npz["arr_0"]
        b = npz["arr_1"]
        c = npz["arr_2"]
    else:
        cali = np.empty((0, trim_size, trim_size))
        err = np.empty((2, len(calibrate_temp)))
        calibrate_temp_buff = np.empty(len(calibrate_temp))
        std_buff = np.empty(1000)
        img_ave_buff = np.empty(5)
        std_boxcox_buff = np.empty(5)
        avg_boxcox_buff = np.empty(5)
        lambda_buff = np.empty(5)
        for i in range(5):
            print('{} calibration'.format(calibrate_folder[i]))
            files = glob.glob("{}/{}/*.{}".format(calibrate_path, calibrate_folder[i], cali_ext))
            img_sum = np.zeros((trim_size, trim_size))
            j = 0
            sns.set(style = 'ticks', font_scale = 2)
            plt.figure(figsize=(12.80, 7.20))
            plt.title('{}C'.format(calibrate_temp[i]))
            calibrate_temp_buff[i] = float(calibrate_temp[i])
            for f in files:
                img = trim(cv2.imread(f, 2), trim_size)     #img = [[352, 365, ~ , 253], [362, 325, ~ , 243], [~] , [332, 315, ~ , 253]] 輝度の値が格納 trimsize×trimsize
                img_sum = img_sum + img                     #輝度を1フレームずつ(j)trimsize×trimsize個足している
                std_buff[j] = int(img[0][0])
                plt.scatter(j, img[0][0], c = 'black')
                plt.plot(j, img[0][0])
                j = j + 1
            std = stdev(std_buff)
            med = median(std_buff)
            avg = mean(std_buff)
            luminance_boxcox = boxcox(std_buff)
            avg_boxcox = mean(luminance_boxcox[0])
            std_boxcox = stdev(luminance_boxcox[0])
            med_boxcox = median(luminance_boxcox[0])
            std_boxcox_buff[i] = std_boxcox
            avg_boxcox_buff[i] = avg_boxcox
            lambda_buff[i] = luminance_boxcox[1]

            # print('std : {}   med : {}  avg : {}'.format(std, med, avg))
            # print('std : {}   med : {}  avg : {}'.format(std_boxcox, med_boxcox, avg_boxcox))
            # print('lambda:{}'.format(luminance_boxcox[1]))

            img_ave = img_sum//j                            #各ピクセルの1000フレーム足したやつを1000で割って平均を出している（trimsize×trimsize個のピクセルを一気に）
            img_ave_buff[i] = img_ave[0][0]
            err[0][i] = 2 * std     #標準偏差エラーバーの下限値
            err[1][i] = 2 * std     #標準偏差エラーバーの上限値
            cali = np.vstack((cali, [img_ave]))
            plt.hlines(img_ave[0][0], 0, 1001, color = 'red', linestyles='solid', label='average = {}'.format(img_ave), linewidth = 3)
            plt.xticks(np.arange(0, 1001, 100))
            plt.yticks(np.arange(200, 501, 50))
            plt.xlabel('flame')
            plt.ylabel('luminance')
            plt.text(800, 470, '標準偏差 : {:.2f}'.format(std), size=18, color = 'black', fontname = 'MS Gothic')
            plt.text(-60, img_ave[0][0], '{}'.format(int(img_ave[0][0])), size=18, color = 'red')
            plt.savefig(
                '{}/{}/{}_{}.{}'.format(calibrate_path, 'flame_luminance', 'flame_luminance', calibrate_temp[i],
                                        result_ext))
            plt.close()

            sns.set(style = 'darkgrid')
            sns.histplot(luminance_boxcox[0], kde = True)
            plt.savefig(
                '{}/{}/{}_{}.{}'.format(calibrate_path, 'normal_dist', 'normal_dist', calibrate_temp[i],
                                        result_ext))
            plt.close()

        #------------------校正曲線を求める----------------------------------------------------------------------------------------------------------------------#
        parameter_initial = np.array([0.0, 0.0, 0.0])
        k, covariance = scipy.optimize.curve_fit(func, calibrate_temp_buff, img_ave_buff, p0 = parameter_initial)
        y = func(calibrate_temp_buff, k[0], k[1], k[2])
        #------------------------------------------------------------------------------------------------------------------------------------------------------#


        #print('{}x^2  {}x  {}'.format(k[2], k[1], k[0]))

        x_max, x_min = np.round(confidence_interval(k[0], k[1], k[2], avg_boxcox_buff, std_boxcox_buff, lambda_buff, confidence_interval_percent), 1)
        list_x = [
            ['', '{}C'.format(calibrate_temp[0]), '{}C'.format(calibrate_temp[1]), '{}C'.format(calibrate_temp[2]), '{}C'.format(calibrate_temp[3]), '{}C'.format(calibrate_temp[4])],
            ['Tmax [C]', x_max[0], x_max[1], x_max[2], x_max[3], x_max[4]],
            ['Tmin [C]', x_min[0], x_min[1], x_min[2], x_min[3], x_min[4]]
        ]
        df = pd.DataFrame(list_x)
        fig, ax = plt.subplots(figsize = (9, 3))
        ax.axis('off')
        confidence_interval_table = ax.table(cellText = df.values, loc = 'center', bbox = [0, 0, 1, 1])
        confidence_interval_table.set_fontsize(18)
        plt.savefig(
                '{}/{}/{}_{}.{}'.format(calibrate_path, 'confidence_interval', 'confidence_interval', confidence_interval_percent,
                                        result_ext))
        plt.close()
        print('max : {}  min : {}'.format(x_max, x_min))

        a = np.zeros([trim_size, trim_size])
        b = np.zeros([trim_size, trim_size])
        c = np.zeros([trim_size, trim_size])
        for i in range(trim_size):
            for j in range(trim_size):
                t = np.polyfit(cali[:, i, j], calibrate_temp, 2)
                a[i][j] = t[0]
                b[i][j] = t[1]
                c[i][j] = t[2]
        #------------------校正曲線を求める----------------------------------------------------------------------------------------------------------------------#
        sns.set(style='ticks', font_scale=2)
        plt.figure(figsize=(7.20, 7.20))
        plt.title('Calibration Curve')
        plt.xticks(np.arange(calibrate_temp[0], calibrate_temp[4] + 1, 5))
        plt.errorbar(calibrate_temp, img_ave_buff, err, marker = '.', markersize = 2, elinewidth = 2, capsize = 3, ecolor = 'black', linestyle = 'None')
        plt.plot(calibrate_temp, img_ave_buff, marker = '.', markersize = 13, mec = 'black', mfc = 'black', linestyle = 'None')
        plt.plot(calibrate_temp, y, '-', color = 'red')
        plt.xlabel('temperature[℃]')
        plt.ylabel('luminance')
        plt.tight_layout()
        plt.savefig('{}/{}/{}.{}'.format(calibrate_path, 'calibration_curve', 'calibration_curve', result_ext))
        #------------------------------------------------------------------------------------------------------------------------------------------------------#
        np.savez("{}/{}".format(calibrate_path, "calibrate_2d"), a, b, c)
        print('{}'.format(err))
    print("Calibration finished")
    return a, b, c


# 遅い
# def image_process(data_path, data_ext, calibrate_path, calibrate_folder, cali_ext, calibrate_temp, trim_size):
#     # a, b = calibrate_1(calibrate_path, cali_ext, calibrate_temp, trim_size)  # 1次関数
#     # a, b, c = calibrate_2(calibrate_path, cali_ext, calibrate_temp, trim_size)  # ２次関数
#     a, b, c = calibrate_2_2(calibrate_path, calibrate_folder, cali_ext, calibrate_temp, trim_size)  # ２次関数　複数画像
#     files = glob.glob("{}/*.{}".format(data_path, data_ext))
#     img_temperature_stack = np.empty((0, trim_size, trim_size))
#     i = 0
#     for f in files:
#         img = trim(cv2.imread(f, cv2.IMREAD_ANYDEPTH), trim_size)
#         # img_temperature = img * a + b
#         img_temperature = img ** 2 * a + b * img + c
#         img_temperature_stack = np.vstack((img_temperature_stack, [img_temperature]))
#         i = i + 1
#         print(i)
#     return img_temperature_stack


@jit(nopython=False, parallel=True)
def image_calculate(img_stack, I_reconst, img_temperature_stack, normalize_temperature_stack, a, b, c):
    for i in prange(img_stack.shape[0]):
        img_temperature_stack[i, :, :] = img_stack[i, :, :] ** 2 * a + b * img_stack[i, :, :] + c
    for i in prange(I_reconst.shape[0]):
        normalize_temperature_stack[i, :, :] = I_reconst[i, :, :] ** 2 * a + b * I_reconst[i, :, :] + c
    return img_temperature_stack, normalize_temperature_stack


def image_process_2(result_path, result_ext, data_path, data_ext, calibrate_path, calibrate_folder, cali_ext, data_temp,
                    calibrate_temp, trim_size, confidence_interval_percent, w):
    a, b, c = calibrate_2_2(result_ext, calibrate_path, calibrate_folder, cali_ext, calibrate_temp, trim_size, confidence_interval_percent)  # ２次関数　複数画像
    files = glob.glob("{}/*.{}".format(data_path, data_ext))
    number_of_images = len(files)
    img_stack = np.empty((number_of_images, trim_size, trim_size))
    i = 0
    for f in files:
        img_stack[i, :, :] = trim(cv2.imread(f, cv2.IMREAD_ANYDEPTH), trim_size)
        i = i+1
        print('loading images: {}/{}'.format(str(i), str(number_of_images)))
    #print('{}'.format(img_stack[1, 200, :]))

    LAMBDA = 8
    THRE = k[0] + k[1] * data_temp[w] + k[2] * data_temp[w] * data_temp[w]

    normalize = img_stack

    B = normalize - THRE
    t = np.sign(B)*B - LAMBDA
    I_reconst = t*(t>0)*np.sign(B)+THRE

    img_temperature_stack, normalize_temperature_stack = image_calculate(img_stack, I_reconst,
                                                                         np.empty((img_stack.shape[0], trim_size, trim_size)), np.empty((normalize.shape[0], trim_size, trim_size)), a, b, c)

    return img_temperature_stack, normalize_temperature_stack
