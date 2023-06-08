import os
import numpy as np
from matplotlib import pyplot as plt
import TSPCalibration
import HeatConductionSimulation
import time
import seaborn as sns

start = time.time()

DataType = 1  # 1--Image / 2--Movie
calibrate_path = "data/2022.11.03/calibration1"
calibrate_folder = ['80C', '85C', '90C', '95C', "100C"]  #書き換える
calibrate_temp = [80, 85, 90, 95, 100]
data_temp = [96, 97, 98, 99]
data_size = 3
data_path_hoge = "data/2022.11.03/94C/01/image"  # 実験データの保存場所 #20001020の部分だけ変更
result_path_hoge = "data/2022.11.03/94C/01/result_image" #20001020の部分だけ変更
cali_ext = "tif"
data_ext = cali_ext  # image用
result_ext = "png"  # aviutlにはpngがいい
mpp = 1.2 * 10 ** (-5)  # m/pixel
trim_size = 512
T_max = 100
T_min = 85
FPS = 2000
CONFIDENCE_INTERVAL_PERCENT = 99        #68, 95, 99しかできない
save_surface_temperature = True
heat_conduction_simulation = False
save_npy = True


for w in range(len(data_temp)):
    for j in range(1, data_size + 1):
        data_path = (data_path_hoge[:(data_path_hoge.find('C') - 2)] + str(data_temp[w]) + data_path_hoge[(data_path_hoge.find('C')):(data_path_hoge.find('C') + 2)] + '0' + str(j) + data_path_hoge[(data_path_hoge.find('C') + 4):])
        result_path = (result_path_hoge[:(result_path_hoge.find('C') - 2)] + str(data_temp[w]) + result_path_hoge[(result_path_hoge.find('C')):(data_path_hoge.find('C') + 2)] + '0' + str(j) + data_path_hoge[(data_path_hoge.find('C') + 4):])

        print('data_path: {}'.format(w))
        # make dirs
        os.makedirs(result_path, exist_ok=True)
        if save_surface_temperature:
            os.makedirs('{}/{}'.format(result_path, 'surface_temperature'), exist_ok=True)
            os.makedirs('{}/{}'.format(result_path, 'normalize_surface_temperature'), exist_ok=True)
            os.makedirs('{}/{}'.format(result_path, 'Data/tsp_calibration_data'), exist_ok=True)
            os.makedirs('{}/{}'.format(calibrate_path, 'flame_luminance'), exist_ok=True)
            os.makedirs('{}/{}'.format(calibrate_path, 'calibration_curve'), exist_ok=True)
            os.makedirs('{}/{}'.format(calibrate_path, 'normal_dist'), exist_ok=True)
            os.makedirs('{}/{}'.format(calibrate_path, 'confidence_interval'), exist_ok=True)
            os.makedirs('{}/{}'.format(result_path, 'Data/surface_temp_dist'), exist_ok=True)
        if heat_conduction_simulation:
            os.makedirs('{}/{}'.format(result_path, 'heat_flux'), exist_ok=True)
            os.makedirs('{}/{}'.format(result_path, 'temperature_dist1'), exist_ok=True)
            os.makedirs('{}/{}'.format(result_path, 'temperature_dist2'), exist_ok=True)
            os.makedirs('{}/{}'.format(result_path, 'Data/calc_data'), exist_ok=True)

        # data output
        surface_temperature_stack, normalize_surface_temperature_stack = TSPCalibration.image_process_2(result_path, result_ext, data_path, data_ext, calibrate_path, calibrate_folder,
                                                                   cali_ext, data_temp, calibrate_temp, trim_size, CONFIDENCE_INTERVAL_PERCENT, w)

        if save_npy:
            np.save("{}/{}/{}".format(result_path, "Data/tsp_calibration_data", 'surface_temperature_stack'),
                    surface_temperature_stack)
            np.save("{}/{}/{}".format(result_path, "Data/tsp_calibration_data", 'normalize_surface_temperature_stack'),
                    normalize_surface_temperature_stack)

        value = np.zeros(surface_temperature_stack.shape[0])

        # save surface temperature、温度でスタックしたのを画像にしていくやつ
        if save_surface_temperature:
            for i in range(surface_temperature_stack.shape[0]):
                for k in range(surface_temperature_stack.shape[1]):
                    for j in range(surface_temperature_stack.shape[2]):
                        value[i] = value[i] + surface_temperature_stack[i, k, j]

                plt.figure(figsize=(12.80, 7.20))
                plt.rcParams["font.size"] = 18
                contour_surface_temperature = plt.imshow(np.floor(surface_temperature_stack[i, :, :]), cmap='jet', vmax=T_max,
                                                         vmin=T_min)
                plt.xticks(np.arange(0, trim_size, 1.0 / mpp / 1000),
                           np.round(np.arange(0, trim_size * mpp * 1000, 1.0), decimals=1))
                plt.xlabel('L [mm]')
                plt.ylabel('pixel')
                plt.text(10, -20, "time : {} [msec]".format(str(((1000 / FPS) * i))), size = 18)
                plt.colorbar(contour_surface_temperature)
                plt.savefig(
                    '{}/{}/{}_{}.{}'.format(result_path, 'surface_temperature', 'surface_temperature', str(i).zfill(4),
                                            result_ext))
                print('save images: {}/{}'.format(str(i + 1), str(surface_temperature_stack.shape[0])))
                plt.close()

                plt.figure(figsize=(12.80, 7.20))
                plt.rcParams["font.size"] = 18
                contour_surface_temperature = plt.imshow(np.floor(normalize_surface_temperature_stack[i, :, :]), cmap='jet',
                                                         vmax=T_max,
                                                         vmin=T_min)
                plt.xticks(np.arange(0, trim_size, 1.0 / mpp / 1000),
                           np.round(np.arange(0, trim_size * mpp * 1000, 1.0), decimals=1))
                plt.xlabel('L [mm]')
                plt.ylabel('pixel')
                plt.text(10, -20, "time : {} [msec]".format(str(((1000 / FPS) * i))), size=18)
                plt.colorbar(contour_surface_temperature)
                plt.savefig(
                    '{}/{}/{}_{}.{}'.format(result_path, 'normalize_surface_temperature', 'normalize_surface_temperature', str(i).zfill(4),
                                            result_ext))
                print('save images: {}/{}'.format(str(i + 1), str(normalize_surface_temperature_stack.shape[0])))
                plt.close()

            before_value = value[0]
            for i in range(surface_temperature_stack.shape[0]):
                if (before_value - 10) > value[i]:
                    landing_point = i
                    break
                before_value = value[i]

            # plt.figure(figsize=(12.80, 7.20))
            # plt.rcParams["font.size"] = 18
            # plt.title('surface temperature')
            # plt.xlabel('Length [mm]')
            # plt.ylabel('temperature [℃]')

            x_memori = np.zeros(surface_temperature_stack.shape[1])
            y_memori = np.zeros(surface_temperature_stack.shape[1])

            # col = ['black', 'blue', 'limegreen']
            # for l in range(3):
            #     flame = int(landing_point + ((1 * l) / (1000 / FPS))) #0ms, 5ms, 10ms
            #     for x in range(surface_temperature_stack.shape[1]):
            #         # plt.scatter(x * mpp * 1000, surface_temperature_stack[landing_point, x, 216])
            #         x_memori[x] = x
            #         y_memori[x] = surface_temperature_stack[flame, x, 200]
            #         # print('x:{}  y:{}'.format(x * mpp * 1000, surface_temperature_stack[landing_point, x, 216]))
            #         # plt.plot(x, surface_temperature_stack[landing_point + (10 / (1 / FPS)), x, 216])
            #         # plt.plot(x, surface_temperature_stack[landing_point + (50 / (1 / FPS)), x, 216])
            #     print('flame: {}'.format(flame))
            #     plt.plot(x_memori, y_memori, color = col[l], label = '{:.1f}ms'.format(1.0 * l))
            #
            # for x in range(surface_temperature_stack.shape[1]):
            #     y_memori[x] = data_temp[w]
            # plt.plot(x_memori, y_memori, color = 'red', lw = 3)
            # plt.text(-20, data_temp[w], '{}C'.format(data_temp[w]), size=18, color = 'red', fontname = 'MS Gothic')
            # plt.xticks(np.arange(0, trim_size, 1.0 / mpp / 1000), np.round(np.arange(0, trim_size * mpp * 1000, 1.0), decimals=1))
            # plt.yticks(np.arange(75, 105, 5))
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig("{}/{}/{}.{}".format(result_path, "Data/surface_temp_dist", "surface_temp_dist", result_ext))
            for l in range(surface_temperature_stack.shape[0]):
                plt.figure(figsize=(12.80, 7.20))
                plt.rcParams["font.size"] = 18
                # plt.title('surface temperature')
                plt.xlabel('L [mm]')
                plt.ylabel('temperature [℃]')
                plt.title('{}C : {:.1f}ms'.format(data_temp[w], 0.5 * l))
                for x in range(surface_temperature_stack.shape[1]):
                    # plt.scatter(x * mpp * 1000, surface_temperature_stack[landing_point, x, 216])
                    x_memori[x] = x
                    y_memori[x] = surface_temperature_stack[l, x, 200]
                    # print('x:{}  y:{}'.format(x * mpp * 1000, surface_temperature_stack[landing_point, x, 216]))
                    # plt.plot(x, surface_temperature_stack[landing_point + (10 / (1 / FPS)), x, 216])
                    # plt.plot(x, surface_temperature_stack[landing_point + (50 / (1 / FPS)), x, 216])
                plt.plot(x_memori, y_memori, color='black')
                for x in range(surface_temperature_stack.shape[1]):
                    y_memori[x] = data_temp[w]
                plt.plot(x_memori, y_memori, color='red', lw=3)
                plt.text(-20, data_temp[w], '{}C'.format(data_temp[w]), size=18, color='red', fontname='MS Gothic')
                plt.xticks(np.arange(0, trim_size, 1.0 / mpp / 1000),
                           np.round(np.arange(0, trim_size * mpp * 1000, 1.0), decimals=1))
                plt.yticks(np.arange(80, 111, 5))
                plt.tight_layout()
                plt.savefig("{}/{}/{}{}_{}.{}".format(result_path, "Data/surface_temp_dist", "surface_temp_dist", data_temp[w], l, result_ext))
                plt.close()
            #
            # for x in range(surface_temperature_stack.shape[1]):
            #     y_memori[x] = data_temp[w]
            # plt.plot(x_memori, y_memori, color='red', lw=3)
            # plt.text(-20, data_temp[w], '{}C'.format(data_temp[w]), size=18, color='red', fontname='MS Gothic')
            # plt.xticks(np.arange(0, trim_size, 1.0 / mpp / 1000),
            #            np.round(np.arange(0, trim_size * mpp * 1000, 1.0), decimals=1))
            # plt.yticks(np.arange(75, 105, 5))
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig("{}/{}/{}.{}".format(result_path, "Data/surface_temp_dist", "surface_temp_dist", result_ext))
            #plt.close()

        if heat_conduction_simulation:
            gen = HeatConductionSimulation.heat_conduction(surface_temperature_stack)
            x, y, z = gen.__next__()

            print('{}'.format(surface_temperature_stack[0]))

            if save_npy:
                np.savez("{}/{}/{}".format(result_path, "Data/calc_data", 'xyz'), x, y, z)

            j = 0
            fig1 = plt.figure(figsize=(12.80, 7.20))
            fig2 = plt.figure(figsize=(12.80, 7.20))
            fig3 = plt.figure(figsize=(12.80, 7.20))
            ax1 = fig1.add_subplot(1, 1, 1)
            ax2 = fig2.add_subplot(1, 1, 1)
            ax3 = fig3.add_subplot(1, 1, 1)
            point_temperature = np.zeros(surface_temperature_stack.shape[0])

            for i in gen:
                temperature, heat_flux = i
                if save_npy:
                    np.savez_compressed("{}/{}/{}_{}".format(result_path, "Data/calc_data", 'temperature_heatflux',
                                                             str(j).zfill(4)), temperature, heat_flux)

                # fig1 = plt.figure(figsize=(12.80, 7.20))
                # ax1 = fig1.add_subplot(1, 1, 1)
                # contour_heat_flux = ax1.imshow(heat_flux, cmap='jet', vmin=-5.0 * 10 ** 2, vmax=5.0 * 10 ** 2)
                # fig1.colorbar(contour_heat_flux)
                # fig1.savefig('{}/{}/{}_{}.{}'.format(result_path, 'heat_flux', 'heat_flux', str(j).zfill(4), result_ext))
                # plt.close(fig1)

                fig2 = plt.figure(figsize=(12.80, 7.20))
                ax2 = fig2.add_subplot(1, 1, 1)
                ax2.plot(z[1:101, 207, 207], temperature[1:101, 207, 207], marker='.', markersize=2)
                ax2.set_ylim([55, 105])
                fig2.savefig('{}/{}/{}_{}.{}'.format(result_path, 'temperature_dist2', 'T_dist2', str(j).zfill(4), result_ext),
                             transparent=True)
                # a = 0
                # p = np.zeros(3)
                # min_z = np.zeros(100000000)
                # for l in range(100000000):
                #      min_z[l] = 120
                #
                # for l in range(trim_size + 1):
                #     for m in range(trim_size + 1):
                #         for n in range(101):
                #             if min_z[a] > temperature[n, l, m]:
                #                 min_z[a] = temperature[n, l, m]
                #                 if temperature[n, l, m] < 50:
                #                     p[0] = n
                #                     p[1] = l
                #                     p[2] = m
                #             a = a + 1
                #
                # print('min : {}'.format(min(min_z[np.nonzero(min_z)])))
                # print('{}'.format(p))
                plt.close(fig2)



                '''
                fig3 = plt.figure()
                ax3 = fig3.add_subplot(1, 1, 1)
                contour_temperature_distribution = ax3.imshow(temperature[:, :, 300], cmap='jet', vmin=75, vmax=95)
                fig3.colorbar(contour_temperature_distribution)
                fig3.savefig('{}/{}/{}_{}.{}'.format(result_path, 'temperature_dist1', 'T_dist1', str(j).zfill(4), result_ext))
                plt.close(fig3)
                '''

                # fig3 = plt.figure(figsize=(12.80, 7.20))
                # ax3 = fig3.add_subplot(1, 1, 1)
                # contour_temperature_distribution = ax3.pcolormesh(x[1:101, 1:trim_size + 1, 300] * 1000,
                #                                                   z[1:101, 1:trim_size + 1, 300] * 1000,
                #                                                   temperature[1:101, 1:trim_size + 1, 300],
                #                                                   cmap='jet', vmin=55, vmax=105, shading='gouraud')
                # ax3.set_aspect('equal')
                # ax3.invert_xaxis()
                # ax3.invert_yaxis()
                # fig3.colorbar(contour_temperature_distribution)
                # fig3.savefig('{}/{}/{}_{}.{}'.format(result_path, 'temperature_dist1', 'T_dist1', str(j).zfill(4), result_ext))
                # plt.close(fig3)

                point_temperature[j] = temperature[1, 300, 300]

                j = j + 1
            plt.close()

            # fig4 = plt.figure(figsize=(12.80, 7.20))
            # ax4 = fig4.add_subplot(1, 1, 1)
            # ax4.plot(point_temperature, marker='.', markersize=2)
            # fig2.savefig('{}/{}/{}.{}'.format(result_path, 'temperature_dist1', 'point_temperature', result_ext),
            #              transparent=True)
            # plt.close(fig4)



        elapsed_time = time.time() - start
        print("\nelapsed_time:{0}".format(elapsed_time) + "[sec]")