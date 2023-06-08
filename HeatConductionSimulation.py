import numpy as np
import solver


def heat_conduction(boundary_condition):
    heat_conduction_2d = True

    # meter per pixel
    mpp = 1.9 * 10 ** (-6)  # m/pixel

    # 物質の数
    #    S = 1

    # 1層目の厚さ [m]
    tsp_thickness = 0  #tsp厚さ

    # 物性値
    # TSP
    k_1 = 0.37  # W/(m-K)
    rho_1 = 1200  # kg/m3
    c_1 = 1300  # J/(g-K)
    # サファイア
    k_2 = 25  # W/(m-K)
    rho_2 = 3980  # kg/m3
    c_2 = 780  # J/(g-K)


    # 空間離散化
    z_max = 100  # 分割数
    length = 5  # [mm]
    r = 1.1
    expression_type = 1  # y = a1 * x --- 1, y = a1 * a2 ^ x --- 2

    # 時間離散化
    time_step_size = 0.0005

    # 誤差
    eps = 10 ** (-3)

    # ------------------------------------------------------------------------------------------

    x_max = boundary_condition.shape[1]
    y_max = boundary_condition.shape[2]
    size_x = x_max + 2
    size_y = y_max + 2
    size_z = z_max + 2

    number_of_time_steps = boundary_condition.shape[0]

    # 初期条件
    t_ini_1 = np.zeros([size_z, size_x, size_y])
    # t_ini_1[1:z_max +1, 1:x_max + 1, 1:y_max + 1] = np.average(boundary_condition[0, :, :])
    t_ini_1[2:z_max, 2:x_max, 2:y_max] = np.average(boundary_condition[0, :, :])  # 一枚目の全体の温度の平均を初期値

    # 境界条件
    # 上
    x_bou_top = 1   # 境界条件の位置
    t_bou_top = np.zeros([size_x, size_y])
    t_bou_top[1:x_max + 1, 1:y_max + 1] = boundary_condition[0, :, :]

    # 下
    # x_bou_b = z_max
    # t_bou_b = np.zeros([size_x, size_y])
    # t_bou_b[1:x_max+1, 1:y_max+1] = boundary_condition[0, :, :]

    # ------------------------------------------------------------------------------------------

    # 初期化
    x = np.zeros([size_z, size_x, size_y])
    y = np.zeros([size_z, size_x, size_y])
    z = np.zeros([size_z, size_x, size_y])
    k = np.zeros([size_z, size_x, size_y])
    k_t = np.zeros([size_z, size_x, size_y])
    k_b = np.zeros([size_z, size_x, size_y])
    rho = np.zeros([size_z, size_x, size_y])
    c = np.zeros([size_z, size_x, size_y])
    a_p = np.ones([size_z, size_x, size_y])
    a_e = np.zeros([size_z, size_x, size_y])
    a_w = np.zeros([size_z, size_x, size_y])
    a_n = np.zeros([size_z, size_x, size_y])
    a_s = np.zeros([size_z, size_x, size_y])
    a_t = np.zeros([size_z, size_x, size_y])
    a_b = np.zeros([size_z, size_x, size_y])
    b = np.zeros([size_z, size_x, size_y])

    t = np.zeros([size_z, size_x, size_y])

    a_p0 = np.zeros([size_z, size_x, size_y])
    t_p0 = np.zeros([size_z, size_x, size_y])
    t_pm1 = np.zeros([size_z, size_x, size_y])

    dz = np.zeros([size_z, size_x, size_y])

    res1 = np.zeros([size_x, size_y])
    res2 = np.zeros([size_x, size_y])
    res_w = np.zeros([size_z, size_y])
    res_e = np.zeros([size_z, size_y])
    res_n = np.zeros([size_z, size_x])
    res_s = np.zeros([size_z, size_x])

    heat_flux = np.zeros([size_x, size_y])
    # zeros = np.zeros([size_z, size_x, size_y])

    #----------------------------------------------------------------------------------------------------------------

    # メッシュ作成
    if expression_type == 1:
        for i in range(size_z):
            a1 = length / z_max
            z[i, :, :] = (i - 1) * a1
    if expression_type == 2:
        a1 = length / (r ** (z_max + 1) - r)
        for i in range(size_z):
            z[i, :, :] = a1 * (r ** i - r)
    # dz[0, :, :] = (x[1, :, :] - x[0, :, :]) / 2
    # dz[n + 1, :, :] = (x[size_z, :, :] - x[size_z - 1, :, :]) / 2
    for i in range(size_x):
        x[:, i, :] = mpp * (i - 1)
    for i in range(size_y):
        y[:, :, i] = mpp * (i - 1)

    # メッシュ出力
    yield x, y, z

    for i in range(2, z_max):
        dz[i, :, :] = (z[i + 1, :, :] - z[i, :, :]) / 2 + (z[i, :, :] - z[i - 1, :, :]) / 2

    # 境界の位置
    z_tsp_substrate = int(tsp_thickness//dz[2,0,0])

    # 物性値
    k[1:1+z_tsp_substrate,:,:] = k_1
    k[1+z_tsp_substrate:z_max+1,:,:] = k_2
    rho[1:1+z_tsp_substrate,:,:] = rho_1
    rho[1+z_tsp_substrate:z_max+1,:,:] = rho_2
    c[1:1+z_tsp_substrate,:,:] = c_1
    c[1+z_tsp_substrate:z_max+1,:,:] = c_2

    # 初期値
    t[:, :, :] = t_ini_1[:, :, :]
    t_p0[:, :, :] = t_ini_1[:, :, :]
    t_pm1[:, :, :] = t_ini_1[:, :, :]

    # t[1, :, :] = t_bou_t[:, :]
    # t[z_max, :, :] = t_bou_b[:, :]

    dt = time_step_size


    # 係数
    for i in range(2, z_max):
        k_b[i, 2:x_max, 2:y_max] = 2 * k[i,2:x_max, 2:y_max]*k[i+1,2:x_max, 2:y_max]/(k[i,2:x_max, 2:y_max]+k[i+1,2:x_max, 2:y_max])
        k_t[i, 2:x_max, 2:y_max] = 2 * k[i, 2:x_max, 2:y_max] * k[i - 1, 2:x_max, 2:y_max] / (k[i, 2:x_max, 2:y_max] + k[i - 1, 2:x_max, 2:y_max])
        a_b[i, 2:x_max, 2:y_max] = k_b[i, 2:x_max, 2:y_max] * mpp * mpp / (z[i + 1, 2:x_max, 2:y_max] - z[i, 2:x_max, 2:y_max])
        a_t[i, 2:x_max, 2:y_max] = k_t[i, 2:x_max, 2:y_max] * mpp * mpp / (z[i, 2:x_max, 2:y_max] - z[i - 1, 2:x_max, 2:y_max])
        if heat_conduction_2d:
            a_e[i, 2:x_max, 2:y_max] = k[i, 2:x_max, 2:y_max] * mpp * dz[i, 2:x_max, 2:y_max] / mpp  # m/pixel
            a_w[i, 2:x_max, 2:y_max] = k[i, 2:x_max, 2:y_max] * mpp * dz[i, 2:x_max, 2:y_max] / mpp
            a_n[i, 2:x_max, 2:y_max] = k[i, 2:x_max, 2:y_max] * mpp * dz[i, 2:x_max, 2:y_max] / mpp
            a_s[i, 2:x_max, 2:y_max] = k[i, 2:x_max, 2:y_max] * mpp * dz[i, 2:x_max, 2:y_max] / mpp
        else:
            a_e[2:z_max, 2:x_max, 2:y_max] = 0  # m/pixel
            a_w[2:z_max, 2:x_max, 2:y_max] = 0
            a_n[2:z_max, 2:x_max, 2:y_max] = 0
            a_s[2:z_max, 2:x_max, 2:y_max] = 0
        a_p0[i, 2:x_max, 2:y_max] = rho[i, 2:x_max, 2:y_max] * c[i, 2:x_max, 2:y_max] * mpp * mpp * dz[i, 2:x_max, 2:y_max] / dt

        # a_p[i, 2:x_max, 2:y_max] = a_e[i, 2:x_max, 2:y_max] + a_w[i, 2:x_max, 2:y_max] + a_n[i, 2:x_max, 2:y_max] + a_s[i, 2:x_max, 2:y_max] + a_t[i, 2:x_max, 2:y_max] + a_b[i, 2:x_max, 2:y_max] \
        #                                                                                         + a_p0[i, 2:x_max, 2:y_max]

        # 時間微分　二次精度？
        a_p[i, 2:x_max, 2:y_max] = a_e[i, 2:x_max, 2:y_max] + a_w[i, 2:x_max, 2:y_max] + a_n[i, 2:x_max, 2:y_max] + a_s[i, 2:x_max, 2:y_max] + a_t[i, 2:x_max, 2:y_max] + a_b[i, 2:x_max, 2:y_max] \
                                                                                                + 3/2 * a_p0[i, 2:x_max, 2:y_max]

    # 境界条件
    # 上端
    a_e[1, 2:x_max, 2:y_max] = 0
    a_w[1, 2:x_max, 2:y_max] = 0
    a_n[1, 2:x_max, 2:y_max] = 0
    a_s[1, 2:x_max, 2:y_max] = 0
    a_t[1, 2:x_max, 2:y_max] = 1
    a_b[1, 2:x_max, 2:y_max] = 0
    a_p[1, 2:x_max, 2:y_max] = 1
    # b[1, 1:x_max+1, 1:y_max+1] = t[1, 1:x_max+1, 1:y_max+1]
    b[1, 2:x_max, 2:y_max] = t_bou_top[2:x_max, 2:y_max]  # x_bou_t = 1 の時用

    # 下端
    # ディリクレ
    # a_e[n, :, :] = 0
    # a_w[n, :, :] = 0
    # a_p[n, :, :] = 1
    # b[n, :, :] = t[n, :, :]
    # ノイマン
    a_e[z_max, 2:x_max, 2:y_max] = 0
    a_w[z_max, 2:x_max, 2:y_max] = 0
    a_n[z_max, 2:x_max, 2:y_max] = 0
    a_s[z_max, 2:x_max, 2:y_max] = 0
    a_t[z_max, 2:x_max, 2:y_max] = 1
    a_b[z_max, 2:x_max, 2:y_max] = 0
    a_p[z_max, 2:x_max, 2:y_max] = 1
    b[z_max, 2:x_max, 2:y_max] = 0

    # 西端
    # ノイマン
    a_e[2:z_max, 1, 2:y_max] = 1
    a_w[2:z_max, 1, 2:y_max] = 0
    a_n[2:z_max, 1, 2:y_max] = 0
    a_s[2:z_max, 1, 2:y_max] = 0
    a_t[2:z_max, 1, 2:y_max] = 0
    a_b[2:z_max, 1, 2:y_max] = 0
    a_p[2:z_max, 1, 2:y_max] = 1
    b[2:z_max, 1, 2:y_max] = 0

    # 東端
    # ノイマン
    a_e[2:z_max, x_max, 2:y_max] = 0
    a_w[2:z_max, x_max, 2:y_max] = 1
    a_n[2:z_max, x_max, 2:y_max] = 0
    a_s[2:z_max, x_max, 2:y_max] = 0
    a_t[2:z_max, x_max, 2:y_max] = 0
    a_b[2:z_max, x_max, 2:y_max] = 0
    a_p[2:z_max, x_max, 2:y_max] = 1
    b[2:z_max, x_max, 2:y_max] = 0

    # 北端
    # ノイマン
    a_e[2:z_max, 2:x_max, 1] = 0
    a_w[2:z_max, 2:x_max, 1] = 0
    a_n[2:z_max, 2:x_max, 1] = 0
    a_s[2:z_max, 2:x_max, 1] = 1
    a_t[2:z_max, 2:x_max, 1] = 0
    a_b[2:z_max, 2:x_max, 1] = 0
    a_p[2:z_max, 2:x_max, 1] = 1
    b[2:z_max, 2:x_max, 1] = 0

    # 南端
    a_e[2:z_max, 2:x_max, y_max] = 0
    a_w[2:z_max, 2:x_max, y_max] = 0
    a_n[2:z_max, 2:x_max, y_max] = 1
    a_s[2:z_max, 2:x_max, y_max] = 0
    a_t[2:z_max, 2:x_max, y_max] = 0
    a_b[2:z_max, 2:x_max, y_max] = 0
    a_p[2:z_max, 2:x_max, y_max] = 1
    b[2:z_max, 2:x_max, y_max] = 0

    print('\ninit')
    # t = solver.sor(size_z, size_x, size_y, zeros, a_p-a_p0, a_e, a_w, a_n, a_s, a_t, a_b, b, zeros, t_p0, t_pm1)

    # ガウス消去法
    for step in range(number_of_time_steps):
        print('\nStep: {}/{}-----------------------------------------------'.format(str(step + 1),
                                                                                    str(number_of_time_steps)))

        t_bou_top[1:x_max + 1, 1:y_max + 1] = boundary_condition[step, :, :]

        # b[2:z_max, 2:x_max, 2:y_max] = a_p0[2:z_max, 2:x_max, 2:y_max] * t_p0[2:z_max, 2:x_max, 2:y_max]

        # 時間微分　二次精度？
        b[2:z_max, 2:x_max, 2:y_max] = 2 * a_p0[2:z_max, 2:x_max, 2:y_max] * t_p0[2:z_max, 2:x_max, 2:y_max] - 0.5 * a_p0[2:z_max, 2:x_max, 2:y_max] * t_pm1[2:z_max, 2:x_max, 2:y_max]

        b[1, 2:x_max, 2:y_max] = t_bou_top[2:x_max, 2:y_max]  # 上端　ディリクレ    # TSPの熱伝導絵を考慮しない
        # b[1, 2:x_max, 2:y_max] = t_p0[1, 2:x_max, 2:y_max]                     # TSPの熱伝導絵を考慮する
        b[z_max, 2:x_max, 2:y_max] = 0  # 下端　ノイマン
        b[1:z_max + 1, 1, 1:y_max + 1] = 0  # 西端  ノイマン
        b[1:z_max + 1, x_max, 1:y_max + 1] = 0  # 東端　ノイマン
        b[1:z_max + 1, 2:x_max, 1] = 0  # 北端　ノイマン
        b[1:z_max + 1, 2:x_max, y_max] = 0  # 南端　ノイマン

        t = solver.sor(size_z, size_x, size_y, t, a_p, a_e, a_w, a_n, a_s, a_t, a_b, b, a_p0, t_p0, t_pm1)

        # 条件との残差
        res1[2:x_max, 2:y_max] = t_bou_top[2:x_max, 2:y_max] - t[x_bou_top, 2:x_max, 2:y_max]
        # res2[:, :] = t_bou_r[:, :] - t[x_bou_r, :, :]
        res2[2:x_max, 2:y_max] = t[z_max - 1, 2:x_max, 2:y_max] - t[z_max, 2:x_max, 2:y_max]
        res_w[2:z_max, 2:y_max] = t[2:z_max, 2, 2:y_max] - t[2:z_max, 1, 2:y_max]
        res_e[2:z_max, 2:y_max] = t[2:z_max, x_max - 1, 2:y_max] - t[2:z_max, x_max, 2:y_max]
        res_n[2:z_max, 2:x_max] = t[2:z_max, 2:x_max, 2] - t[2:z_max, 2:x_max, 1]
        res_s[2:z_max, 2:x_max] = t[2:z_max, 2:x_max, y_max - 1] - t[2:z_max, 2:x_max, y_max]

        print(np.amax(np.abs(res1)), np.amax(np.abs(res2)), np.amax(np.abs(res_w)), np.amax(np.abs(res_e)),
              np.amax(np.abs(res_n)), np.amax(np.abs(res_s)))

        # 条件を満たすまで繰り返し
        while np.amax(np.abs(res1)) >= eps or np.amax(np.abs(res2)) >= eps or np.amax(np.abs(res_w)) >= eps \
                or np.amax(np.abs(res_e)) >= eps or np.amax(np.abs(res_n)) >= eps or np.amax(np.abs(res_s)) >= eps:
            t[1, 2:x_max, 2:y_max] = t[1, 2:x_max, 2:y_max] + res1[2:x_max, 2:y_max]
            b[1, 1:x_max + 1, 1:y_max + 1] = t[1, 1:x_max + 1, 1:y_max + 1]

            t = solver.sor(size_z, size_x, size_y, t, a_p, a_e, a_w, a_n, a_s, a_t, a_b, b, a_p0, t_p0, t_pm1)

            # 条件との残差
            res1[2:x_max, 2:y_max] = t_bou_top[2:x_max, 2:y_max] - t[x_bou_top, 2:x_max, 2:y_max]
            # res2[:, :] = t_bou_r[:, :] - t[x_bou_r, :, :]
            res2[2:x_max, 2:y_max] = t[z_max - 1, 2:x_max, 2:y_max] - t[z_max, 2:x_max, 2:y_max]
            res_w[2:z_max, 2:y_max] = t[2:z_max, 2, 2:y_max] - t[2:z_max, 1, 2:y_max]
            res_e[2:z_max, 2:y_max] = t[2:z_max, x_max - 1, 2:y_max] - t[2:z_max, x_max, 2:y_max]
            res_n[2:z_max, 2:x_max] = t[2:z_max, 2:x_max, 2] - t[2:z_max, 2:x_max, 1]
            res_s[2:z_max, 2:x_max] = t[2:z_max, 2:x_max, y_max - 1] - t[2:z_max, 2:x_max, y_max]

            print(np.amax(np.abs(res1)), np.amax(np.abs(res2)), np.amax(np.abs(res_w)), np.amax(np.abs(res_e)),
                  np.amax(np.abs(res_n)), np.amax(np.abs(res_s)))

        # 温度分布更新
        t_pm1[:, :, :] = t_p0[:, :, :]
        t_p0[:, :, :] = t[:, :, :]

        heat_flux[1:x_max + 1, 1:y_max + 1] = k[1, 1:x_max + 1, 1:y_max + 1] * (t[1, 1:x_max + 1, 1:y_max + 1] - t[2, 1:x_max + 1, 1:y_max + 1]) / (
                z[2, 1:x_max + 1, 1:y_max + 1] - z[1, 1:x_max + 1, 1:y_max + 1]) / 10000  # W/cm^2

        yield t, heat_flux

    # elapsed_time = time.time() - start
    # print("elapsed_time:{0}".format(elapsed_time) + "[sec]")


def heat_conduction_old(boundary_condition):
    heat_conduction_2d = True

    # 物質の数
    #    S = 1

    # meter per pixel
    mpp = 1.9 * 10 ** (-6)

    # 物性値
    k = 25  # W/(m-K)
    rho = 3980  # kg/m3
    c = 780  # J/(g-K)

    # 空間離散化
    z_max = 100  # 分割数
    length = 0.005  # [m]
    r = 1.1
    expression_type = 1  # y = a1 * x --- 1, y = a1 * a2 ^ x --- 2

    # 時間離散化
    time_step_size = 0.001

    # 誤差
    eps = 0.001

    # ------------------------------------------------------------------------------------------

    x_max = boundary_condition.shape[1]
    y_max = boundary_condition.shape[2]
    size_x = x_max + 2
    size_y = y_max + 2

    size_z = z_max + 2

    number_of_time_steps = boundary_condition.shape[0]

    # 初期条件
    t_ini_1 = np.zeros([size_z, size_x, size_y])
    # t_ini_1[1:z_max +1, 1:x_max + 1, 1:y_max + 1] = np.average(boundary_condition[0, :, :])
    t_ini_1[1:z_max + 1, 1:x_max + 1, 1:y_max + 1] = boundary_condition[0, :, :]

    # 境界条件
    # 左
    x_bou_t = 1
    t_bou_t = np.zeros([size_x, size_y])
    t_bou_t[1:x_max + 1, 1:y_max + 1] = boundary_condition[0, :, :]

    # 右
    # x_bou_b = z_max
    # t_bou_b = np.zeros([size_x, size_y])
    # t_bou_b[1:x_max+1, 1:y_max+1] = boundary_condition[0, :, :]

    # ------------------------------------------------------------------------------------------

    # 初期化
    x = np.zeros([size_z, size_x, size_y])
    a_p = np.zeros([size_z, size_x, size_y])
    a_e = np.zeros([size_z, size_x, size_y])
    a_w = np.zeros([size_z, size_x, size_y])
    a_n = np.zeros([size_z, size_x, size_y])
    a_s = np.zeros([size_z, size_x, size_y])
    a_t = np.zeros([size_z, size_x, size_y])
    a_b = np.zeros([size_z, size_x, size_y])
    b = np.zeros([size_z, size_x, size_y])

    t = np.zeros([size_z, size_x, size_y])

    a_p0 = np.zeros([size_z, size_x, size_y])
    t_p0 = np.zeros([size_z, size_x, size_y])
    t_pm1 = np.zeros([size_z, size_x, size_y])

    dx = np.zeros([size_z, size_x, size_y])

    res1 = np.zeros([size_x, size_y])
    res2 = np.zeros([size_x, size_y])
    res_w = np.zeros([size_z, size_y])
    res_e = np.zeros([size_z, size_y])
    res_n = np.zeros([size_z, size_x])
    res_s = np.zeros([size_z, size_x])

    heat_flux = np.zeros([size_x, size_y])
    # zeros = np.zeros([size_z, size_x, size_y])

    # 初期値
    t[:, :, :] = t_ini_1[:, :, :]
    t_p0[:, :, :] = t_ini_1[:, :, :]
    t_pm1[:, :, :] = t_ini_1[:, :, :]

    # t[1, :, :] = t_bou_t[:, :]
    # t[z_max, :, :] = t_bou_b[:, :]

    dt = time_step_size

    # メッシュ作成

    if expression_type == 1:
        for i in range(size_z):
            a1 = length / z_max
            x[i, :, :] = i * a1
    if expression_type == 2:
        a1 = length / (r ** z_max)
        for i in range(size_z):
            x[i, :, :] = a1 * (r ** i - 1)
    # dx[0, :, :] = (x[1, :, :] - x[0, :, :]) / 2
    # dx[n + 1, :, :] = (x[size_z, :, :] - x[size_z - 1, :, :]) / 2

    # メッシュ出力
    yield x

    # 係数
    if heat_conduction_2d:
        a_e[1:z_max + 1, 1:x_max + 1, 1:y_max + 1] = k / mpp  # m/pixel
        a_w[1:z_max + 1, 1:x_max + 1, 1:y_max + 1] = k / mpp
        a_n[1:z_max + 1, 1:x_max + 1, 1:y_max + 1] = k / mpp
        a_s[1:z_max + 1, 1:x_max + 1, 1:y_max + 1] = k / mpp

    else:
        a_e[2, 1:x_max + 1, 1:y_max + 1] = 0  # m/pixel
        a_w[2, 1:x_max + 1, 1:y_max + 1] = 0
        a_n[2, 1:x_max + 1, 1:y_max + 1] = 0
        a_s[2, 1:x_max + 1, 1:y_max + 1] = 0

    for i in range(1, z_max + 1):
        dx[i, :, :] = (x[i + 1, :, :] - x[i, :, :]) / 2 + (x[i, :, :] - x[i - 1, :, :]) / 2
        a_b[i, :, :] = k / (x[i + 1, :, :] - x[i, :, :])
        a_t[i, :, :] = k / (x[i, :, :] - x[i - 1, :, :])
        a_p0[i, :, :] = rho * c * dx[i, :, :] / dt
        a_p[i, :, :] = a_e[i, :, :] + a_w[i, :, :] + a_n[i, :, :] + a_s[i, :, :] + a_t[i, :, :] + a_b[i, :, :] \
                                                                                                + a_p0[i, :, :]
    b[:, :, :] = a_p0[:, :, :] * t_p0[:, :, :]

    # 境界条件
    # 上端
    a_e[1, 1:x_max + 1, 1:y_max + 1] = 0
    a_w[1, 1:x_max + 1, 1:y_max + 1] = 0
    a_n[1, 1:x_max + 1, 1:y_max + 1] = 0
    a_s[1, 1:x_max + 1, 1:y_max + 1] = 0
    a_t[1, 1:x_max + 1, 1:y_max + 1] = 0
    a_b[1, 1:x_max + 1, 1:y_max + 1] = 0
    a_p[1, 1:x_max + 1, 1:y_max + 1] = 1
    # b[1, 1:x_max+1, 1:y_max+1] = t[1, 1:x_max+1, 1:y_max+1]
    b[1, 1:x_max + 1, 1:y_max + 1] = t_bou_t[1:x_max + 1, 1:y_max + 1]  # x_bou_t = 1 の時用

    # 下端
    # ディリクレ
    # a_e[n, :, :] = 0
    # a_w[n, :, :] = 0
    # a_p[n, :, :] = 1
    # b[n, :, :] = t[n, :, :]
    # ノイマン
    a_e[z_max, 1:x_max + 1, 1:y_max + 1] = 0
    a_w[z_max, 1:x_max + 1, 1:y_max + 1] = 0
    a_n[z_max, 1:x_max + 1, 1:y_max + 1] = 0
    a_s[z_max, 1:x_max + 1, 1:y_max + 1] = 0
    a_t[z_max, 1:x_max + 1, 1:y_max + 1] = 1
    a_b[z_max, 1:x_max + 1, 1:y_max + 1] = 0
    a_p[z_max, 1:x_max + 1, 1:y_max + 1] = 1
    b[z_max, 1:x_max + 1, 1:y_max + 1] = 0

    # 西端
    # ノイマン
    a_e[2:z_max, 1, 1:y_max + 1] = 1
    a_w[2:z_max, 1, 1:y_max + 1] = 0
    a_n[2:z_max, 1, 1:y_max + 1] = 0
    a_s[2:z_max, 1, 1:y_max + 1] = 0
    a_t[2:z_max, 1, 1:y_max + 1] = 0
    a_b[2:z_max, 1, 1:y_max + 1] = 0
    a_p[2:z_max, 1, 1:y_max + 1] = 1
    b[2:z_max, 1, 1:y_max + 1] = 0

    # 東端
    # ノイマン
    a_e[2:z_max, x_max, 1:y_max + 1] = 0
    a_w[2:z_max, x_max, 1:y_max + 1] = 1
    a_n[2:z_max, x_max, 1:y_max + 1] = 0
    a_s[2:z_max, x_max, 1:y_max + 1] = 0
    a_t[2:z_max, x_max, 1:y_max + 1] = 0
    a_b[2:z_max, x_max, 1:y_max + 1] = 0
    a_p[2:z_max, x_max, 1:y_max + 1] = 1
    b[2:z_max, x_max, 1:y_max + 1] = 0

    # 北端
    # ノイマン
    a_e[2:z_max, 1:x_max + 1, 1] = 0
    a_w[2:z_max, 1:x_max + 1, 1] = 0
    a_n[2:z_max, 1:x_max + 1, 1] = 0
    a_s[2:z_max, 1:x_max + 1, 1] = 1
    a_t[2:z_max, 1:x_max + 1, 1] = 0
    a_b[2:z_max, 1:x_max + 1, 1] = 0
    a_p[2:z_max, 1:x_max + 1, 1] = 1
    b[2:z_max, 1:x_max + 1, 1] = 0

    # 南端
    a_e[2:z_max, 1:x_max + 1, y_max] = 0
    a_w[2:z_max, 1:x_max + 1, y_max] = 0
    a_n[2:z_max, 1:x_max + 1, y_max] = 1
    a_s[2:z_max, 1:x_max + 1, y_max] = 0
    a_t[2:z_max, 1:x_max + 1, y_max] = 0
    a_b[2:z_max, 1:x_max + 1, y_max] = 0
    a_p[2:z_max, 1:x_max + 1, y_max] = 1
    b[2:z_max, 1:x_max + 1, y_max] = 0

    print('\ninit')
    # t = solver.sor_old(size_z, size_x, size_y, t, a_p, a_e, a_w, a_n, a_s, a_t, a_b, b, zeros, t_p0, t_pm1)

    # ガウス消去法
    for step in range(number_of_time_steps):
        print('\nStep: {}/{}-----------------------------------------------'.format(str(step + 1),
                                                                                    str(number_of_time_steps)))
        t_bou_t[2:x_max, 2:y_max] = boundary_condition[step, 1:x_max - 1, 1:y_max - 1]
        b[:, :, :] = a_p0[:, :, :] * t_p0[:, :, :]

        b[1, 1:x_max + 1, 1:y_max + 1] = t_bou_t[1:x_max + 1, 1:y_max + 1]      # 上端　ディリクレ   # x_bou_t = 1 の時用
        b[z_max, 1:x_max+1, 1:y_max+1] = 0                                      # 下端　ノイマン
        b[2:z_max, 1, 1:y_max+1] = 0                                            # 西端  ノイマン
        b[2:z_max, x_max, 1:y_max+1] = 0                                        # 東端　ノイマン
        b[2:z_max, 1:x_max+1, 1] = 0                                            # 北端　ノイマン
        b[2:z_max, 1:x_max+1, y_max] = 0                                        # 南端　ノイマン

        t = solver.sor_old(size_z, size_x, size_y, t, a_p, a_e, a_w, a_n, a_s, a_t, a_b, b, a_p0, t_p0, t_pm1)

        # 条件との残差
        res1[2:x_max, 2:y_max] = t_bou_t[2:x_max, 2:y_max] - t[x_bou_t, 2:x_max, 2:y_max]
        # res2[:, :] = t_bou_r[:, :] - t[x_bou_r, :, :]
        res2[2:x_max, 2:y_max] = t[z_max - 1, 2:x_max, 2:y_max] - t[z_max, 2:x_max, 2:y_max]
        res_w[1:z_max + 1, :] = t[1:z_max + 1, 2, :] - t[1:z_max + 1, 1, :]
        res_e[1:z_max + 1, :] = t[1:z_max + 1, x_max - 1, :] - t[1:z_max + 1, x_max, :]
        res_n[1:z_max + 1, 2:x_max] = t[1:z_max + 1, 2:x_max, 2] - t[1:z_max + 1, 2:x_max, 1]
        res_s[1:z_max + 1, 2:x_max] = t[1:z_max + 1, 2:x_max, y_max - 1] - t[1:z_max + 1, 2:x_max, y_max]

        print(np.amax(np.abs(res1)), np.amax(np.abs(res2)), np.amax(np.abs(res_w)), np.amax(np.abs(res_e)),
              np.amax(np.abs(res_n)), np.amax(np.abs(res_s)))

        # 条件を満たすまで繰り返し
        while np.amax(np.abs(res1)) >= eps or np.amax(np.abs(res2)) >= eps or np.amax(np.abs(res_w)) >= eps or np.amax(
                np.abs(res_e)) >= eps or np.amax(np.abs(res_n)) >= eps or np.amax(np.abs(res_s)) >= eps:
            t[1, 2:x_max, 2:y_max] = t[1, 2:x_max, 2:y_max] + 1.2 * res1[2:x_max, 2:y_max]
            t[z_max, 2:x_max, 2:y_max] = t[z_max, 2:x_max, 2:y_max] + 1.2 * res2[2:x_max, 2:y_max]
            t[1:z_max + 1, 1, 1:y_max + 1] = t[1:z_max + 1, 1, 1:y_max + 1] + 1 * res_w[1:z_max + 1, 1:y_max + 1]
            t[1:z_max + 1, x_max, 1:y_max + 1] = t[1:z_max + 1, x_max, 1:y_max + 1] + 1 * res_e[1:z_max + 1,
                                                                                                  1:y_max + 1]
            t[1:z_max + 1, 2:x_max, 1] = t[1:z_max + 1, 2:x_max, 1] + 1 * res_n[1:z_max + 1, 2:x_max]
            t[1:z_max + 1, 2:x_max, y_max] = t[1:z_max + 1, 2:x_max, y_max] + 1 * res_s[1:z_max + 1, 2:x_max]

            t = solver.sor_old(size_z, size_x, size_y, t, a_p, a_e, a_w, a_n, a_s, a_t, a_b, b, a_p0, t_p0, t_pm1)

            # 条件との残差
            res1[2:x_max, 2:y_max] = t_bou_t[2:x_max, 2:y_max] - t[x_bou_t, 2:x_max, 2:y_max]
            # res2[:, :] = t_bou_r[:, :] - t[x_bou_r, :, :]
            res2[2:x_max, 2:y_max] = t[z_max - 1, 2:x_max, 2:y_max] - t[z_max, 2:x_max, 2:y_max]
            res_w[1:z_max + 1, :] = t[1:z_max + 1, 2, :] - t[1:z_max + 1, 1, :]
            res_e[1:z_max + 1, :] = t[1:z_max + 1, x_max - 1, :] - t[1:z_max + 1, x_max, :]
            res_n[1:z_max + 1, 2:x_max] = t[1:z_max + 1, 2:x_max, 2] - t[1:z_max + 1, 2:x_max, 1]
            res_s[1:z_max + 1, 2:x_max] = t[1:z_max + 1, 2:x_max, y_max - 1] - t[1:z_max + 1, 2:x_max, y_max]

            print(np.amax(np.abs(res1)), np.amax(np.abs(res2)), np.amax(np.abs(res_w)), np.amax(np.abs(res_e)),
                  np.amax(np.abs(res_n)), np.amax(np.abs(res_s)))

        # 温度分布更新
        t_pm1[:, :, :] = t_p0[:, :, :]
        t_p0[:, :, :] = t[:, :, :]

        heat_flux[1:x_max + 1, 1:y_max + 1] = k * (t[1, 1:x_max + 1, 1:y_max + 1] - t[2, 1:x_max + 1, 1:y_max + 1]) / (
                x[2, 1:x_max + 1, 1:y_max + 1] - x[1, 1:x_max + 1, 1:y_max + 1]) / 10000  # W/cm^2

        yield t, heat_flux

    # elapsed_time = time.time() - start
    # print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
