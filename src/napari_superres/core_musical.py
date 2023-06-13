# -*- coding: utf-8 -*-
import time
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.special import j1, gamma
from skimage import io
from scipy.stats import multivariate_normal
from napari.utils import progress

class musical_class:
    def __init__(self):
        self.first = "init"
    def conection_test(self):
        print("conected")

    def compute_PSF(self, lam, NA, mag, PixelSize, test_per_pixel):
        N_w=1+2*np.floor(0.61*lam*mag/(NA*PixelSize))
        self.N=N_w**2
        pixel_dim=(PixelSize/1000)/mag #micron
        subpixel_dim=1000*pixel_dim/test_per_pixel
        lam=lam/1000 #micron
        vec_r_ccd, vec_r_foc, x_foc, y_foc = self.generate_coordinates(pixel_dim,N_w,test_per_pixel)
        G_PSF = self.generate_PSF_airy(lam,NA,vec_r_foc,vec_r_ccd)
        return G_PSF, x_foc, y_foc, self.N, subpixel_dim

    def generate_coordinates(self, pixel_dim, N_w, test_per_pixel):
        lateral_view_area = pixel_dim * N_w
        N_x_foc = test_per_pixel * N_w
        x_foc = np.linspace(-0.5 * lateral_view_area, 0.5 * lateral_view_area, int(N_x_foc + 1))
        x_foc = (x_foc[:-1] + x_foc[1:]) / 2.
        y_foc = x_foc
        x_sensor = np.linspace(-1, 1, int(N_w))
        x_sensor = pixel_dim * x_sensor / (x_sensor[1] - x_sensor[0])
        y_sensor = x_sensor
        xx, yy = np.meshgrid(x_sensor, y_sensor)
        phi, r = np.arctan2(-xx, -yy), np.sqrt(xx ** 2 + yy ** 2)
        vec_r_ccd = np.column_stack((r.ravel(), phi.ravel()))
        xx, yy = np.meshgrid(x_foc, y_foc)
        phi, r = np.arctan2(xx, yy), np.sqrt(xx ** 2 + yy ** 2)
        vec_r_foc = np.column_stack((r.ravel(), phi.ravel()))
        return vec_r_ccd, vec_r_foc, x_foc, y_foc


    def generate_PSF_airy(self, lam, NA, vec_r_foc, vec_r_ccd):
        k = 2 * np.pi / lam
        x_foc, y_foc = np.cos(vec_r_foc[:, 1]) * vec_r_foc[:, 0], np.sin(vec_r_foc[:, 1]) * vec_r_foc[:, 0]
        x_ccd, y_ccd = np.cos(vec_r_ccd[:, 1]) * vec_r_ccd[:, 0], np.sin(vec_r_ccd[:, 1]) * vec_r_ccd[:, 0]
        X_ccd, X_foc = np.meshgrid(x_ccd, x_foc)
        Y_ccd, Y_foc = np.meshgrid(y_ccd, y_foc)
        x, y = X_ccd - X_foc, Y_ccd - Y_foc
        rho = np.sqrt(x ** 2 + y ** 2)
        rho_ = NA * rho
        I_out = (j1(k * rho_) / (k * rho_)) ** 2
        I_out[np.isnan(I_out)] = (0.5 / gamma(2)) ** 2
        G_PSF = I_out.T
        return G_PSF


    def function_SVD(self, Data_CCD_illum, window_weights):
        window_weights = np.diag(window_weights)
        Data = window_weights.dot(Data_CCD_illum)
        s  = np.linalg.svd(Data, full_matrices=False, compute_uv=False)
        return s

    def function_SVD_scan_parallel(self, imageStack, x_val, y_val, N_w, G_PSF, x_foc, y_foc, test_points_per_pixel):
        # data window pixels
        data_pixels = int(math.sqrt(G_PSF.shape[0]))
        ceil_half_data_window = math.ceil(math.sqrt(G_PSF.shape[0]) / 2)
        data_window_pixels = np.array(range(int(ceil_half_data_window - N_w // 2), int(ceil_half_data_window + N_w // 2 + 1)))
        length_data_window_pixels = len(data_window_pixels)

        # test interpolation window pixels
        test_points = int(math.sqrt(G_PSF.shape[1]))
        ceil_half_test_window = math.ceil(math.sqrt(G_PSF.shape[1]) / (2 * test_points_per_pixel))
        test_window_pixels = np.array(range(int(test_points_per_pixel * (ceil_half_test_window - np.floor(N_w / 2) - 1) + 1), int(1 + test_points_per_pixel * (ceil_half_test_window + np.floor(N_w / 2)))))
        length_test_window_pixels = len(test_window_pixels)

        # Reducing PSF to desired data and test point selections
        G_PSF = G_PSF.reshape(data_pixels, data_pixels, test_points, test_points)
        G_PSF = G_PSF.reshape(length_data_window_pixels**2, length_test_window_pixels**2)

        x_foc = x_foc[test_window_pixels-1]
        y_foc = y_foc[test_window_pixels-1]


        # Making filter pad
        floor_half_data_window = int(np.floor(np.sqrt(G_PSF.shape[0]) / 2))
        rho = floor_half_data_window
        data_window = int(np.sqrt(G_PSF.shape[0]))
        x_temp, y_temp = np.meshgrid(np.arange(1, data_window + 1), np.arange(1, data_window + 1))
        Gaussian_pad = multivariate_normal.pdf(np.column_stack((x_temp.ravel(), y_temp.ravel())), mean= np.array(floor_half_data_window) + [1, 1], cov=[rho, rho])
        Gaussian_pad = Gaussian_pad / np.max(Gaussian_pad)
        S_matrix = np.zeros((int(N_w**2), len(y_val), len(x_val)))
        numberOfImages = imageStack.shape[0]

        for y_ind in progress(range(len(y_val))):
            y = y_val[y_ind]
            S = np.zeros((int(N_w**2), len(x_val)))

            for x_ind in range(len(x_val)):
                x = x_val[x_ind]

                y_min = max(y - floor_half_data_window, 1)
                y_max = min(y + floor_half_data_window, imageStack.shape[1])
                x_min = max(x - floor_half_data_window, 1)
                x_max = min(x + floor_half_data_window, imageStack.shape[2])
                im = imageStack[:,y_min-1:y_max, x_min-1:x_max]

                im_mat = np.transpose(im, (1, 2, 0))
                im_ = im_mat.reshape(-1, numberOfImages, order="F")

                Py_min = max(1, (floor_half_data_window + 1) - (y - 1))
                Py_max = min(2 * floor_half_data_window + 1, (floor_half_data_window + 1) + (imageStack.shape[1] - y))
                Px_min = max(1, (floor_half_data_window + 1) - (x - 1))
                Px_max = min(2 * floor_half_data_window + 1, (floor_half_data_window + 1) + (imageStack.shape[2] - x))

                Gaussian_pad_ = Gaussian_pad.reshape(data_window, data_window)
                Gaussian_pad_ = Gaussian_pad_[Py_min-1:Py_max, Px_min-1:Px_max]
                data_window_weights = Gaussian_pad_.ravel()

                s = self.function_SVD(im_, data_window_weights)

                if len(s) < N_w**2:
                    s = np.concatenate((s, np.zeros(int(N_w**2) - len(s))))
                S[:, x_ind] = s

            S_matrix[:, y_ind, :] = S

        S_matrix = S_matrix.reshape(S_matrix.shape[0], -1)

        return S_matrix


    def function_MUSIC_scan_parallel(self, imageStack, x_val, y_val, SNR_cutoff, N_w, G_PSF, x_foc, y_foc, test_points_per_pixel, alpha):
        # data window pixels
        data_pixels = int(math.sqrt(G_PSF.shape[0]))
        ceil_half_data_window = math.ceil(math.sqrt(G_PSF.shape[0]) / 2)
        data_window_pixels = np.array(range(int(ceil_half_data_window - N_w // 2), int(ceil_half_data_window + N_w // 2 + 1)))
        length_data_window_pixels = len(data_window_pixels)
        numberOfImages, y_in, x_in = imageStack.shape

        # test interpolation window pixels
        test_points = int(math.sqrt(G_PSF.shape[1]))
        ceil_half_test_window = math.ceil(math.sqrt(G_PSF.shape[1]) / (2 * test_points_per_pixel))
        test_window_pixels = np.array(range(int(test_points_per_pixel * (ceil_half_test_window - np.floor(N_w / 2) - 1) + 1), int(1 + test_points_per_pixel * (ceil_half_test_window + np.floor(N_w / 2)))))
        length_test_window_pixels = len(test_window_pixels)

        # Reducing PSF to desired data and test point selections

        G_PSF = G_PSF.reshape(data_pixels, data_pixels, test_points, test_points, order="F")
        G_PSF = G_PSF.reshape(length_data_window_pixels**2, length_test_window_pixels**2, order="F")

        x_foc = x_foc[test_window_pixels-1]
        y_foc = y_foc[test_window_pixels-1]

        # Making filter pad
        floor_half_data_window = int(np.floor(np.sqrt(G_PSF.shape[0]) / 2))
        rho = floor_half_data_window
        data_window = int(np.sqrt(G_PSF.shape[0]))
        x_temp, y_temp = np.meshgrid(np.arange(1, data_window + 1), np.arange(1, data_window + 1))
        Gaussian_pad = multivariate_normal.pdf(np.column_stack((x_temp.ravel(), y_temp.ravel())), mean= np.array(floor_half_data_window) + [1, 1], cov=[rho, rho])
        Gaussian_pad = Gaussian_pad / np.max(Gaussian_pad)

        #used later in interleaving the test windows:
        floor_half_interpol_window=np.floor(N_w/2);
        ceil_half_interpol_window=np.ceil(N_w/2);

        numberOfImages = imageStack.shape[0]

        MUSIC = np.zeros((len(y_val) * test_points_per_pixel, len(x_val) * test_points_per_pixel))
        MUSIC_factor = np.zeros((len(y_val) * test_points_per_pixel, len(x_val) * test_points_per_pixel))

        M = np.zeros((len(y_val), len(x_val)))

        for y_ind in progress(range(len(y_val))):
            y = y_val[y_ind]
            S = np.zeros((int(N_w**2), len(x_val)))
            Mx = np.zeros(len(x_val))

            for x_ind in range(len(x_val)):
                x = x_val[x_ind]
                y_min = max(y - floor_half_data_window, 1)
                y_max = min(y + floor_half_data_window, imageStack.shape[1])
                x_min = max(x - floor_half_data_window, 1)
                x_max = min(x + floor_half_data_window, imageStack.shape[2])
                im = imageStack[:,y_min-1:y_max, x_min-1:x_max]

                im_mat = np.transpose(im, (1, 2, 0))
                im_ = im_mat.reshape(-1, numberOfImages, order="F")


                Py_min = max(1, (floor_half_data_window + 1) - (y - 1))
                Py_max = min(2 * floor_half_data_window + 1, (floor_half_data_window + 1) + (imageStack.shape[1] - y))
                Px_min = max(1, (floor_half_data_window + 1) - (x - 1))
                Px_max = min(2 * floor_half_data_window + 1, (floor_half_data_window + 1) + (imageStack.shape[2] - x))



                G_PSF_ = G_PSF.reshape(data_window, data_window, -1, order="F")
                G_PSF_f = G_PSF_[Py_min-1:Py_max, Px_min-1:Px_max, :].reshape(-1, G_PSF_.shape[2], order="F")

                Gaussian_pad_ = Gaussian_pad.reshape(data_window, data_window, order="F")
                Gaussian_pad_ = Gaussian_pad_[Py_min-1:Py_max, Px_min-1:Px_max]
                data_window_weights = Gaussian_pad_.reshape(-1, order="F")

                Pseudospectrum, Mx[x_ind] = self.function_MUSIC(im_, G_PSF_f, data_window_weights, SNR_cutoff, alpha)

                current = np.flipud(np.fliplr(Pseudospectrum.reshape(len(y_foc), len(x_foc), order="F")))

                x_ind_result = np.arange(len(x_foc)) - (floor_half_interpol_window * test_points_per_pixel) + ((x_ind) * test_points_per_pixel)
                y_ind_result = np.arange(len(y_foc)) - (floor_half_interpol_window * test_points_per_pixel) + ((y_ind) * test_points_per_pixel)

                x_ind_result = x_ind_result.astype(int)
                y_ind_result = y_ind_result.astype(int)

                keep_x = (x_ind_result >= 0) & (x_ind_result < MUSIC.shape[1])
                keep_y = (y_ind_result >= 0) & (y_ind_result < MUSIC.shape[0])

                a = np.where(keep_y == True)[0]
                b = np.where(keep_x == True)[0]

                MUSIC[y_ind_result[keep_y][0]:y_ind_result[keep_y][-1]+1, x_ind_result[keep_x][0]:x_ind_result[keep_x][-1]+1] += current[a[0]:a[-1]+1,b[0]:b[-1]+1]
                MUSIC_factor[y_ind_result[keep_y][0]:y_ind_result[keep_y][-1]+1, x_ind_result[keep_x][0]:x_ind_result[keep_x][-1]+1] += 1

                M[y_ind, :] = Mx
        MUSIC_out = np.divide(MUSIC, MUSIC_factor)

        minColorBarPercent = 1
        maxColorBarPercent = 99.99
        hh, bb = np.histogram(MUSIC_out.flatten(), bins=10000)
        hh = np.cumsum(hh)
        hh = hh / np.max(hh)
        max_ind = np.argmax(hh > maxColorBarPercent / 100)
        min_ind = np.argmin(hh < minColorBarPercent / 100)

        if min_ind is None:
            min_ind = 0

        minColorBarValue = max(0, bb[min_ind])
        maxColorBarValue = min(bb[-1], bb[max_ind])

        MUSICAL_use = MUSIC_out.copy()
        MUSICAL_use[MUSICAL_use < minColorBarValue] = minColorBarValue
        MUSICAL_use[MUSICAL_use > maxColorBarValue] = maxColorBarValue
        MUSICAL_use = MUSICAL_use - minColorBarValue
        MUSICAL_use = MUSICAL_use / np.max(MUSICAL_use)

        minRegion = np.sqrt(self.N) + ((np.sqrt(self.N) - 1) / 2.)
        if min(x_in - 1, y_in - 1) > minRegion:
            Exclude = (np.sqrt(self.N) - 1) / 2.
        else:
            Exclude = 0

        ExcludeSubPixelLevel = int(test_points_per_pixel * Exclude)
        MUSICAL_use = MUSICAL_use[(1 + ExcludeSubPixelLevel):-(ExcludeSubPixelLevel + 1),(1 + ExcludeSubPixelLevel):-(ExcludeSubPixelLevel + 1)]

        MUSICAL_use = MUSICAL_use / np.max(MUSICAL_use)

        return MUSICAL_use, M



    def function_MUSIC(self, Data_CCD_illum, Mapping_CCD_test_pt, window_weights, SNR_cutoff, alpha):
        if window_weights.size == 0:
            window_weights = np.eye(Data_CCD_illum.shape[0])
        elif np.min(window_weights.shape):
            window_weights = np.diag(window_weights)

        Mapping_CCD_test_pt = np.dot(window_weights, Mapping_CCD_test_pt) / np.max(Mapping_CCD_test_pt)
        Data = np.dot(window_weights, Data_CCD_illum)

        u, s, _ = np.linalg.svd(Data, full_matrices=False)

        M = np.where(np.log10(s) < SNR_cutoff)[0][0] + 1

        if M is None:
            M = 0

        if M >= min(Data.shape) or M <= 0:
            M = min(Data.shape) - 3


        u_red = u[:, :M]
        u_orth = -1*u[:, M:]
        MM = np.dot(u_orth.T, Mapping_CCD_test_pt)
        PP = np.dot(u_red.T, Mapping_CCD_test_pt)
        d_PN = np.sqrt(np.sum(np.abs(MM) ** 2, axis=0))
        d_PS = np.sqrt(np.sum(np.abs(PP) ** 2, axis=0))
        Pseudospectrum = (d_PS / d_PN) ** alpha

        return Pseudospectrum, M



# imageStack = io.imread("CardioMyoblast_Mitochondria_100Frames_NA1.4_80nm_1x_510nmemission.tif").astype(float)
# imageStack = imageStack / np.max(imageStack)
# numberOfImages, y_in, x_in = imageStack.shape
#
#
# lam = 510
# NA = 1.4
# M = 100
# PixelSize = 8000
# TestPointsPerPixel = 20
#
# G_PSF, x_foc, y_foc, N, subpixel_size = compute_PSF(lam, NA, M, PixelSize, TestPointsPerPixel)
#
# x_val = list(range(1, x_in + 1))
# y_val = list(range(1, y_in + 1))
#
# Threshold = -0.5
# N_w = np.sqrt(N)
# Alpha = 4
#
# # S_matrix = function_SVD_scan_parallel(imageStack, x_val, y_val, N_w, G_PSF, x_foc, y_foc, TestPointsPerPixel)
# #
# # plt.plot(np.log10(S_matrix))
# # plt.show()
# musical_im, _ = function_MUSIC_scan_parallel(imageStack, x_val, y_val, Threshold, N_w, G_PSF, x_foc, y_foc, TestPointsPerPixel, Alpha)
# # plt.imshow(musical_im)
# # plt.show()
#
# minColorBarPercent = 1
# maxColorBarPercent = 99.99
# hh, bb = np.histogram(musical_im.flatten(), bins=10000)
# hh = np.cumsum(hh)
# hh = hh / np.max(hh)
# max_ind = np.argmax(hh > maxColorBarPercent / 100)
# min_ind = np.argmin(hh < minColorBarPercent / 100)
#
# if min_ind is None:
#     min_ind = 0
#
# minColorBarValue = max(0, bb[min_ind])
# maxColorBarValue = min(bb[-1], bb[max_ind])
#
# MUSICAL_use = musical_im.copy()
# MUSICAL_use[MUSICAL_use < minColorBarValue] = minColorBarValue
# MUSICAL_use[MUSICAL_use > maxColorBarValue] = maxColorBarValue
# MUSICAL_use = MUSICAL_use - minColorBarValue
# MUSICAL_use = MUSICAL_use / np.max(MUSICAL_use)
#
#
# # plt.imshow(MUSICAL_use)
# # plt.show()
#
# minRegion = np.sqrt(N) + ((np.sqrt(N) - 1) / 2.)
# if min(x_in - 1, y_in - 1) > minRegion:
#     Exclude = (np.sqrt(N) - 1) / 2.
# else:
#     Exclude = 0
#
# ExcludeSubPixelLevel = int(TestPointsPerPixel * Exclude)
# MUSICAL_use = MUSICAL_use[(1 + ExcludeSubPixelLevel):-(ExcludeSubPixelLevel + 1),(1 + ExcludeSubPixelLevel):-(ExcludeSubPixelLevel + 1)]
#
# MUSICAL_use = MUSICAL_use / np.max(MUSICAL_use)
#
# plt.imshow(MUSICAL_use, cmap="gray")
# plt.colorbar()
# plt.show()
