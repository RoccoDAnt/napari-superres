import math

import napari.utils
import numba
import numba.cuda
import numba_progress
import numpy as np
import scipy.interpolate as interpolate


@numba.njit(parallel=True)
def cpu_max_diff(img: np.ndarray, hs: int, progress_proxy=None) -> np.ndarray:
    height, width = img.shape
    max_diff = np.zeros((height, width))
    for i in numba.prange(height):
        for j in numba.prange(width):
            max_diff[i, j] = np.max(np.abs(img[max(i - hs, 0) : i + hs + 1, max(j - hs, 0) : j + hs + 1] - img[i, j]))
        if progress_proxy is not None:
            progress_proxy.update(width)
    return max_diff


@numba.njit(parallel=True)
def cpu_mean_shift(padded: np.ndarray, hs: int, max_diff: np.ndarray, kernel: np.ndarray, progress_proxy=None) -> np.ndarray:
    height, width = padded.shape
    y = np.zeros((height - 2 * hs, width - 2 * hs), dtype=np.float64)
    for i in numba.prange(height - 2 * hs):
        for j in numba.prange(width - 2 * hs):
            window = padded[i : i + 2 * hs + 1, j : j + 2 * hs + 1]
            weights = np.exp(-(((window - padded[i + hs, j + hs]) / max_diff[i, j]) ** 2)) * kernel
            y[i, j] = (window * weights).sum() / weights.sum()
        if progress_proxy is not None:
            progress_proxy.update(width)

    return y


# These will be compiled using the first visible device.
# As compiling is expensive, we should not let the user switch device at runtime
# But the user can switch device before running the code by setting CUDA_VISIBLE_DEVICES variable
@numba.cuda.jit()
def cuda_max_diff(img: np.ndarray, hs: int, output: np.ndarray):
    i, j = numba.cuda.grid(2)
    if i >= output.shape[0]:
        return
    if j >= output.shape[1]:
        return
    value = 0
    for k in range(i - hs, i + hs + 1):
        if k < 0 or k >= img.shape[0]:
            continue
        for l in range(j - hs, j + hs + 1):
            if l < 0 or l >= img.shape[1]:
                continue
            value = max(abs(img[k, l] - img[i, j]), value)

    output[i, j] = value if value > 0 else 1


@numba.cuda.jit()
def cuda_mean_shift(padded: np.ndarray, hs: int, max_diff: np.ndarray, kernel: np.ndarray, output: np.ndarray):
    i, j = numba.cuda.grid(2)
    if i >= output.shape[0]:
        return
    if j >= output.shape[1]:
        return
    cum_value = 0
    cum_weight = 0
    for k in range(2 * hs + 1):
        for l in range(2 * hs + 1):
            weight = (padded[i + k, j + l] - padded[i + hs, j + hs]) / max_diff[i, j]
            weight = math.exp(-(weight**2)) * kernel[k, l]

            cum_weight += weight
            cum_value += padded[i + k, j + l] * weight

    output[i, j] = cum_value / cum_weight


class mssr_class:
    def __init__(self):
        self.first = "init"
    def conection_test(self):
        print("conected")

    #Bicubic Interpolation
    def bicInter(self,img, amp, mesh):
        width, height = img.shape
        y=np.linspace(1, width, width)
        x=np.linspace(1, height, height)
        imgInter=interpolate.interp2d(x, y, img, kind='cubic')
        y2=np.linspace(1, width, width*amp)
        x2=np.linspace(1, height, height*amp)
        Z2 = imgInter(x2, y2)
        if mesh:
            Z2 = self.meshing(Z2, amp)
        return Z2

    #Fourier Interpolation
    def ftInterp(self,img, amp, mesh):
        width, height = img.shape
        mdX = math.ceil(width/2) + 1
        mdY = math.ceil(height/2) + 1

        extraBorder = math.ceil(amp/2)
        Nwidth = (width*amp) + extraBorder
        Nheight = (height*amp) + extraBorder

        lnX = len(np.arange((mdX),width))
        lnY = len(np.arange((mdY),height))

        imgFt = np.fft.fft2(img)
        imgFt = imgFt * (Nwidth/width) * (Nheight/height)
    #	imgFt = imgFt * amp * amp

        fM = np.zeros((Nwidth, Nheight), dtype=complex)
        fM[0:mdX, 0:mdY] = imgFt[0:mdX, 0:mdY]; #izq sup cuadrante
        fM[0:mdX, (Nheight-lnY):Nheight] = imgFt[0:mdX, (mdY):height]; #der sup cuadrante
        fM[(Nwidth-lnX):Nwidth, 0:mdY] = imgFt[(mdX):width, 0:mdY]; #izq inf cuadrante
        fM[(Nwidth-lnX):Nwidth, (Nheight-lnY):Nheight] = imgFt[(mdX):width, (mdY):height]; #der inf cuadrante

        Z2 = (np.fft.ifft2(fM)).real
        Z2 = Z2[0:(width*amp), 0:(height*amp)]
        if mesh:
            Z2 = self.meshing(Z2, amp)
        return Z2

    #Mesh compensation
    def meshing(self,img, amp):
        width, height = img.shape
        desp = math.ceil(amp/2)
        imgPad = np.pad(img, desp, 'symmetric')
        imgS1 = imgPad[0:width, desp:height+desp]
        imgS2 = imgPad[(desp*2):width+(desp*2), desp:height+desp]
        imgS3 = imgPad[desp:width+desp, 0:height]
        imgS4 = imgPad[desp:width+desp, (desp*2):height+(desp*2)]
        imgF = (img + imgS1 + imgS2 + imgS3 + imgS4) / 5
        return imgF


    # Spatial MSSR
    def sfMSSR(self, img, fwhm, amp, order, mesh=True, ftI=False, intNorm=True, device="cuda"):
        napari_progress = napari.utils.progress(total=10)
        napari_progress.set_description("Interpolate")
        assert device in ("cpu", "cuda")

        if device == "cuda" and not numba.cuda.is_available():
            print("GPU not supported on this system, switch to cpu")
            device = "cpu"

        if not np.issubdtype(img.dtype, np.floating):
            img = img.astype(np.float64)  # Convert any int into float64

        hs = round(0.5 * fwhm * amp)
        if hs < 1:
            hs = 1

        if amp > 1 and not ftI:
            img = self.bicInter(img, amp, mesh)
        elif amp > 1 and ftI:
            img = self.ftInterp(img, amp, mesh)

        napari_progress.update(2)
        napari_progress.set_description("Max diff")

        i = np.arange(-hs, hs + 1)
        j = np.arange(-hs, hs + 1)
        kernel = np.exp(-(i[None] ** 2 + j[:, None] ** 2) / hs**2)
        kernel[hs, hs] = 0

        if device == "cpu":
            with numba_progress.ProgressBar(total=img.size, desc="Max Diff", leave=False) as progress_proxy:
                max_diff = cpu_max_diff(img, hs, progress_proxy)
                max_diff[max_diff == 0] = 1  # Prevent 0 division

            napari_progress.update(3)
            napari_progress.set_description("Mean Shift")
            with numba_progress.ProgressBar(total=img.size, desc="Mean Shift", leave=False) as progress_proxy:
                MS = img - cpu_mean_shift(np.pad(img, hs, "symmetric"), hs, max_diff, kernel, progress_proxy)
            napari_progress.update(5)
            napari_progress.set_description("MSSR")
        else:
            max_tpb = numba.cuda.get_current_device().MAX_THREADS_PER_BLOCK
            max_tpb = 20
            tpb = int(math.sqrt(max_tpb))
            blocks = (math.ceil(img.shape[0] / tpb), math.ceil(img.shape[1] / tpb))

            max_diff = numba.cuda.device_array_like(img)
            output = numba.cuda.device_array_like(img)
            cuda_max_diff[blocks, (tpb, tpb)](numba.cuda.to_device(img), hs, max_diff)

            napari_progress.update(3)
            napari_progress.set_description("Mean Shift")
            cuda_mean_shift[blocks, (tpb, tpb)](
                numba.cuda.to_device(np.pad(img, hs, "symmetric")),
                hs,
                max_diff,
                numba.cuda.to_device(kernel),
                output,
            )
            MS = img - output
            napari_progress.update(5)
            napari_progress.set_description("MSSR")

        MS[MS < 0] = 0
        I3 = MS / MS.max()
        x3 = img / img.max()
        for i in range(order):
            I4 = x3 - I3
            I5 = I4.max() - I4
            I5 = I5 / I5.max()
            I6 = I5 * I3
            I7 = I6 / I6.max()
            x3 = I3
            I3 = I7
        I3[np.isnan(I3)] = 0
        if intNorm:
            IMSSR = I3 * img
        else:
            IMSSR = I3

        napari_progress.close()
        return IMSSR

    #Temporal MSSR
    def tMSSR(self,img_layer, fwhm, amp, order, mesh = True, ftI = False, intNorm = True, device="cuda"):
        img=np.array(img_layer.data)
        nFrames, width, height = img.shape
        imgMSSR = np.zeros((nFrames,width*amp,height*amp))
        for nI in range(nFrames):
            print("Image " + str(nI+1))
            imgMSSR[nI, :, :] = self.sfMSSR(img[nI], fwhm, amp, order, mesh, ftI, intNorm, device)
        return imgMSSR

    #Mean
    def tMean(self,img):
        return np.mean(img, 0)

    #Variance
    def tVar(self,img):
        return np.var(img, 0)

    #Temporal Product Mean (TPM)
    def TPM(self,img):
        nFrames, width, height = img.shape
        SumTPM = np.zeros((width,height))
        iTPM = np.zeros((width,height))
        for i in range(nFrames):
            SumTPM = SumTPM + img[i]
        for i in range(nFrames):
            iTPM = iTPM + (SumTPM * img[i])
        return iTPM

    #Auto-Cummulants (SOFI)
    def TRAC(self,img, k):
        nFrames, width, height = img.shape
        avg = np.mean(img, 0)
        d0 = np.absolute(img - avg)
        d1 = d0[1:nFrames, :, :]
        d2 = d0[2:nFrames, :, :]
        d3 = d0[3:nFrames, :, :]
        if k == 2:
            trac = np.mean(d0[1:nFrames, :, :]*d1,0)
        elif k == 3:
            trac = np.mean(d0[2:nFrames, :, :]*d1[1:nFrames, :, :]*d2,0)
        else:
            t1 = np.mean(d0[3:nFrames, :, :]*d1[2:nFrames, :, :]*d2[1:nFrames, :, :]*d3,0)
            t2 = np.mean(d0[3:nFrames, :, :]*d1[2:nFrames, :, :],0)*np.mean(d2[1:nFrames, :, :]*d3,0)
            t3 = np.mean(d0[3:nFrames, :, :]*d2[1:nFrames, :, :],0)*np.mean(d1[2:nFrames, :, :]*d3,0)
            t4 = np.mean(d0[3:nFrames, :, :]*d3,0)*np.mean(d1[2:nFrames, :, :]*d2[1:nFrames, :, :],0)
            trac = np.absolute(t1-t2-t3-t4)
        return trac

    #Variation Coefficient
    #standard deviation is replaced by variance
    def varCoef(self,img):
        my_mean = np.mean(img, 0)
        my_mean[my_mean==0] = 1
        my_std = np.var(img,0)
        return np.divide(my_std,my_mean)

    #Empirical cumulative distribution function
    def ecdf(self,data):
        """ Compute ECDF """
        data = np.reshape(data, -1)
        x = np.sort(data)
        n = len(x)
        y = np.arange(1, n+1) / n
        return(x,y)

    #Exclude Outliers
    def excOutliers(self,data, th):
        th = (100 - th)/100
        x, f = self.ecdf(data)
        found = np.where(f > th)
        mnX = x[found[0][0]]
        data = np.where(data >= mnX, mnX, data)
        return data
