# -*- coding: utf-8 -*-

import numpy as np
#import cupy as cp

def apodImRect(in_, N):
    """
    Applies a cosine apodization to the edges of an image.

    Args:
    in_: numpy array, input image.
    N: int, size of the apodization region in pixels.

    Returns:
    A tuple of two numpy arrays: the apodized image, and the apodization mask.
    """
    Ny, Nx = in_.shape

    x = np.abs(np.linspace(-Nx/2, Nx/2, Nx))
    y = np.abs(np.linspace(-Ny/2, Ny/2, Ny))

    mapx = x > Nx/2 - N
    mapy = y > Ny/2 - N

    val = np.mean(in_)

    d = (-np.abs(x) - np.mean(-np.abs(x[mapx]))) * mapx
    d = np.interp(d, (d.min(), d.max()), (-np.pi/2, np.pi/2))
    d[~mapx] = np.pi/2
    maskx = (np.sin(d) + 1) / 2

    d = (-np.abs(y) - np.mean(-np.abs(y[mapy]))) * mapy
    d = np.interp(d, (d.min(), d.max()), (-np.pi/2, np.pi/2))
    d[~mapy] = np.pi/2
    masky = (np.sin(d) + 1) / 2

    # make it 2D
    mask = np.outer(masky, maskx)

    out = (in_ - val) * mask + val

    return out, mask

def getCorrcoef(I1, I2, c1=None, c2=None):
    if c1 is None:
        c1 = np.sqrt(np.sum(np.abs(I1)**2))
    if c2 is None:
        c2 = np.sqrt(np.sum(np.abs(I2)**2))

    dumy_mask = np.asarray(c1*c2)
    dumy_mask[dumy_mask==0] = 0.1
    cc = np.sum(np.real(I1*np.conj(I2))) / dumy_mask
    cc = np.floor(1000*cc) / 1000

    return cc

def getDcorrLocalMax(d):
    Nr = len(d)
    if Nr < 2:
        ind = [0]
        A = d[0]

    else:
        # find maxima of d
        ind = [np.argmax(d)]
        A = d[ind[0]]
        while len(d) > 1:
            if ind[0] == len(d) - 1:
                d = d[:-1]
                ind = [np.argmax(d)]
                A = d[ind[0]]
            elif ind[0] == 0:
                break
            elif A - np.min(d[ind[0]:]) >= 0.0005:
                break
            else:
                d = d[:-1]
                ind = [np.argmax(d)]
                A = d[ind[0]]
        if not ind:
            ind = [0]
            A = d[0]
        else:
            A = d[ind[0]]
            ind = [ind[-1]]


    return ind, A


def getDcorr(im, r=np.linspace(0,1,50), Ng=10, figID=0):
    """
    Estimate the image cut-off frequency based on decorrelation analysis

    Inputs:
    im: 2D image to be analyzed
    r: Fourier space sampling of the analysis (default: r = linspace(0,1,50))
    Ng: Number of high-pass filtering (default: Ng = 10)
    figID: If figID > 1, curves will be plotted in figure(figID)
           if figID == 'fast', enable fast resolution estimate mode

    Outputs:
    kcMax: Estimated cut-off frequency of the image in normalized frequency
    A0: Amplitude of the local maxima of d0
    d0: Decorrelation function before high-pass filtering
    d: All decorrelation functions
    """

    if isinstance(figID, str):
        figID = 0
        fastMode = 1
    else:
        fastMode = 0

    if len(r) < 30:
        r = np.linspace(np.min(r), np.max(r), 30)

    if Ng < 5:
        Ng = 5

    im = np.single(im)
    if im.shape[0] % 2 == 0:
        im = im[:-1, :]
    if im.shape[1] % 2 == 0:
        im = im[:, :-1]

    X, Y = np.meshgrid(np.linspace(-1, 1, im.shape[1]), np.linspace(-1, 1, im.shape[0]))

    R = np.sqrt(X**2 + Y**2)
    Nr = len(r)

    if isinstance(im, np.ndarray):
        r = np.array(r)
        R = np.array(R)

    # In : Fourier normalized image
    In = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(im)))
    In = In / np.abs(In)
    In[np.isinf(In)] = 0
    In[np.isnan(In)] = 0

    mask0 = R**2 < 1**2
    In = mask0*In  # restrict all the analysis to the region r < 1

    if figID:
        print('Computing dcorr: ')

    # Ik : Fourier transform of im
    Ik = mask0 * np.fft.fftshift(np.fft.fftn(np.fft.fftshift(im)))
    c = np.sqrt(np.sum(np.abs(Ik)**2))

    r0 = np.linspace(r[0], r[-1], Nr)
    d0 = np.empty(len(r))

    for k in range(len(r0)-1, -1, -1):

        cc = getCorrcoef(Ik, (R**2 < r0[k]**2)*In, c)
        if np.isnan(cc):
            cc = 0
        d0[k] = cc

        if fastMode == 1:
            ind0, snr0 = getDcorrLocalMax(d0[k:])
            ind0 += k-1
            if ind0 > k:
                break

    if fastMode == 0:
        ind0, snr0 = getDcorrLocalMax(d0[k:])
        snr0 = d0[ind0]

    k0 = r[ind0]

    gMax = 2/r[ind0]
    if np.isinf(gMax):
        gMax = max(im.shape[0],im.shape[1])/2

    g = np.exp(np.linspace(np.log(gMax[0]),np.log(0.15),Ng))
    g = np.insert(g, 0, im.shape[0]/4)
    d = np.zeros((Nr,2*Ng))
    kc = k0*np.ones(2*Ng+2)

    SNR = snr0*np.ones(2*Ng+2)
    if fastMode == 0:
        ind0 = 0
    else:
        if ind0 > 0:
            ind0 -= 1

    for refin in range(1,3): # two step refinement

        for h in range(len(g)):

            Ir = Ik*(1 - np.exp(-2*g[h]*g[h]*R*R)) # Fourier Gaussian filtering
            c = np.sqrt(np.sum(np.abs(Ir)**2))

            for k in range(len(r)-1, ind0-1, -1):

                #if isinstance(im, cuda.devicearray.DeviceNDArray):
                if 1 > 2:
                    cc = getCorrcoef(Ir,In*(R*R < r[k]*r[k]),c)
                    if np.isnan(cc): cc = 0
                    d[k, h + Ng*(refin-1)] = np.asarray(cc)
                else:
                    mask = (R*R < r[k]*r[k])
                    cc = getCorrcoef(Ir[mask],In[mask],c)
                    if np.isnan(cc): cc = 0
                    d[k, h + Ng*(refin-1)] = cc
                if fastMode:
                    ind, snr = getDcorrLocalMax(d[k:, h + Ng*(refin-1)])
                    ind += k
                    if ind > k: # found a local maxima, skip the calculation
                        break

            if fastMode == 0:
                ind, _ = getDcorrLocalMax(d[k:,h + Ng*(refin-1)])
                ind = ind[0]
                snr = d[ind, h + Ng*(refin-1)]
                ind += k

            kc[h + Ng*(refin-1)+1] = r[ind]
            SNR[h + Ng*(refin-1)+1] = snr


        # refining the high-pass threshold and the radius sampling
        if refin == 1:

            # high-pass filtering refinement
            indmax = np.argmax(kc)
            ind1 = indmax
            if ind1 == 0: # peak only without highpass
                ind1 = 0
                ind2 = 1
                g1 = im.shape[0]
                g2 = g[0]
            elif ind1 >= len(g)-1:
                ind2 = ind1-1
                ind1 = ind1-2
                g1 = g[ind1]; g2 = g[ind2];
            else:
                ind2 = ind1
                ind1 = ind1-1
                g1 = g[ind1];
                g2 = g[ind2];
            g = np.exp(np.linspace(np.log(g1), np.log(g2), Ng))

            # radius sampling refinement
            r1 = kc[indmax]-(r[1]-r[0])
            r2 = kc[indmax]+0.4
            if r1 < 0: r1 = 0
            if r2 > 1: r2 = 1
            r = np.linspace(r1, min(r2,r[-1]), Nr)
            ind0 = 0
            r2 = r

    if figID:
        print(' -- Computation done -- \n')

    # add d0 results to the analysis (useful for high noise images)
    kc = np.concatenate([kc, k0])
    SNR = np.concatenate([SNR, snr0])

    # need at least 0.05 of SNR to be even considered
    kc[SNR < 0.05] = 0
    SNR[SNR < 0.05] = 0

    snr = SNR

    if kc.size > 0:
        # highest resolution found
        ind = np.argmax(kc)
        kcMax = kc[ind]
        AMax = SNR[ind]
        A0 = snr0 # average image contrast has to be estimated from original image
    else:
        kcMax = r[1]
        AMax = 0
        res = r[1]
        A0 = 0

    return kcMax, A0, d0, np.nan
