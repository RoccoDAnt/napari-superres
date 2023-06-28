import numpy as np
from scipy import signal
import sys
import math
from napari.utils import progress
from scipy.interpolate import griddata
from .finterp import *
from .deconvsk import *


class sofi_class:
    def __init__(self):
        # self.filename = filename
        # self.filepath = filepath
        self.stack = []
        self.ave = None
        self.finterp_factor = 1
        self.morder_lst = []
        self.morder_finterp_lst = []
        self.moments_set = {}
        self.moments_set_bc = {}
        self.moments_finterp_set = {}
        self.cumulants_set = {}
        self.cumulants_set_bc = {}
        self.morder = 0
        self.corder = 0
        self.fbc = 1
        #self.n_frames, self.xdim, self.ydim =

    def conection_test(self):
      print("conected")

    def moment_image(self, imstack, order=6, mean_im=None, mvlength=[],
                     finterp=False, interp_num=1):

        if finterp is False:
            if order in self.morder_lst:
                print("this order has been calculated")
                print('\n')
            else:
                moment_im = self.calc_moment_im(imstack, order, mvlength)
                self.morder_lst.append(order)
                self.moments_set[order] = moment_im
                return self.moments_set[order]
        else:
            if self.finterp_factor != 1 and interp_num != self.finterp_factor:
                print('Different interpolation factor calculating ...')
                print('\n')
            else:
                if order in self.morder_finterp_lst:
                    print("this order has been calculated")
                    print('\n')
            moment_im = self.moment_im_with_finterp(imstack,
                                                              order, interp_num,
                                                              mvlength)
            #print(np.shape(moment_im))
            self.morder_finterp_lst.append(order)
            self.moments_finterp_set[order] = moment_im
            self.finterp_factor = interp_num
            return self.moments_finterp_set[order]

    def calc_moment_im(self, stack, order, frames=[], mean_im=None):

        if mean_im is None:
            mean_im = np.mean(stack, axis=0)
        imstack = stack
        #xdim, ydim = np.shape(imstack.pages[0])
        xdim, ydim = imstack.shape[1:3]
        moment_im = np.zeros((xdim, ydim))
        print('Calculating the %s-order moment ...' %
              order)
        if frames:
            for frame_num in range(frames[0], frames[1]):
                #im = tiff.imread(filepath + '/' + filename, key=frame_num)
                im = imstack[frame_num,:,:]
                moment_im = moment_im + (im - mean_im)**order
                sys.stdout.write('\r')
                sys.stdout.write("[{:{}}] {:.1f}%".format(
                    "="*int(30/(frames[1]-frames[0])*(frame_num-frames[0]+1)), 29,
                    (100/(frames[1]-frames[0])*(frame_num-frames[0]+1))))
                sys.stdout.flush()
            moment_im = moment_im / (frames[1] - frames[0])
        else:
            #mvlength = len(imstack.pages)
            mvlength = imstack.shape[0]
            for frame_num in range(mvlength):
                im = imstack[frame_num]
                moment_im = moment_im + (im - mean_im)**order
                sys.stdout.write('\r')
                sys.stdout.write("[{:{}}] {:.1f}%".format(
                    "="*int(30/mvlength*(frame_num+1)), 29,
                    (100/mvlength*(frame_num+1))))
                sys.stdout.flush()
            moment_im = moment_im / mvlength
        print('\n')
        return moment_im


    def moment_im_with_finterp(self, stack, order, interp_num,
                               frames=[], mean_im=None):

        if mean_im is None:
            mean_im = self.average_image_with_finterp(stack, interp_num)

        imstack = stack
        xdim, ydim = imstack.shape[1:3]
        moment_im = np.zeros(((xdim-1)*interp_num+1, (ydim-1)*interp_num+1))
        if frames:
            for frame_num in progress(range(frames[0], frames[1])):
                im = imstack[frame_num,:,:]
                interp_im = fourier_interp_array(im, [interp_num])[0]
                moment_im = moment_im + (interp_im - mean_im)**order
                sys.stdout.write('\r')
                sys.stdout.write("[{:{}}] {:.1f}%".format(
                    "="*int(30/(frames[1]-frames[0])*(frame_num-frames[0]+1)), 29,
                    (100/(frames[1]-frames[0])*(frame_num-frames[0]+1))))
                sys.stdout.flush()
            moment_im = np.int64(moment_im / (frames[1] - frames[0]))
        else:
            #mvlength = len(imstack.pages)
            mvlength = imstack.shape[0]
            for frame_num in progress(range(mvlength)):
                #im = tiff.imread(filepath + '/' + filename, key=frame_num)
                im = imstack[frame_num,:,:]
                interp_im = fourier_interp_array(im, [interp_num])[0]
                moment_im = moment_im + (interp_im - mean_im)**order
                sys.stdout.write('\r')
                sys.stdout.write("[{:{}}] {:.1f}%".format(
                    "="*int(30/mvlength*(frame_num+1)), 29,
                    (100/mvlength*(frame_num+1))))
                sys.stdout.flush()
            moment_im = np.int64(moment_im / mvlength)
        return moment_im

    def average_image_with_finterp(self, stack, interp_num):

        original_mean_im = np.mean(stack, axis=0)
        finterp_mean_im = fourier_interp_array(original_mean_im, [interp_num])

        return finterp_mean_im[0]

    def gauss2d_mask(self, shape=(3, 3), sigma=0.5):

        mdim, ndim = [(pixel-1) / 2 for pixel in shape]
        y, x = np.ogrid[-mdim:(mdim + 1), -ndim:(ndim + 1)]
        h = np.exp(-(x*x + y*y) / (2*sigma*sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h


    def gauss1d_mask(self, shape=(1, 3), sigma=0.5):
        """Generate a 1D gaussian mask."""
        return self.gauss2d_mask(shape, sigma)[0]

    def deconvsk(self, est_psf, input_im, deconv_lambda, deconv_iter):
        xdim, ydim = np.shape(input_im)
        deconv_im = np.append(np.append(input_im, np.fliplr(input_im), axis=1),
            np.append(np.flipud(input_im), np.rot90(input_im, 2), axis=1), axis=0)
        # Perform mirror extension to the image in order sto surpress ringing
        # artifacts associated with fourier transform due to truncation effect.
        psf0 = est_psf / np.max(est_psf)
        for iter_num in progress(range(deconv_iter)):
            alpha = deconv_lambda**(iter_num+1) / (deconv_lambda - 1)
            deconv_psf, deconv_im = richardson_lucy(deconv_im, psf0**alpha, 1)
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" % ('='*iter_num,
                             100/(deconv_iter-1)*iter_num))
            sys.stdout.flush()

        deconv_im = deconv_im[0:xdim, 0:ydim]
        return deconv_im

    def ldrc(self, mask_im, input_im, order=1, window_size=[25, 25]):
        xdim_mask, ydim_mask = np.shape(mask_im)
        xdim, ydim = np.shape(input_im)
        if xdim == xdim_mask and ydim == ydim_mask:
            mask = mask_im
        else:
            # Resize mask to the image dimension if not the same dimension
            mod_xdim = (xdim_mask-1)*order + 1    # new mask x dimemsion
            mod_ydim = (ydim_mask-1)*order + 1    # new mask y dimemsion
            px = np.arange(0, mod_xdim, order)
            py = np.arange(0, mod_ydim, order)

            # Create coordinate list for interpolation
            coor_lst = []
            for i in px:
                for j in py:
                    coor_lst.append([i, j])
            coor_lst = np.array(coor_lst)

            orderjx = complex(str(mod_xdim) + 'j')
            orderjy = complex(str(mod_ydim) + 'j')
            # New coordinates for interpolated mask
            px_new, py_new = np.mgrid[0:mod_xdim-1:orderjx, 0:mod_ydim-1:orderjy]

            interp_mask = griddata(coor_lst, mask_im.reshape(-1, 1),
                                   (px_new, py_new), method='cubic')
            mask = interp_mask.reshape(px_new.shape)

        seq_map = np.zeros((xdim, ydim))
        ldrc_im = np.zeros((xdim, ydim))
        for i in progress(range(xdim - window_size[0] + 1)):
            for j in range(ydim - window_size[1] + 1):
                window = input_im[i:i+window_size[0], j:j+window_size[1]]
                norm_window = (window - np.min(window)) / \
                              (np.max(window) - np.min(window))
                # norm_window = window / np.max(window)
                ldrc_im[i:i+window_size[0], j:j+window_size[1]] = \
                    ldrc_im[i:i+window_size[0], j:j+window_size[1]] + \
                    norm_window * \
                    np.max(mask[i:i+window_size[0], j:j+window_size[1]])
                seq_map[i:i+window_size[0], j:j+window_size[1]] = \
                    seq_map[i:i+window_size[0], j:j+window_size[1]] + 1
            sys.stdout.write('\r')
            sys.stdout.write("[%-20s] %d%%" %
                             ('='*int(20*(i+1)/(xdim - window_size[0] + 1)),
                              100*(i+1)/(xdim - window_size[0] + 1)))
            sys.stdout.flush()
        ldrc_im = ldrc_im / seq_map
        return ldrc_im
