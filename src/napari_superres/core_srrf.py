################# SRRF ########
from skimage import io
import math
import numpy as np
from scipy.interpolate import griddata
from napari.utils import progress

from nanopyx.core.utils.timeit import timeit2
from nanopyx.methods.SRRF_workflow import SRRF


##################### Functions #####################
##### buildRing
class srrf_class:
    def __init__(self):
        self.first = "init"
    def conection_test(self):
        print("conected")

    @timeit2
    def srrf(self, img_layer, magnification, spatialRadius, fstart, fend) :
        return SRRF(img_layer.data[fstart:fend+1],
                    magnification=magnification,
                    ringRadius=spatialRadius).calculate()
