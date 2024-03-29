################# SRRF ########
import numpy as np
from nanopyx.methods import SRRF
from nanopyx.core.transform.sr_temporal_correlations import calculate_SRRF_temporal_correlations


##################### Functions #####################
##### buildRing
class srrf_class:
    def __init__(self):
        self.first = "init"
    def conection_test(self):
        print("conected")

    def srrf(self, img_layer, magnification, spatialRadius, fstart, fend):
        img = np.array(img_layer.data)
        n, w, h, = img.shape
        irm = np.zeros((fend - fstart, w*magnification, h*magnification))
        irm = SRRF(
            img[fstart:fend+1],
            magnification=magnification,
            ringRadius=spatialRadius
            )
        irm = np.array(irm[0])
        return irm
