"""
This module is an example of a barebones function plugin for napari

It implements the ``napari_experimental_provide_function`` hook specification.
see: https://napari.org/docs/dev/plugins/hook_specifications.html

Replace code below according to your needs.
"""
from napari.layers import Image, Labels, Layer, Points
import napari.types
from typing import TYPE_CHECKING

from enum import Enum
import numpy as np
from napari_plugin_engine import napari_hook_implementation

if TYPE_CHECKING:
    import napari

from .MSSR import MSSR, TMSSR #check if this is importing
from .srrf import srrf  #check if this is importing
from .ESI import ESI_Analysis



# This is the actual plugin function, where we export our function
# (The functions themselves are defined below)
@napari_hook_implementation
def napari_experimental_provide_function():
    # we can return a single function
    # or a tuple of (function, magicgui_options)
    # or a list of multiple functions with or without options, as shown here:
    return [srrf_module, mssr_module, esi_module]

def srrf_module(viewer: 'napari.Viewer', layer: Image, magnification: int = 4, spatial_radius: int = 5, symmetryAxis: int = 6, fstart: int = 0, fend: int = 100)-> napari.types.ImageData:
    if layer:
        processed_iSRRF = srrf(layer, magnification, spatial_radius, symmetryAxis, fstart, fend)
        #viewer.add_image(processed_iSRRF, scale=layer.scale, name='SRRF_processed of '+str(layer.name))
        viewer.add_image(processed_iSRRF, name='SRRF_processed of '+str(layer.name))

def mssr_module(viewer: 'napari.Viewer', layer: Image, amplification_factor: int = 1, PSF_p: float = 1.0, order: int = 1)-> napari.types.ImageData:
    if layer:
        img_layer = np.array(layer.data)
        if len(img_layer.shape) == 2:
            processed_img = MSSR(img_layer, PSF_p,  amplification_factor, order, True)
            viewer.add_image(processed_img, scale=layer.scale, name='MSSR_processed of '+str(layer.name))
        elif len(img_layer.shape) == 3:
            processed_img = TMSSR(img_layer, PSF_p,  amplification_factor, order, True)
            viewer.add_image(processed_img, scale=layer.scale, name='MSSR_processed of '+str(layer.name))

def esi_module(viewer: 'napari.Viewer', layer: Image, nrResImage: int = 10, nrBins: int = 100, esi_order: int = 4, normOutput: bool= True)-> napari.types.ImageData:
    if layer:
        img_layer = np.array(layer.data)
        esi_SR = ESI_Analysis(img_layer, np.amin(img_layer), np.amax(img_layer), nrBins, esi_order, nrResImage, normOutput)
        viewer.add_image(esi_SR, scale=layer.scale, name='ESI order='+str(esi_order)+' of '+str(layer.name))
