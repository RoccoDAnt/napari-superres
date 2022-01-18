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


# This is the actual plugin function, where we export our function
# (The functions themselves are defined below)
@napari_hook_implementation
def napari_experimental_provide_function():
    # we can return a single function
    # or a tuple of (function, magicgui_options)
    # or a list of multiple functions with or without options, as shown here:
    return [threshold, image_arithmetic, srrf_module, mssr_module, esi_module]


# 1.  First example, a simple function that thresholds an image and creates a labels layer
def threshold(data: "napari.types.ImageData", threshold: int) -> "napari.types.LabelsData":
    """Threshold an image and return a mask."""
    return (data > threshold).astype(int)


# 2. Second example, a function that adds, subtracts, multiplies, or divides two layers

# using Enums is a good way to get a dropdown menu.  Used here to select from np functions
class Operation(Enum):
    add = np.add
    subtract = np.subtract
    multiply = np.multiply
    divide = np.divide


def image_arithmetic(
    layerA: "napari.types.ImageData", operation: Operation, layerB: "napari.types.ImageData"
) -> "napari.types.LayerDataTuple":
    """Adds, subtracts, multiplies, or divides two same-shaped image layers."""
    return (operation.value(layerA, layerB), {"colormap": "turbo"})


def srrf_module(viewer: 'napari.Viewer', layer: Image, magnification: int = 4, spatial_radius: int = 5, simmetry_axis: int = 6, fstart: int = 0, fend: int = 100)-> napari.types.ImageData:
    pass
#    if layer:
#        th=layer.data>threshold
#        viewer.add_image(th, scale=layer.scale, name='Threshold th='+str(threshold)+' of '+str(layer.name))

def mssr_module(viewer: 'napari.Viewer', layer: Image, amplification_factor: int = 1, PSF_p: float = 1.0, order: int = 1)-> napari.types.ImageData:
    #pass
    if layer:
#        th=layer.data>threshold
#        viewer.add_image(th, scale=layer.scale, name='Threshold th='+str(threshold)+' of '+str(layer.name))
        processed_img=TMSSR(layer, PSF_p,  amplification_factor, order, True)
        viewer.add_image(processed_img, scale=layer.scale, name='MSSR_processed of '+str(layer.name))

def esi_module(viewer: 'napari.Viewer', layer: Image, nrResImage: int = 10, nrBins: int = 100, esi_order: int = 4, normOutput: bool= True)-> napari.types.ImageData:
    pass
#    if layer:
#        th=layer.data>threshold
#        viewer.add_image(th, scale=layer.scale, name='Threshold th='+str(threshold)+' of '+str(layer.name))
