"""
napari-superres: Super-resolution napari plugins collection
"""
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton, QGridLayout, QGroupBox
from napari.layers import Image, Labels, Layer, Points
from magicgui import magicgui, magic_factory
import napari
from napari import Viewer
try:
    from napari.settings import SETTINGS
except ImportError:
    print("Warning: import of napari.settings failed - 'save window geometry' option will not be used")
from magicgui.widgets import SpinBox, FileEdit, Slider, FloatSlider, Label, Container, MainWindow, ComboBox, TextEdit, PushButton, ProgressBar, Select
import skimage
import skimage.morphology
import skimage.filters
from skimage.feature import peak_local_max
from skimage.transform import rotate
from skimage.segmentation import watershed
from skimage import measure
import numpy as np
#import pandas as pd
import datatable as dt
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
import json
import os
import inspect

protocols={'SRRF','MSSR'}


@magic_factory(labels=False,
         label={'widget_type':'Label', 'value':"SRRF_Parameters"},
         threshold={'widget_type': 'FloatSlider', "max": 65535.0, 'min':0.0},
         call_button="Apply",
         persist=True
         )
def SRRF_module(viewer: 'napari.Viewer', label, layer: Image, amplification_factor: int = 1, PSF_p: int = 1, order: int = 1)-> napari.types.ImageData:
    if layer:
        th=layer.data>threshold
        viewer.add_image(th, scale=layer.scale, name='Threshold th='+str(threshold)+' of '+str(layer.name))

@magic_factory(labels=False,
         label={'widget_type':'Label', 'value':"MSSR_Parameters"},
         threshold={'widget_type': 'FloatSlider', "max": 65535.0, 'min':0.0},
         call_button="Apply",
         persist=True
         )
def MSSR_module(viewer: 'napari.Viewer', label, layer: Image, amplification_factor: int = 1, PSF_p: int = 1, order: int = 1)-> napari.types.ImageData:
    if layer:
        th=layer.data>threshold
        viewer.add_image(th, scale=layer.scale, name='Threshold th='+str(threshold)+' of '+str(layer.name))

@magic_factory(
               auto_call=False,
               call_button=True,
               dropdown={"choices": protocols},
               textbox={'widget_type': 'TextEdit', 'value': protocols_description, 'label':'napari-superres'},
               labels=False
                )
def launch_superres(
        viewer: 'napari.Viewer',
        textbox,
        protocols: list= protocols,
        dropdown: str= 'Segment a single population'
        ):

        try:
            SETTINGS.application.save_window_geometry = "False"
        except:
            pass


        dock_widgets=MainWindow(name='napari_superres plugin', annotation=None, label=None, tooltip=None, visible=True,
                               enabled=True, gui_only=False, backend_kwargs={}, layout='horizontal', widgets=(), labels=True)
        viewer.window.add_dock_widget(dock_widgets, name=str(dropdown), area='bottom')
        if dropdown == 'SRRF':
            SRRF_processing=Container(name='', annotation=None, label=None, visible=True, enabled=True,
                                          gui_only=False, layout='horizontal', labels=False)
            SRRF_processing.insert(0, SRRF_module)

            dock_widgets.insert(0, SRRF_processing)

            launch_superres._call_button.text = 'Restart with the selected plugin'

        if dropdown == 'MSSR':
            MSSR_processing=Container(name='', annotation=None, label=None, visible=True, enabled=True,
                                          gui_only=False, layout='horizontal', labels=False)
            MSSR_processing.insert(0, MSSR_module)

            dock_widgets.insert(0, MSSR_processing)

            launch_superres._call_button.text = 'Restart with the selected plugin'
