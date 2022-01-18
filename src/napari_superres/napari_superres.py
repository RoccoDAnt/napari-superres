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
            single_pop_protocol=Container(name='', annotation=None, label=None, visible=True, enabled=True,
                                          gui_only=False, layout='horizontal', labels=False)
            single_pop_protocol.insert(0, image_calibration)
            single_pop_protocol.insert(1, gaussian_blur_one_pop)
            single_pop_protocol.insert(2, threshold_one_pop)

            dock_widgets.insert(0,single_pop_protocol)

            launch_ZELDA._call_button.text = 'Restart with the selected plugin'

        if dropdown == 'MRRF':
            single_pop_protocol=Container(name='', annotation=None, label=None, visible=True, enabled=True,
                                          gui_only=False, layout='horizontal', labels=False)
            single_pop_protocol.insert(0, image_calibration)
            single_pop_protocol.insert(1, gaussian_blur_one_pop)
            single_pop_protocol.insert(2, threshold_one_pop)

            dock_widgets.insert(0,single_pop_protocol)

            launch_superres._call_button.text = 'Restart with the selected plugin'
