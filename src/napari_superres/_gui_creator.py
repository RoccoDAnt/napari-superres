"""
This module is creates the GUI for the FF SRM methods
"""
from typing import TYPE_CHECKING
#from napari.layers import Image, Labels, Layer, Points

from magicgui import magic_factory, magicgui
from qtpy import   QtCore, QtGui
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,\
                           QWidget, QSpinBox, QLabel, QSpacerItem, QSizePolicy,\
                           QDoubleSpinBox, QComboBox, QCheckBox, QSlider, QFileDialog,\
                           QMessageBox, QMainWindow, QApplication
from skimage import io
import datetime
import pathlib


if TYPE_CHECKING:
    import napari

import napari
from napari.types import ImageData
import numpy as np
from vispy.color import Colormap
import os
import sys


from .core_mssr import mssr_class
from .my_popW import Ui_MainWindow
from .core_decor import *


#Import MSSR core
my_mssr = mssr_class()


class mssr_caller(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.flag = False
        self.flagBatch = False
        self.popWindow = Ui_MainWindow()

#widgets instantiation to be added to the viewer layout
#only vatriables to be called for other functions are set as self.
        self.build()

    def build(self):
        #instanciating the widgets items
        label1 = QLabel()
        label1.setText("Amplification Factor")
        self.spinBox1 = QSpinBox()
        self.spinBox1.setMinimum(1)
        self.spinBox1.setValue(1)

        label3 = QLabel()
        label3.setText("Order")
        self.spinBox3 = QSpinBox()
        self.spinBox3.setMinimum(0)
        self.spinBox3.setValue(0)

        label2 = QLabel()
        label2.setText("PSF FWHM")
        self.DoubleSpinBox1=QDoubleSpinBox()
        self.DoubleSpinBox1.setMinimum(0.0)
        self.DoubleSpinBox1.setValue(4.2)

        btn1 = QPushButton("Compute \n PSF FWHM")
        btn1.clicked.connect(self.calculator)

        label4 = QLabel()
        label4.setText("Interpolation Type")
        self.ComboBox4 = QComboBox()
        self.ComboBox4.clear()
        self.ComboBox4.addItems(["Bicubic","Fourier"])

        self.CheckBox1 = QCheckBox()
        self.CheckBox1.setText("Minimize Meshing")
        self.CheckBox1.setChecked(True)

        self.CheckBox4 = QCheckBox()
        self.CheckBox4.setText("Intensity Normalization")
        self.CheckBox4.setChecked(True)

        self.CheckBox5 = QCheckBox()
        self.CheckBox5.setText("Temporal Analysis")
        self.CheckBox5.setChecked(False)
        self.CheckBox5.toggled.connect(self.setTemp)

        self.labelT = QLabel()
        self.labelT.setText("Statistical Integration")
        self.labelT.setHidden(True)
        self.ComboBoxT = QComboBox()
        self.ComboBoxT.clear()
        self.ComboBoxT.addItems(["TPM","Var","Mean","SOFI","CV·σ"])
        self.ComboBoxT.currentTextChanged.connect(self.onActivated)
        self.ComboBoxT.setHidden(True)

        self.spinBoxS = QSpinBox()
        self.spinBoxS.setMinimum(1)
        self.spinBoxS.setMaximum(3)
        self.spinBoxS.setHidden(True)

        btnB = QPushButton("Batch Analysis")
        btnB.clicked.connect(self.batch)

        btnG = QPushButton("Run")
        myFont=QtGui.QFont()
        myFont.setBold(True)
        btnG.setFont(myFont)
        btnG.clicked.connect(self._run)

#Adding the items to the QBoxLayout (vertical configuration)
#Only the widgets configured here and modified in the call functions will be defined as 'self'
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(label1)
        self.layout().addWidget(self.spinBox1)
        self.layout().addWidget(label3)
        self.layout().addWidget(self.spinBox3)
        self.layout().addSpacing(30) #adds space to the elements in the layout
        self.layout().addWidget(label2)
        self.layout().addWidget(self.DoubleSpinBox1)
        self.layout().addWidget(btn1)
        self.layout().addSpacing(30)
        self.layout().addWidget(label4)
        self.layout().addWidget(self.ComboBox4)
        self.layout().addSpacing(30)
        self.layout().addWidget(self.CheckBox1)
        self.layout().addWidget(self.CheckBox4)
        self.layout().addWidget(self.CheckBox5)
        self.layout().addSpacing(30)
        self.layout().addWidget(self.labelT)
        self.layout().addWidget(self.ComboBoxT)
        self.layout().addWidget(self.spinBoxS)
        self.layout().addSpacing(20)
        self.layout().addWidget(btnB)
        self.layout().addSpacing(30)
        self.layout().addWidget(btnG)

#Rest of methods of the mssr_caller class
    def setTemp(self,d):
        if d == True:
            self.labelT.setHidden(False)
            self.ComboBoxT.setHidden(False)
        else:
            self.labelT.setHidden(True)
            self.ComboBoxT.setHidden(True)
            self.spinBoxS.setHidden(True)

    def onActivated(self,s):
        if s == "SOFI":
            self.spinBoxS.setHidden(False)
        else:
            self.spinBoxS.setHidden(True)

    def calculator(self):
        self.msg = self.popWindow
        self.msg.setupUi(self.msg)
        self.msg.retranslateUi(self.msg)
        self.msg.pushButtonCom.clicked.connect(self.bypass)
        self.msg.pushButtonCom2.clicked.connect(self.decorr)
        self.msg.show()

    def bypass(self):
        self.DoubleSpinBox1.setValue(self.msg.su_resul())


    def decorr(self):
        # load image
        image_input = self.viewer.layers.selection.active.data
        im_dim = image_input.shape
        if  len(im_dim) > 2:
            dummy = np.zeros((im_dim[0]))
            for i in range(im_dim[0]):
                dummy[i] = np.var(image_input[i,:,:])
                sl = np.argmax(dummy)
            image=image_input[sl,:,:]
            print("selected layer: ",sl)
        else:
            image = image_input

        pps = 5  # projected pixel size of 15nm
        # typical parameters for resolution estimate
        Nr = 50
        Ng = 10
        r = np.linspace(0, 1, Nr)
        GPU = False

        # Apodize image edges with a cosine function over 20 pixels
        image, mask = apodImRect(image, 20)

        # Compute resolution
        figID = 100
        if GPU:
            from numba import cuda
            image_gpu = cuda.to_device(image)
            kcMax, A0, _, _ = getDcorr(image_gpu, r, Ng, figID)
        else:
            kcMax, A0, _, _ = getDcorr(image, r, Ng, figID)
        # Max resolution in pixels
        res = 2/kcMax

        #print(f'kcMax : {kcMax:.3f}, A0 : {A0[0]:.3f}')
        #print(f'Resolution: {2/kcMax:.3f}, [pixels]')
        self.DoubleSpinBox1.setValue(res)


    def batch(self):
        self.flagBatch = True
        self.my_files, other_data = QFileDialog.getOpenFileNames(self, "Select Files")
        first = self.my_files[0].split("/")
        first.pop()
        my_dir = "/".join(first)
        self.results_dir = my_dir+"/MSSR_resuslts"
        os.mkdir(self.results_dir)




    def _run(self):
        if self.viewer.layers.selection.active.rgb == True:
            raise TypeError("Only single channel images are allowed")
        else:
            fwhm = self.DoubleSpinBox1.value()
            amp = self.spinBox1.value()
            order = self.spinBox3.value()

            if self.CheckBox1.checkState() == 0:
                mesh = False
            else:
                mesh = True
            if self.ComboBox4.currentText() == "Fourier":
                ftI = True
            else:
                ftI = False
            if self.CheckBox4.checkState() == 0:
                intNorm = False
            else:
                intNorm = True
            if self.CheckBox5.checkState() == 0:
                tempAn = False
            else:
                tempAn = True

            if self.flagBatch == True:
                self.flagBatch = False
                for el in self.my_files:
                    img = io.imread(el)
                    if len(img.shape) == 2:
                        processed_img = my_mssr.sfMSSR(img, fwhm, amp, order, mesh, ftI, intNorm)
                        io.imsave(self.results_dir+"/"+"MSSR "+el.split("/").pop(),processed_img)
                    elif len(img.shape) == 3:
                        processed_img = my_mssr.tMSSR(img, fwhm, amp, order, mesh, ftI, intNorm)
                        io.imsave(self.results_dir+"/"+"MSSR "+el.split("/").pop(),processed_img)
            else:

                self.selected_im_name = str(self.viewer.layers.selection.active)
                img = self.viewer.layers[self.selected_im_name].data
                if self.flag == True:
                    if self.track_name not in self.viewer.layers:
                        self.flag = False

                if len(img.shape) == 2:
                    processed_img = my_mssr.sfMSSR(img, fwhm, amp, order, mesh, ftI, intNorm)
                    self.viewer.add_image(processed_img, name="MSSR "+self.selected_im_name)
                elif len(img.shape) == 3 and tempAn == False:
                    processed_img = my_mssr.tMSSR(img, fwhm, amp, order, mesh, ftI, intNorm)
                    self.track_name = "MSSR "+self.selected_im_name
                    self.viewer.add_image(processed_img, name= self.track_name)
                    self.flag = True
                elif len(img.shape) == 3 and tempAn == True:
                    if self.flag == True:
                        self.call_statistical_int(self.viewer.layers.selection.active.data)
                    else:
                        processed_img = my_mssr.tMSSR(img, fwhm, amp, order, mesh, ftI, intNorm)
                        self.track_name = "MSSR "+self.selected_im_name
                        self.viewer.add_image(processed_img, name= self.track_name)
                        self.call_statistical_int(processed_img)
                        self.flag = True


    def call_statistical_int(self,processed_img):
        staMeth = self.ComboBoxT.currentText()
        if staMeth == "TPM":
            print("TPM")
            tIm = my_mssr.TPM(processed_img)
        elif staMeth == "Var":
            tIm = my_mssr.tVar(processed_img)
            print("Var")
        elif staMeth == "Mean":
            tIm = my_mssr.tMean(processed_img)
            print("Mean")
        elif staMeth == "SOFI":
            tIm = my_mssr.TRAC(processed_img, self.spinBoxS.value())
            print("SOFI")
        elif staMeth == "CV·σ":
            print("Coefficient of Variation times Standard Deviation")
            tIm = my_mssr.varCoef(processed_img)
        if self.flag == True:
            self.viewer.add_image(tIm, name="t"+self.selected_im_name+" "+staMeth)
        else:
            self.viewer.add_image(tIm, name="tMSSR "+self.selected_im_name+" "+staMeth)



class esi_caller(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
    #widgets instantiation to be added to the viewer layout
    #only vatriables to be called for other functions are set as self.
        self.build()

    def build(self):
        #instanciating the widgets items
        label1 = QLabel()
        label1.setText("Upcoming...")
        #Seting up widget layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(label1)

    #Rest of methods of the esi_caller class

class SplitChannelsWidget(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Create a layout for the widget
        layout = QVBoxLayout()

        # Create a button to trigger the channel splitting
        self.split_button = QPushButton("Split Channels")
        self.split_button.clicked.connect(self.split_channels)
        layout.addWidget(self.split_button)

        # Set the layout for the widget
        self.setLayout(layout)

    def split_channels(self):
        # Get the active layer data from napari
        layer_data = self.viewer.layers.selection.active.data

        # Check if the layer is an RGB image
        if layer_data.ndim != 3 or layer_data.shape[-1] != 3:
            raise TypeError("The current layer is not an RGB image")

        # Split the RGB image into separate channels
        red = layer_data[:, :, 0]
        green = layer_data[:, :, 1]
        blue = layer_data[:, :, 2]

        # Create three new ImageData objects for each channel
        red_data = ImageData(red)
        green_data = ImageData(green)
        blue_data = ImageData(blue)

        # Create three LayerDataTuple objects to add the new layers to napari
        self.viewer.add_image(data=red_data, name="Red", colormap="red")
        self.viewer.add_image(data=green_data, name="Green", colormap="green")
        self.viewer.add_image(data=blue_data, name="Blue", colormap="blue")
