"""
This module is creates the GUI for the FF SRM methods
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory, magicgui
from qtpy import   QtCore, QtGui
from qtpy.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,\
                           QWidget, QSpinBox, QLabel, QSpacerItem, QSizePolicy,\
                           QDoubleSpinBox, QComboBox, QCheckBox, QSlider, QFileDialog,\
                           QMessageBox, QMainWindow, QApplication

from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from skimage import io
import datetime
import pathlib


if TYPE_CHECKING:
    import napari

import napari
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from vispy.color import Colormap
import os
import sys


from .core_mssr import mssr_class
from .core_esi import esi_class
from .my_popW import Ui_MainWindow
from .core_decor import *
from .core_sofi import sofi_class
from .core_srrf import srrf_class
from .core_musical import musical_class

#Import FF-SRM methods cores
my_mssr = mssr_class()
my_esi = esi_class()
my_sofi = sofi_class()
my_srrf = srrf_class()
my_musical = musical_class()


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
        self.msg.close()


    def batch(self):
        self.flagBatch = True
        self.my_files, other_data = QFileDialog.getOpenFileNames(self, "Select Files")
        first = self.my_files[0].split("/")
        first.pop()
        my_dir = "/".join(first)
        self.results_dir = my_dir+"/MSSR_results"


    def _run(self):
        if str(self.viewer.layers.selection.active) == 'None' and self.flagBatch == True:
            self.flagBatch = False
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

            if tempAn == False:
                os.mkdir(self.results_dir)
                for el in self.my_files:
                    try:
                        img = io.imread(el)
                    except:
                        continue
                    if len(img.shape) == 2:
                        processed_img = my_mssr.sfMSSR(img, fwhm, amp, order, mesh, ftI, intNorm)
                        io.imsave(self.results_dir+"/"+"MSSR "+el.split("/").pop(),processed_img)
                    elif len(img.shape) == 3:
                        processed_img = my_mssr.tMSSR(img, fwhm, amp, order, mesh, ftI, intNorm)
                        io.imsave(self.results_dir+"/"+"MSSR "+el.split("/").pop(),processed_img)
            else:
                first = self.results_dir.split("/")
                first.pop()
                my_dir = "/".join(first)
                if  first[-1] == "MSSR_results":
                    tempAn_dir = my_dir + "/tMSSR_results"
                    if os.path.exists(tempAn_dir) == False:
                        os.mkdir(tempAn_dir)

                    for el in self.my_files:
                        try:
                            img = io.imread(el)
                        except:
                            continue
                        if len(img.shape) == 3:
                            temp_procesed, staMeth = self.call_statistical_int_batch(img)
                            el_name = el.split("/")[-1].split(".")[0]
                            el_format = el.split(".")[-1]
                            io.imsave(tempAn_dir+"/"+"t"+ el_name + " " + staMeth + "." + el_format,temp_procesed)

                else:
                    os.mkdir(self.results_dir)
                    for el in self.my_files:
                        try:
                            img = io.imread(el)
                        except:
                            continue
                        if len(img.shape) == 2:
                            processed_img = my_mssr.sfMSSR(img, fwhm, amp, order, mesh, ftI, intNorm)
                            io.imsave(self.results_dir+"/"+"MSSR "+el.split("/").pop(),processed_img)
                        elif len(img.shape) == 3:
                            tempAn_dir = self.results_dir + "/tMSSR_results"
                            if os.path.exists(tempAn_dir) == False:
                                os.mkdir(tempAn_dir)
                            processed_img = my_mssr.tMSSR(img, fwhm, amp, order, mesh, ftI, intNorm)
                            io.imsave(self.results_dir+"/"+"MSSR "+el.split("/").pop(),processed_img)
                            temp_procesed, staMeth = self.call_statistical_int_batch(processed_img)
                            el_name = el.split("/").pop().split(".")[0]
                            el_format = el.split("/").pop().split(".")[-1]
                            io.imsave(tempAn_dir+"/"+"tMSSR "+ el_name + " " + staMeth + "." + el_format,temp_procesed)



        elif self.viewer.layers.selection.active.rgb == True:
            raise TypeError("Only single channel images are allowed")

        else:
            self.selected_im_name = str(self.viewer.layers.selection.active)
            img = self.viewer.layers[self.selected_im_name].data

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

        napari.utils.notifications.show_info("Process complete")




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


    def call_statistical_int_batch(self,processed_img):
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
        return tIm, staMeth

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
        label1.setText("# images in output")
        self.spinBox1 = QSpinBox()
        self.spinBox1.setMinimum(1)
        self.spinBox1.setMaximum(200)
        self.spinBox1.setValue(10)

        label2 = QLabel()
        label2.setText("# bins for entropy")
        self.spinBox2 = QSpinBox()
        self.spinBox2.setMinimum(1)
        self.spinBox2.setMaximum(200)
        self.spinBox2.setValue(100)

        label3 = QLabel()
        label3.setText("Order")
        self.spinBox3 = QSpinBox()
        self.spinBox3.setMinimum(1)
        self.spinBox3.setValue(4)

        self.CheckBox1 = QCheckBox()
        self.CheckBox1.setText("Intensity Normalization")
        self.CheckBox1.setChecked(True)

        btnG = QPushButton("Run")
        myFont=QtGui.QFont()
        myFont.setBold(True)
        btnG.setFont(myFont)
        btnG.clicked.connect(self._run)


        #Seting up widget layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(label1)
        self.layout().addWidget(self.spinBox1)
        self.layout().addSpacing(20)
        self.layout().addWidget(label2)
        self.layout().addWidget(self.spinBox2)
        self.layout().addSpacing(20)
        self.layout().addWidget(label3)
        self.layout().addWidget(self.spinBox3)
        self.layout().addSpacing(20)
        self.layout().addWidget(self.CheckBox1)
        self.layout().addSpacing(30)
        self.layout().addWidget(btnG)

#Rest of methods of the esi_caller class

    def _run(self):
        im = self.viewer.layers.selection.active
        self.selected_im_name = str(self.viewer.layers.selection.active)

        if self.CheckBox1.checkState() == 0:
            normOutput = False
        else:
            normOutput = True

        if im.rgb == True:
            raise TypeError("Only single channel images are allowed")
        elif len(im.data.shape) < 3 or im.data.shape[0] == 1:
            raise TypeError("Image stack should be provided")
        else:
            stck = im.data
            esi_result = my_esi.ESI_Analysis(stck, np.amin(stck), np.amax(stck), self.spinBox2.value(), self.spinBox3.value(), self.spinBox1.value(), normOutput)

            self.viewer.add_image(esi_result, name="ESI "+self.selected_im_name)

            print(esi_result.shape)

            if esi_result.shape[0] > 2:
                sum_esi = np.sum(esi_result, axis=0)
                self.viewer.add_image(sum_esi, name="summed ESI "+self.selected_im_name)

        napari.utils.notifications.show_info("Process complete")

class sofi_caller(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
    #widgets instantiation to be added to the viewer layout
    #only vatriables to be called for other functions are set as self.
        self.build()


    def build(self):
        #instanciating the widgets items
        label1 = QLabel()
        label1.setText("Amplification Factor")
        self.spinBox1 = QSpinBox()
        self.spinBox1.setMinimum(1)
        self.spinBox1.setMaximum(10)
        self.spinBox1.setValue(2)

        label2 = QLabel()
        label2.setText("Moment Order")
        self.spinBox2 = QSpinBox()
        self.spinBox2.setMinimum(1)
        self.spinBox2.setMaximum(100)
        self.spinBox2.setValue(4)

        self.CheckBox1 = QCheckBox()
        self.CheckBox1.setText("\t\tGaussian mask\n \t\t\tparameters")
        self.CheckBox1.setChecked(False)
        self.CheckBox1.toggled.connect(self.setGM)
        self.CheckBox1.setHidden(True)

        self.label3 = QLabel()
        self.label3.setText("Gaussian mask shape")
        self.spinBox3 = QSpinBox()
        self.spinBox3.setMinimum(1)
        self.spinBox3.setMaximum(999)
        self.spinBox3.setValue(204)
        self.label3.setHidden(True)
        self.spinBox3.setHidden(True)

        self.label4 = QLabel()
        self.label4.setText("Gaussian mask \u03C3")
        self.spinBox4 = QSpinBox()
        self.spinBox4.setMinimum(1)
        self.spinBox4.setMaximum(100)
        self.spinBox4.setValue(8)
        self.label4.setHidden(True)
        self.spinBox4.setHidden(True)

        self.CheckBox2 = QCheckBox()
        self.CheckBox2.setText("\tSK deconvolution\n \t\t\tparameters")
        self.CheckBox2.setChecked(False)
        self.CheckBox2.toggled.connect(self.setSKD)

        self.label5 = QLabel()
        self.label5.setText("\u03BB parameter")
        self.spinBox5 = QDoubleSpinBox()
        self.spinBox5.setMinimum(1)
        self.spinBox5.setMaximum(99)
        self.spinBox5.setValue(1.5)
        self.label5.setHidden(True)
        self.spinBox5.setHidden(True)

        self.label6 = QLabel()
        self.label6.setText("No. of iterations")
        self.spinBox6 = QSpinBox()
        self.spinBox6.setMinimum(1)
        self.spinBox6.setMaximum(100)
        self.spinBox6.setValue(20)
        self.label6.setHidden(True)
        self.spinBox6.setHidden(True)

        self.label7 = QLabel()
        self.label7.setText("Window size")
        self.spinBox7 = QSpinBox()
        self.spinBox7.setMinimum(1)
        self.spinBox7.setMaximum(999)
        self.spinBox7.setValue(100)
        self.label7.setHidden(True)
        self.spinBox7.setHidden(True)


        btnRun = QPushButton("Run")
        myFont=QtGui.QFont()
        myFont.setBold(True)
        btnRun.setFont(myFont)
        btnRun.clicked.connect(self._run)


        #Seting up widget layout
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(label1)
        self.layout().addWidget(self.spinBox1)
        self.layout().addSpacing(20)
        self.layout().addWidget(label2)
        self.layout().addWidget(self.spinBox2)
        self.layout().addSpacing(20)
        self.layout().addWidget(self.CheckBox2)
        self.layout().addSpacing(20)
        self.layout().addWidget(self.label5)
        self.layout().addWidget(self.spinBox5)
        self.layout().addWidget(self.label6)
        self.layout().addWidget(self.spinBox6)
        self.layout().addWidget(self.label7)
        self.layout().addWidget(self.spinBox7)
        self.layout().addSpacing(20)
        self.layout().addWidget(self.CheckBox1)
        self.layout().addSpacing(20)
        self.layout().addWidget(self.label3)
        self.layout().addWidget(self.spinBox3)
        self.layout().addWidget(self.label4)
        self.layout().addWidget(self.spinBox4)
        self.layout().addSpacing(30)
        self.layout().addWidget(btnRun)

    def setGM(self, val):
        if val == True:
            self.label3.setHidden(False)
            self.spinBox3.setHidden(False)
            self.label4.setHidden(False)
            self.spinBox4.setHidden(False)
        else:
            self.label3.setHidden(True)
            self.spinBox3.setHidden(True)
            self.label4.setHidden(True)
            self.spinBox4.setHidden(True)

    def setSKD(self, val):
        if val == True:
            self.label5.setHidden(False)
            self.spinBox5.setHidden(False)
            self.label6.setHidden(False)
            self.spinBox6.setHidden(False)
            self.label7.setHidden(False)
            self.spinBox7.setHidden(False)
            self.CheckBox1.setHidden(False)
        else:
            self.label5.setHidden(True)
            self.spinBox5.setHidden(True)
            self.label6.setHidden(True)
            self.spinBox6.setHidden(True)
            self.label7.setHidden(True)
            self.spinBox7.setHidden(True)
            self.CheckBox1.setHidden(True)
            self.label3.setHidden(True)
            self.spinBox3.setHidden(True)
            self.label4.setHidden(True)
            self.spinBox4.setHidden(True)


    def _run(self):
        im = self.viewer.layers.selection.active.data
        self.selected_im_name = str(self.viewer.layers.selection.active)

        if im.ndim < 3 or im.shape[0] < 3:
            raise TypeError("The current layer is not an STACK!")


        #"Amplification Factor"
        interp = self.spinBox1.value()
        #"Moment Order"
        order_val = self.spinBox2.value()
        #"Gaussian mask shape"
        gms=self.spinBox3.value()
        #"Gaussian mask sigma
        gm_sigma = self.spinBox4.value()
        #lambda parameter"
        deconv_lambda = self.spinBox5.value()
        #"No. of iterations"
        deconv_iter = self.spinBox6.value()
        #"Window size"
        window_size = [self.spinBox7.value(), self.spinBox7.value()]


        moment_im = my_sofi.moment_image(im, order=order_val, mean_im=None,finterp=True, interp_num = interp)
        deconv_psf0 = my_sofi.gauss2d_mask((gms, gms), gm_sigma)
        deconv_psf0 = deconv_psf0 / np.max(deconv_psf0)
        deconv_im = my_sofi.deconvsk(deconv_psf0, moment_im, deconv_lambda, deconv_iter)
        mask_im = my_sofi.average_image_with_finterp(im,interp)
        ldrc_im = my_sofi.ldrc(window_size = window_size, mask_im = mask_im, input_im = deconv_im)
        self.viewer.add_image(ldrc_im, name="SOFI "+self.selected_im_name)
        napari.utils.notifications.show_info("Process complete")

class srrf_caller(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.viewer.layers.selection.events.active.connect(self.pbta)
        self.the_name = " "
    #widgets instantiation to be added to the viewer layout
    #only vatriables to be called for other functions are set as self.
        self.build()
        self.pbta()



    def build(self):
        #instanciating the widgets items
        label1 = QLabel()
        label1.setText("Amplification Factor")
        self.spinBox1 = QSpinBox()
        self.spinBox1.setMinimum(1)
        self.spinBox1.setMaximum(10)
        self.spinBox1.setValue(1)

        label2 = QLabel()
        label2.setText("Spatial radius")
        self.spinBox2 = QSpinBox()
        self.spinBox2.setMinimum(1)
        self.spinBox2.setMaximum(10)
        self.spinBox2.setValue(5)

        label3 = QLabel()
        label3.setText("Symetry Axis")
        self.spinBox3 = QSpinBox()
        self.spinBox3.setMinimum(1)
        self.spinBox3.setMaximum(10)
        self.spinBox3.setValue(6)

        label4 = QLabel()
        label4.setText("Start frame")
        self.spinBox4 = QSpinBox()
        self.spinBox4.setMinimum(0)
        self.spinBox4.setMaximum(1000)
        self.spinBox4.setValue(0)

        label5 = QLabel()
        label5.setText("End frame")
        self.spinBox5 = QSpinBox()
        self.spinBox5.setMinimum(0)
        self.spinBox5.setMaximum(1000)
        self.spinBox5.setValue(2)

        self.CheckBox1 = QCheckBox()
        self.CheckBox1.setText("Change statistical\nintegration")
        self.CheckBox1.setChecked(False)
        self.CheckBox1.toggled.connect(self.setTemp)

        self.ComboBoxT = QComboBox()
        self.ComboBoxT.clear()
        self.ComboBoxT.addItems(["TPM","Var","SOFI","CV·σ"])
        self.ComboBoxT.currentTextChanged.connect(self.onActivated)
        self.ComboBoxT.setHidden(True)

        self.spinBoxS = QSpinBox()
        self.spinBoxS.setMinimum(1)
        self.spinBoxS.setMaximum(3)
        self.spinBoxS.setHidden(True)

        btnRun = QPushButton("Run")
        myFont=QtGui.QFont()
        myFont.setBold(True)
        btnRun.setFont(myFont)
        btnRun.clicked.connect(self._run)

        #Seting up widget layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(label1)
        self.layout().addWidget(self.spinBox1)
        self.layout().addSpacing(30)
        self.layout().addWidget(label2)
        self.layout().addWidget(self.spinBox2)
        self.layout().addSpacing(30)
        self.layout().addWidget(label3)
        self.layout().addWidget(self.spinBox3)
        self.layout().addSpacing(30)
        self.layout().addWidget(label4)
        self.layout().addWidget(self.spinBox4)
        self.layout().addSpacing(30)
        self.layout().addWidget(label5)
        self.layout().addWidget(self.spinBox5)
        self.layout().addSpacing(30)
        self.layout().addWidget(self.CheckBox1)
        self.layout().addWidget(self.ComboBoxT)
        self.layout().addWidget(self.spinBoxS)
        self.layout().addSpacing(40)
        self.layout().addWidget(btnRun)

    def setTemp(self,d):
        if d == True:
            #self.labelT.setHidden(False)
            self.ComboBoxT.setHidden(False)
        else:
            #self.labelT.setHidden(True)
            self.ComboBoxT.setHidden(True)
            self.spinBoxS.setHidden(True)

    def pbta(self):
        try:
            im = self.viewer.layers.selection.active.data
            if len(im.shape) < 3:
                self.spinBox5.setValue(0)
            else:
                len_im = im.shape[0] - 1
                self.spinBox5.setValue(len_im)
        except:
            print(" ")

    def onActivated(self,s):
        if s == "SOFI":
            self.spinBoxS.setHidden(False)
        else:
            self.spinBoxS.setHidden(True)


    def _run(self):

        if self.the_name in self.viewer.layers:
            exist_flag = True
        else:
            exist_flag = False

        if exist_flag and self.CheckBox1.checkState() == 2:
            staMeth = self.ComboBoxT.currentText()
            processed_iSRRF = self.viewer.layers[self.the_name].data
            if staMeth == "TPM":
                tIm = my_mssr.TPM(processed_iSRRF)
            elif staMeth == "Var":
                tIm = my_mssr.tVar(processed_iSRRF)
            elif staMeth == "SOFI":
                tIm = my_mssr.TRAC(processed_iSRRF, self.spinBoxS.value())
            elif staMeth == "CV·σ":
                tIm = my_mssr.varCoef(processed_iSRRF)
            output_name = staMeth + " SRRF " + self.stack_name
            self.viewer.add_image(tIm, name=output_name)

        elif not(exist_flag) and self.CheckBox1.checkState() == 2:

            im = self.viewer.layers.selection.active.data
            self.selected_im_name = str(self.viewer.layers.selection.active)

            magnification = self.spinBox1.value() #4
            spatial_radius = self.spinBox2.value()
            symmetryAxis = self.spinBox3.value()
            fstart = self.spinBox4.value()
            fend = self.spinBox5.value()

            processed_iSRRF = my_srrf.srrf(im, magnification, spatial_radius, symmetryAxis, fstart, fend)
            self.the_name = "processed "+ self.selected_im_name
            self.stack_name = self.selected_im_name
            self.viewer.add_image(processed_iSRRF, name=self.the_name)
            stack_flag = True
            srrf_im = my_mssr.tMean(processed_iSRRF)
            self.viewer.add_image(srrf_im, name="SRRF "+ self.selected_im_name)

            staMeth = self.ComboBoxT.currentText()
            if staMeth == "TPM":
                tIm = my_mssr.TPM(processed_iSRRF)
            elif staMeth == "Var":
                tIm = my_mssr.tVar(processed_iSRRF)
            elif staMeth == "SOFI":
                tIm = my_mssr.TRAC(processed_iSRRF, self.spinBoxS.value())
            elif staMeth == "CV·σ":
                tIm = my_mssr.varCoef(processed_iSRRF)
            self.viewer.add_image(tIm, name=staMeth + " SRRF " + self.selected_im_name)

        else:
            im = self.viewer.layers.selection.active.data
            self.selected_im_name = str(self.viewer.layers.selection.active)

            magnification = self.spinBox1.value() #4
            spatial_radius = self.spinBox2.value()
            symmetryAxis = self.spinBox3.value()
            fstart = self.spinBox4.value()
            fend = self.spinBox5.value()

            processed_iSRRF = my_srrf.srrf(im, magnification, spatial_radius, symmetryAxis, fstart, fend)
            self.the_name = "processed " + self.selected_im_name
            self.stack_name = self.selected_im_name
            self.viewer.add_image(processed_iSRRF, name=self.the_name)
            stack_flag = True
            srrf_im = my_mssr.tMean(processed_iSRRF)
            self.viewer.add_image(srrf_im, name="SRRF "+self.selected_im_name)

        if self.the_name in self.viewer.layers:
            exist_flag = True
        else:
            self.the_name = " "

        napari.utils.notifications.show_info("Process complete")


class musical_caller(QWidget):

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.build()

    def build(self):
        #instanciating the widgets items
        label1 = QLabel()
        label1.setText("Emission λ [nm]")
        self.spinBox1 = QSpinBox()
        self.spinBox1.setMinimum(150)
        self.spinBox1.setMaximum(999)
        self.spinBox1.setValue(510)

        label2 = QLabel()
        label2.setText("Numerical Aperture")
        self.DspinBox1 = QDoubleSpinBox()
        self.DspinBox1.setMinimum(0)
        self.DspinBox1.setMaximum(10)
        self.DspinBox1.setValue(1.4)

        label3 = QLabel()
        label3.setText("Magnification")
        self.spinBox2 = QSpinBox()
        self.spinBox2.setMinimum(1)
        self.spinBox2.setMaximum(999)
        self.spinBox2.setValue(100)

        label4 = QLabel()
        label4.setText("Pixel size [nm]")
        self.spinBox3 = QSpinBox()
        self.spinBox3.setMinimum(1)
        self.spinBox3.setMaximum(9999)
        self.spinBox3.setValue(8000)

        self.plot_button = QPushButton("Plot singular values")
        self.plot_button.clicked.connect(self.set_plot)

        label5 = QLabel()
        label5.setText("Threshold")
        self.DspinBox2 = QDoubleSpinBox()
        self.DspinBox2.setMinimum(-10)
        self.DspinBox2.setMaximum(10)
        self.DspinBox2.setValue(-0.5)

        label6 = QLabel()
        label6.setText("Alpha")
        self.spinBox4 = QSpinBox()
        self.spinBox4.setMinimum(0)
        self.spinBox4.setMaximum(99)
        self.spinBox4.setValue(4)

        label7 = QLabel()
        label7.setText("Subpixels per pixel")
        self.spinBox5 = QSpinBox()
        self.spinBox5.setMinimum(1)
        self.spinBox5.setMaximum(99)
        self.spinBox5.setValue(20)

        btnRun = QPushButton("Run")
        myFont=QtGui.QFont()
        myFont.setBold(True)
        btnRun.setFont(myFont)
        btnRun.clicked.connect(self.image_processing)

        #Seting up widget layout
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(label1)
        self.layout().addWidget(self.spinBox1)
        #self.layout().addSpacing(30)
        self.layout().addWidget(label2)
        self.layout().addWidget(self.DspinBox1)
        #self.layout().addSpacing(30)
        self.layout().addWidget(label3)
        self.layout().addWidget(self.spinBox2)
        #self.layout().addSpacing(30)
        self.layout().addWidget(label4)
        self.layout().addWidget(self.spinBox3)
        self.layout().addSpacing(30)
        self.layout().addWidget(self.plot_button)
        self.layout().addSpacing(30)
        self.layout().addWidget(label5)
        self.layout().addWidget(self.DspinBox2)
        #self.layout().addSpacing(30)
        self.layout().addWidget(label6)
        self.layout().addWidget(self.spinBox4)
        #self.layout().addSpacing(30)
        self.layout().addWidget(label7)
        self.layout().addWidget(self.spinBox5)
        self.layout().addSpacing(40)
        self.layout().addWidget(btnRun)

    def set_plot(self):
        imageStack = self.viewer.layers.selection.active.data
        if imageStack.ndim < 3 or imageStack.shape[0] < 3:
            raise TypeError("The current layer is not an STACK!")
        imageStack = imageStack / np.max(imageStack)
        numberOfImages, y_in, x_in = imageStack.shape

        lam = self.spinBox1.value() #510
        NA =  self.DspinBox1.value() #1.4
        M = self.spinBox2.value()#100
        PixelSize = self.spinBox3.value() #8000
        TestPointsPerPixel = self.spinBox5.value() #20

        G_PSF, x_foc, y_foc, N, subpixel_size = my_musical.compute_PSF(lam, NA, M, PixelSize, TestPointsPerPixel)

        x_val = list(range(1, x_in + 1))
        y_val = list(range(1, y_in + 1))

        Threshold = self.DspinBox2.value() #-0.5
        N_w = np.sqrt(N)
        Alpha = self.spinBox4.value() #4

        S_matrix = my_musical.function_SVD_scan_parallel(imageStack, x_val, y_val, N_w, G_PSF, x_foc, y_foc, TestPointsPerPixel)

        with plt.style.context('dark_background'):
            my_plot_widget = FigureCanvas(Figure(constrained_layout=True))
            fig = my_plot_widget.figure
            ax1 = fig.add_subplot()
            ax1.plot(np.log10(S_matrix))
            ax1.set_xlabel('Number')
            ax1.set_ylabel('log10 value')

        plot_dock = self.viewer.window.add_dock_widget(my_plot_widget, name='Plot', area = 'right')

        napari.utils.notifications.show_info("Process complete")

    def image_processing(self):
        imageStack = self.viewer.layers.selection.active.data
        if imageStack.ndim < 3 or imageStack.shape[0] < 3:
            raise TypeError("The current layer is not an STACK!")

        self.selected_im_name = str(self.viewer.layers.selection.active)
        imageStack = imageStack / np.max(imageStack)
        numberOfImages, y_in, x_in = imageStack.shape

        lam = self.spinBox1.value() #510
        NA =  self.DspinBox1.value() #1.4
        M = self.spinBox2.value()#100
        PixelSize = self.spinBox3.value() #8000
        TestPointsPerPixel = self.spinBox5.value() #20
        G_PSF, x_foc, y_foc, N, subpixel_size = my_musical.compute_PSF(lam, NA, M, PixelSize, TestPointsPerPixel)

        x_val = list(range(1, x_in + 1))
        y_val = list(range(1, y_in + 1))

        Threshold = self.DspinBox2.value() #-0.5
        N_w = np.sqrt(N)
        Alpha = self.spinBox4.value() #4

        musical_im, _ = my_musical.function_MUSIC_scan_parallel(imageStack, x_val, y_val, Threshold, N_w, G_PSF, x_foc, y_foc, TestPointsPerPixel, Alpha)

        self.viewer.add_image(musical_im, name="MUSICAL "+self.selected_im_name)
        napari.utils.notifications.show_info("Process complete")


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
        # if layer_data.ndim > 2 and layer_data.shape[-1] != 3:
        #     raise TypeError("The current layer is not a MULTICHANNEL image")
        if self.viewer.layers.selection.active.rgb == False:
            raise TypeError("The current layer is not an RGB image")

        if layer_data.shape[-1] == 3:
            # Split the RGB image into separate channels
            red = layer_data[:, :, 0]
            green = layer_data[:, :, 1]
            blue = layer_data[:, :, 2]

            # Create three new ImageData objects for each channel
            red_data = napari.types.ImageData(red)
            green_data = napari.types.ImageData(green)
            blue_data = napari.types.ImageData(blue)

            # Create three LayerDataTuple objects to add the new layers to napari
            self.viewer.add_image(data=red_data, name="Red", colormap="red")
            self.viewer.add_image(data=green_data, name="Green", colormap="green")
            self.viewer.add_image(data=blue_data, name="Blue", colormap="blue")

        else:
            layer_name = self.viewer.layers.selection.active.name
            for n in range(layer_data.shape[-1]):
                self.viewer.add_image(data=layer_data[:,:,n], name=layer_name + " Ch "+str(n))
