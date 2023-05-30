################# SRRF ########
from skimage import io
import math
import numpy as np
from scipy.interpolate import griddata
from napari.utils import progress



##################### Functions #####################
##### buildRing
class srrf_class:
    def __init__(self):
        self.first = "init"
    def conection_test(self):
        print("conected")

    def buildRing(self, nRingCoordinates, spatialRadius, angleStep):
        x = np.zeros((2, nRingCoordinates))
        for i in list(range(0, (nRingCoordinates))) :
            x[0,i]= spatialRadius * np.cos(angleStep * i)
            x[1,i]= spatialRadius * np.sin(angleStep * i)
        return x

    ##### calculateGxGy
    def calculateGxGy(self, pixels, frame, x, y, height, width):
        v = 0
        for i in [-1,0,1] :
            x_ = min(max(x + i, 1), width - 1)
            y_ = min(max(y, 1), height - 1)
            if (i < 0) :
                v = v - pixels[frame, x_ - 1, y_ - 1]
            elif (i > 0) :
                v = v + pixels[frame, x_ - 1, y_ -1]
        g = v
        v = 0
        for j in [-1, 0, 1] :
            x_ = min(max(x , 1), width - 1)
            y_ = min(max(y + j, 1), height - 1)
            if (j < 0) :
              v = v - pixels[frame, x_ - 1, y_ - 1]
            elif (j > 0) :
              v = v + pixels[frame, x_ - 1, y_- 1]
        g = np.append(g,v)
        return g

    #### Catmull-Rom interpolation
    def cubic(self, x):
        a = 0.5
        if (x < 0) :
            x = -x
        z = 0
        if x < 1 :
            z = x * x * (x * (-a + 2) + (a - 3)) + 1
        elif(x < 2) :
            z = -a * x * x * x + 5 * a * x * x - 8 * a * x + 4 * a
        return z

    #### interpolateGxGy
    def interpolateGxGy(self, x, y, G, isGx, width, height, magnification):
        x = x / magnification
        y = y / magnification
        if ((x<1.5) | (x>width-1.5) | (y<1.5) | (y>height-1.5)) :
            return 0
        if (isGx) :
            u0 = math.floor(x - 0.5)
            v0 = math.floor(y - 0.5)
            q = 0
            for j in [0,1,2,3] :
                v = v0 - 1 + j
                p = 0
               # print(v)
                for i in [0,1,2,3] :
                    u = u0 - 1 + i
                    p = p +  G[0,u - 1,v - 1] * self.cubic(x - (u + 0.5))
                q = q + p * self.cubic(y - (v + 0.5))
            return q
        else :
            u0 = math.floor(x - 0.5)
            v0 = math.floor(y - 0.5)
            q = 0
            for j in [0,1,2,3] :
                v = v0 - 1 + j
                p = 0
                for i in [0,1,2,3] :
                    u = u0 - 1 + i
                    p = p + G[1,u - 1,v - 1] * self.cubic(x - (u + 0.5))
                q = q + p * self.cubic(y - (v + 0.5))
            return q

    ##### calcRadiality
    def calcRadiality(self, X, Y, G, width, height, magnification, shiftX, shiftY, RingCoordinates, nRingCoordinates, spatialRadius):
        radialityPositivityConstraint = True
        Xc = X + shiftX*magnification
        Yc = Y + shiftY*magnification
        x0 = Xc + RingCoordinates[0,:]
        y0 = Yc + RingCoordinates[1,:]
        Gx = np.zeros(nRingCoordinates)
        Gy = np.zeros(nRingCoordinates)
        GdotR =  np.zeros(nRingCoordinates)
        Gmag =  np.zeros(nRingCoordinates)
        for i in list(range(0, nRingCoordinates)):
            Gx[i] = self.interpolateGxGy(x0[i], y0[i], G, True, width, height, magnification)   # Gx
            Gy[i] = self.interpolateGxGy(x0[i], y0[i], G, False, width, height, magnification)  # Gy
            Gmag[i] = np.sqrt( Gx[i]**2 + Gy[i]**2)  # Gmag
            if spatialRadius == 0:
                spatialRadius = 0.01
            Gmag[Gmag==0] = 0.01
            GdotR[i]= np.dot([Gx[i],Gy[i]], RingCoordinates[:,i]) / (Gmag[i]*spatialRadius) # GdotR
        Dk = 1 - (abs(Gy * (Xc - x0) - Gx * (Yc - y0)) / Gmag) / spatialRadius
        Dk = -np.sign(GdotR)*(Dk**2)
        DivDFactor = np.sum(Dk)/nRingCoordinates
        if (radialityPositivityConstraint == True) :
            CGH = max(DivDFactor, 0)
        else :
            CGH = DivDFactor
        return CGH

    def singleFrameRadialityMap(self, pixels, magnification, spatialRadius, symmetryAxis, frame):
        #### parameters
        ParallelGRadius = 1
        PerpendicularGRadius = 0
        border = 1
        shiftX = 1
        shiftY = 1
        nRingCoordinates = symmetryAxis * 2
        angleStep = (np.pi * 2) /  nRingCoordinates
        nFrames, width, height = pixels.shape
        widthM = width * magnification
        heightM = height * magnification
        borderM = border * magnification
        widthMBorderless = widthM - borderM * 2
        heightMBorderless = heightM - borderM * 2
        ######
        RingCoordinates = self.buildRing(nRingCoordinates, spatialRadius, angleStep)
        G = np.zeros((2, width, height))
        for x in progress(list(range(1, width + 1))) :
            for y in list(range(1, height + 1)) :
                g = self.calculateGxGy(pixels,frame,x, y, height, width)
                G[0,x - 1,y - 1] = g[0]
                G[1,x - 1,y - 1] = g[1]
        RM = np.zeros((width*magnification, height*magnification))
        for X in list(range(borderM  * 2, widthMBorderless + 1)) :
            for Y in list(range(borderM  * 2, heightMBorderless + 1)) :
                RM[X - 1,Y - 1] = self.calcRadiality(X, Y, G, width, height, magnification, shiftX, shiftY, RingCoordinates, nRingCoordinates, spatialRadius)
           # print(X)
        RM[np.isnan(RM)]=0
        grid_x, grid_y =  np.mgrid[0:width:width*magnification*1j,0:height:height*magnification*1j]
        count = 0
        points = np.zeros((width*height, 2))
        #for i in np.arange(0, width, 1):
        for i in range(width):
            #for j in np.arange(0, height, 1):
            for j in range(height):
                points[count, 0] = i
                points[count, 1] = j
                count = count + 1
        values = pixels[frame].flatten()
        #grid_z0 = griddata(points, values, (grid_x, grid_y), method='nearest')
        grid_z2 = griddata(points, values, (grid_x, grid_y), method='cubic')
        RMn = RM*grid_z2
        return  RMn

    def srrf(self, img_layer, magnification, spatialRadius, symmetryAxis, fstart, fend) :
        img=np.array(img_layer.data)
        n, w, h, = img.shape
        #imag = np.zeros((fend - fstart, w*magnification, h*magnification))
        irm = np.zeros((fend - fstart, w*magnification, h*magnification))
        for i, frame in enumerate(list(range(fstart, fend)) ):
            print(frame)
            irm[i] = self.singleFrameRadialityMap(img, magnification, spatialRadius, symmetryAxis, frame)
        # imagm = imag.mean(axis = 0)
        # srrf = irm.mean(axis = 0)
        #imagm_res = np.mean(imag, axis = 0)
        #srrf = np.mean(irm, axis = 0)
        return irm
