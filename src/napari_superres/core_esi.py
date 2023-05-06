# import napari
# from napari.layers import Image, Labels, Layer, Points
import numpy as np
from skimage import io
from scipy.ndimage import gaussian_filter


class esi_class:
    def __init__(self):
        self.first = "init"
    # def conection_test(self):
    #     print("conected")
##################### Functions #####################
##### buildRing
    # /** Default parameters for the analysis, can be changed in GUI */
    # nrResImages=100;	// Images in result stack
    # nrBins=100;		// #bins/states in entropy calculation
    # esi_order=4;		// ESI order
    # doMultiCore=true;	// run multi-threaded
    # normOutput=true;	// normalize the output images

    def ESI_Analysis(self, stck, pxMin, pxMax, nrBins=100, esi_order=4, nrResImage=10, normOutput=True):
        stck = np.array(stck, dtype=float)
        imgPerResult = int(stck.shape[0]/nrResImage)

        print("ESI: input images per resulting images: "+ str(imgPerResult))
        # stacks to store the results
        rec = np.zeros((nrResImage,2*stck.shape[1],2*stck.shape[2]))
        # image to show the summed-up result
        summedImg = np.zeros((2*stck.shape[1],2*stck.shape[2]))
        # loop over subimages
        for k in range(0,nrResImage):

            #generate and normalize the TraceStack
            trSt = self.getSubvolume(stck, k*imgPerResult, (k+1)*imgPerResult)
            trSt_nor = self.normalizeFrom(trSt, pxMin, pxMax)
            trSt_prob = self.createNormBinning(trSt_nor, nrBins)

            # run the analysis
            #SINGLE CORE --- TO IMPLEMENT MULTICORE
            reconstruction = self.doESI(trSt_prob, trSt_nor, esi_order)
            reconstruction = gaussian_filter(reconstruction, sigma = 0.8)

            if normOutput:
                #print("NORMALIZE REC")
                reconstruction = self.normalize(reconstruction, 0, 1)
            # add these to the result stacks
            rec[k,:,:] = reconstruction
            # add the slice to the current sum
            summedImg = self.addFP( reconstruction, summedImg)

        print("DONE")
        return(rec)


    def addFP(self, reconstruction, summedImg):
        return (summedImg + reconstruction)

    def getSubvolume(self, stck, start, end):
        if start < 0:
            start = 0
        if end > stck.shape[0]:
            end = stck.shape[0]

        depth = end-start
        return (stck[start:start+depth,:,:])

    def normalize(self, stck, low, high):
        min_ = np.amin(stck)
        max_ = np.amax(stck)
        if min_==max_:
            print("WARNING DIVIDE BY ZERO")
        stck = ((stck-min_)/(max_-min_)) *(high-low) + low
        return (stck)

    def normalizeFrom(self, stck, curMin, curMax):
        stck = np.clip(stck,curMin,curMax)
        #stck = normalize(stck,0,1)
        stck = (stck - curMin)/(curMax-curMin)

        return (stck)



    def createNormBinning(self, stck, nrBins):
        #we assume stck is already normalized to [0,1]
        probs = np.zeros((nrBins,stck.shape[1],stck.shape[2]))
        # map our data to the bins
        for z in range(0,stck.shape[0]):
            for x in range(0,stck.shape[1]):
                for y in range(0,stck.shape[2]):
                    idx = int(stck[z,x,y]*nrBins)

                    if (idx < 0): idx = 0
                    if (idx >= nrBins): idx = nrBins-1
                    # counter up for the bin
                    probs[idx, x, y] = probs[idx, x, y]+1
        #normalize probabilities to sum one
        probs = probs/stck.shape[0]
        return(probs)

    def doESI(self, trSt_prob, trSt_nor, order):
        return(self.ESI_internal(trSt_prob, trSt_nor, order))

    def ESI_internal(self, trSt_prob, trSt_nor, order):
        res = np.zeros((2*trSt_prob.shape[1], 2*trSt_prob.shape[2]))

        for y in range(1, trSt_prob.shape[2]-1): #loop lines (start to end for this thread)
            for x in range(1, trSt_prob.shape[1]-1): #loop pixel position in line
                for i in  range(0,2): #loop res improvement offset in x,y
                    for j in range(0,2):
                        # on existing pixel: replace by cross-correlation of the 4
                        # next-neighbor pixels
                        if (i==0 and j==0):
                            tmp = (self.H_cross2(trSt_prob[:,x-1,y-1], trSt_prob[:,x+1,y+1],trSt_nor[:,x-1,y-1], trSt_nor[:,x+1,y+1],order) +
                            self.H_cross2(trSt_prob[:,x-1,y-1], trSt_prob[:,x+1,y+1],trSt_nor[:,x-1,y-1], trSt_nor[:,x+1,y+1],order))
                            res[x*2,y*2] = tmp/2
                            # #Paul's ansatz
                            # tmp = self.H_cross2(trSt_prob[:,x,y], trSt_prob[:,x,y],trSt_nor[:,x,y], trSt_nor[:,x,y],order)
                            # res[x*2,y*2] = tmp
                        # new pixel, but only offset in x or y, not both:
                        # cross correlation between neighbors
                        elif ((i+j) <2):
                            res[x*2+i,y*2+j] = self.H_cross2(trSt_prob[:,x,y], trSt_prob[:,x+i,y+j],trSt_nor[:,x,y], trSt_nor[:,x+i,y+j],order)
                        # new pixel, on diagonal:
                        # averaged cross-correlation
                        else:
                            tmp = (self.H_cross2(trSt_prob[:,x,y], trSt_prob[:,x+i,y+j],trSt_nor[:,x,y], trSt_nor[:,x+i,y+j],order) +
                            self.H_cross2(trSt_prob[:,x+i,y], trSt_prob[:,x,y+j],trSt_nor[:,x+i,y], trSt_nor[:,x,y+j],order))
                            res[x*2+1,y*2+1] = tmp/2
        return (res)

    def H_cross2(self, X_prob, Y_prob, X_nor, Y_nor, order):
        h_sum_temp = 0

        # Loop over binned probability space, see eq. 2
        for i in range(0, X_prob.shape[0]):
            if (Y_prob[i]>0):
                h_sum_temp = h_sum_temp + X_prob[i]*np.log2(Y_prob[i])

            if (X_prob[i]>0):
                h_sum_temp = h_sum_temp + Y_prob[i]*np.log2(X_prob[i])
        # eq. 8, multiply with joined moment
        return ( -h_sum_temp*self.joint_moment(X_nor,Y_nor,order)/2)

    def joint_moment(self, X_nor, Y_nor, order):
        #Calculate the joint n'th moment of two pixel traces
        meanX = self.weightedMean(X_nor, 1)
        meanY = self.weightedMean(Y_nor, 1)

        # create storage
        dummy = np.power(X_nor-meanX,order) * np.power(Y_nor-meanY,order)
        jmom = self.weightedMean(dummy, 1)
        return (jmom)

    def weightedMean(self, data, weight):
        mean=0
        factor = weight/data.shape[0]

        for  i in range(0, data.shape[0]):
            weight = weight + factor
            mean = mean +weight*data[i]
        return (mean/data.shape[0])
