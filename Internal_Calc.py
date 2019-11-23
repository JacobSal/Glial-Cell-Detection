# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 17:59:23 2019

@author: Das Gooche
"""

from scipy.ndimage import convolve,distance_transform_edt
import numpy as np
import __main__
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from os import path,walk,chdir,remove
#import openCV
from os.path import join
from skimage import img_as_float32
from skimage.measure import label
from matplotlib.colors import ListedColormap
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from skimage.morphology import remove_small_objects,watershed,binary_closing
from IPython import get_ipython
import settings
import math
import time
####


class internal_calc:
    
    figfigcount = int()
    objectcount = int()
    
    def __init__(self):
        self.animal_num = int()
        self.slide_num = int()
        self.row_num = int()
        self.colum_num = int()
        self.cut_width = int()
        self.overlap = int()
        self.channel = int()
        self.group_num = int()
        self.jacobsnum = float()
        self.cell_bool = np.array([]).astype("float32")
        self.body_bool = np.array([]).astype("float32")
        self.filt_width = int()
        self.filt_intensity =  float()
        internal_calc.objectcount += 1
        self.objectcount = internal_calc.objectcount-1
    'end def'
    
    def save_to_settings(self):
        settings.internal_storage.append(self)
    
    def save_settings_ic():
        dirpath = path.dirname(__main__.__file__)
        hand = path.abspath(path.join(dirpath,"..","imagedata.dat"))
        with open(hand,'wb') as f:
            pickle.dump(settings.internal_storage,f)
        'end with'
        
    def save_multiple_f(dirpath,objs):
        seps = [0]
        incr = 100
        while seps[-1] <= len(objs):
            seps.append(incr)
            incr = 100 + incr
        'end while'
        seps[-1] = len(objs)
        for i in range(0,len(seps)-1):
            hand = path.abspath(path.join(dirpath,"..","saved_data/imagedata_%d.dat"%(i)))
            with open(hand,'wb') as f:
                pickle.dump(objs[seps[i]:seps[i+1]],f)
            'end with'
            if path.getsize(hand) == 0:
                remove(hand)
        'end for'
        
#    def load_multiple_f(dirpath):
#        seps = [0]
#        incr = 300
#        while seps[-1] <= len(objs):
#            seps.append(incr)
#            incr = 250 + incr
#        'end while'
#        seps[-1] = len(objs)
#        
#        with open(hand,'rb') as f:
#            objs = pickle.load(f)
#            
    def figfig(image,title='combined_contours layers 1-9',savefig=False):
        plt.ion()
        savedir = r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\Detect_A_Cell\Saved_Figs\Alg_figs"
#        internal_calc.figfigcount += 1
        plt.figure(title)
        plt.imshow(image)
        if savefig == True:
            image = (image*255).astype('uint8')
            binary_img = Image.fromarray(image)
            binary_img.save(path.join(savedir,title+".tif"))
        'end if'
        plt.draw()
        plt.show()
    'end def'
    
    def figfig_cont(contours, title = "default", savefig = False):
        newim = np.zeros((contours[0].shape[0],contours[0].shape[1]))
        for i in range(0,len(contours)):
            newim = np.add(np.multiply(contours[i],i+1),newim)
        'end for'
        plt.ion()
        plt.figure(title)
        plt.imshow(newim);plt.draw();plt.show()
        savedir = r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\Detect_A_Cell\Saved_Figs\Alg_figs"
        if savefig == True:
            plt.savefig(join(savedir,title+".tif"),dpi=600,quality=95)
        'end if'
    'end def'
    
        
        
    def imshow_overlay(im, mask, savefig=False, genfig=True, alpha=0.2, color='red', **kwargs):
        """Show semi-transparent red mask over an image"""
        mask = mask > 0
        mask = np.ma.masked_where(~mask, mask)
        if genfig == True:
            plt.figure('overlayed')  
            plt.imshow(im, **kwargs)
            plt.imshow(mask, alpha=alpha, cmap=ListedColormap([color]))
        'end if'
        
        if savefig == True:
            savedir = r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\Detect_A_Cell\Saved_Figs\Alg_figs"
            plt.savefig(join(savedir,"overlayed"+".tif"),dpi=600,quality=95,pad_inches=0)
        'end if'
    'end def'
    
    def imshow_overlay2(im, mask,mask2, savefig=False, genfig=True, alpha=0.2, color='red', **kwargs):
        """Show semi-transparent red mask over an image"""
        mask = mask > 0
        mask = np.ma.masked_where(~mask, mask)
        mask2 = mask2 >0
        mask2 = np.ma.masked_where(~mask2, mask2)
        if genfig == True:
            plt.figure('overlayed2')  
            plt.imshow(im*2, cmap='gray',**kwargs)
            plt.imshow(mask, alpha=alpha, cmap=ListedColormap([color]))
            plt.imshow(mask2, alpha=0.2, cmap=ListedColormap(['yellow']))
        'end if'
        
        if savefig == True:
            savedir = r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\Detect_A_Cell\Saved_Figs\Alg_figs"
            plt.savefig(join(savedir,"overlayed"+".tif"),dpi=600,quality=95,pad_inches=0)
        'end if'
    'end def'
    
    def show_combinefig(self,j,channel,wait=False):
        plt.ion()
        imi = settings.images[j]
#        imi = imi[:,:,channel]
#        fig2 = plt.figure("Press this to continue")
        plt.show()
        fig = plt.figure('slide '+str(self.slide_num)+' animal '+str(self.row_num)+' number of cells '+str(self.jacobsnum))
        fig.suptitle('pixel density: ' + str(self.jacobsnum),fontsize=12)
        plt.subplot(2,3,1)
        plt.title('Orignial', fontsize=10)
        plt.imshow(imi)
        plt.axis('off')
        plt.subplot(2,3,2)
        plt.title('Cells', fontsize=10)
        plt.imshow(self.cell_bool)
        plt.axis('off')
        plt.subplot(2,3,3)
        plt.title('Body', fontsize=10)
        plt.imshow(self.body_bool)
        plt.axis('off')
        plt.subplot(2,3,5)
        plt.title('Overlayed', fontsize=10)
        plt.imshow(imi)
        plt.imshow(internal_calc.imshow_overlay(imi,self.cell_bool,False,self.cell_bool),alpha=0.2,cmap=ListedColormap(['red']))
        plt.axis('off')
        if wait == True:
            fig.waitforbuttonpress()
        'end if'
        plt.close()
        return fig
    
    def _change_channel(self,channel):
        if channel>2 :
            raise ValueError('channel must be 0, 1, or 2')
        'end if'
        try:
            settings.images[self.objectcount] = settings.images[self.objectcount][:,:,channel]
            self.channel = channel
        except IndexError:
            self.channel = channel
        'end if'
    'end def'
    
    def load_image(rootdir=r"",channel=1):
        im = plt.imread(rootdir)
        try: im.shape[2]
        except IndexError:
            im = img_as_float32(im)
        else:
            im = img_as_float32(im[:,:,channel])
        im[im==0] = "nan"
        im[im==1] = np.nanmin(im)
        im[np.isnan(im)] = np.nanmin(im)
        return im
    
    def load_file(rootdir=r""):
        """loads file into objects that can be passed to internal_calc
        removes stiching errors where image == 1"""
        for root, dirs, files in walk(rootdir):
            for f in files:
                impath = join(root,f)
                im = internal_calc.load_image(impath)
                settings.images.append(im)
                settings.filename.append(f)
            'end for'
        'end for'
    'end def'
        
    def _diffmat(im_array,theta,dim=(10,2)):
        if type(dim) != tuple:
            raise ValueError('dim must be tuple')
        if dim[1]%2 != 0:
            raise ValueError('n must be even')
        'end if'
        outarray = np.zeros((im_array.shape[0],im_array.shape[1]))
        dfmat = np.zeros((max(dim),max(dim)))
        dfmat[:,0:int(dim[1]/2)] = -1
        dfmat[:,int(dim[1]/2):dim[1]] = 1
        
        dmatx = dfmat
        dmaty = np.transpose(dfmat)
        for angle in theta:
            dmat = dmatx*np.cos(angle)+dmaty*np.sin(angle)
            dm = np.divide(convolve(im_array,dmat)**2,math.factorial(max(dim)))
            outarray = np.add(dm,outarray)
        return outarray

        
    def edges(im_array):
        boolarray = np.zeros((im_array.shape[0],im_array.shape[1]))
        dmatx = np.array([[1,-1],[0,0]])
        dmaty = np.array([[1,0],[-1,0]])
        dmatcox = convolve(im_array,dmatx)
        dmatcoy = convolve(im_array,dmaty)
        boolarray = np.add(abs(dmatcox),abs(dmatcoy))
            
        return boolarray
    def d3gaussian(vector_width,multiplier_a,multiplier_d):
        """
        creates a 3D gaussian inlayed into matrix form.
        of the form: G(x,y) = a*e^(-((x-N)**2+(y-N)**2)/(2*d**2))
        """
        x = np.arange(0,vector_width,1)
        y = np.arange(0,vector_width,1)
        d2gau = np.zeros((vector_width,vector_width)).astype(float)
        N = int(vector_width/2)
        for i in x:
            d2gau[:,i] = multiplier_a*np.exp(-((i-N)**2+(y-N)**2)/(2*multiplier_d**2))
        'end for'
        return d2gau     
    'end def'
    
    def onediv_d3gaussian(vector_width,multiplier_a,multiplier_d):
        """
        creates a 3D gaussian inlayed into matrix form.
        of the form: G(x,y) = a*e^(-((x-N)**2+(y-N)**2)/(2*d**2))
        """
        x = np.arange(0,vector_width,1)
        y = np.arange(0,vector_width,1)
        d2gau_x = np.zeros((vector_width,vector_width)).astype(float)
        d2gau_y = np.zeros((vector_width,vector_width)).astype(float)
        N = int(vector_width/2)
        for i in x:
            d2gau_x[:,i] = -((i-N)/multiplier_d**2)*multiplier_a*np.exp(-((i-N)**2+(y-N)**2)/(2*multiplier_d**2))
            d2gau_y[:,i] = -((y-N)/multiplier_d**2)*multiplier_a*np.exp(-((i-N)**2+(y-N)**2)/(2*multiplier_d**2))
        'end for'
        d2gau = np.add(d2gau_x**2,d2gau_y**2)
        return d2gau     
    'end def'
    
    def MatchFilter(array,vector,Dim=1,axis = 0):
        """takes a vector and convolutes it over an array on a axis, multidimensional
        if dim == 1 then need a 1d vector if dim == 2 need a 2d mat"""
        if Dim == 1:
            gfilt = np.zeros((array.shape[0],array.shape[1])).astype('float32')
            for j in range(0,array.shape[axis]):
                if axis == 0:
                    gfilt[j,:] = np.correlate(array[j,:],vector,mode='same')
                elif axis == 1:
                    gfilt[:,j] = np.correlate(array[:,j],vector,mode='same')
                else:
                    print('choose an axis dumbass')
            'end for'
        elif Dim == 2:
            gfilt = convolve(array,vector)
        return gfilt
    'end def'
    
    def _FourierFilt(array,red_tup,reduction_factor,mode="tuple"):
        """narrow band filter based on the concept of minimizing coefficients
        in fourier transform"""
        '''
        if array.shape[0]%2 == 0 :
            array = array[0:array.shape[0]-1,:]
        elif array.shape[1]%2 == 0:
            array = array[:,0:array.shape[1]-1]
        'end if'
#        circ_filter = np.zeros((array.shape[0],array.shape[1])).astype(bool)
#        circle_bool = internal_calc._create_circle(radius,0,True)
#        circ_filter[array.shape[0]//2-circle_bool.shape[0]//2:(array.shape[0]//2)+1+circle_bool.shape[0]//2,
#                    array.shape[1]//2-circle_bool.shape[1]//2:(array.shape[1]//2)+1+circle_bool.shape[1]//2] = circle_bool
        gfilt = np.zeros((array.shape[0],array.shape[1]))
        gfft = np.fft.fftshift(np.fft.fft(array))
        gfft[abs(gfft)<cutoff] = 0 
        gifft = abs(np.fft.ifft(gfft))
        gfilt = np.multiply(gifft,array)
        return gfilt
        '''      
        
        if mode == "tuple":
            began = int(array.shape[0]/2)-int(red_tup[0]/2)
            end = int(array.shape[0]/2)+int(red_tup[0]/2)
            gfilt = np.zeros((array.shape[0],array.shape[1]))
            gfft = np.fft.fftshift(np.fft.fft(array))           
            gfft[:,:began] = reduction_factor*gfft[:,:began]
            gfft[:,end:] = reduction_factor*gfft[:,end:]
            
            began = int(array.shape[1]/2)-int(red_tup[1]/2)
            end = int(array.shape[1]/2)+int(red_tup[1]/2)
            gfft[:began,:] = reduction_factor*gfft[:began,:]
            gfft[end:,:] = reduction_factor*gfft[end:,:]
            gfilt = abs(np.fft.ifft(np.fft.fftshift(gfft)))
            greturn = np.multiply(gfilt,array)
        elif mode == '2D':
            began = int(array.shape[1]/2)-int(red_tup[1]/2)
            end = int(array.shape[1]/2)+int(red_tup[1]/2)
            gfilt = np.zeros((array.shape[0],array.shape[1]))
            gfft = np.fft.fftshift(np.fft.fft(array))  
            gfft[:,:began] = reduction_factor*gfft[:,:began]
            gfft[:,end:] = reduction_factor*gfft[:,end:] 
            gfilt = abs(np.fft.ifft(np.fft.fftshift(gfft)))
            gfilt = abs(np.fft.ifft(np.fft.fftshift(gfft)))
            greturn = np.multiply(gfilt,array)
        elif type(mode) == int:
            try: shape3 = array.shape[2]
            except IndexError:
                shape3 = 1
            'end try'
            for i in range(0,shape3):
                if mode == 0:
                    began = int(array.shape[0]/2)-int(red_tup[0]/2)
                    end = int(array.shape[0]/2)+int(red_tup[0]/2)
                    gfilt = np.zeros((array.shape[0],array.shape[1]))
                    gfilt_vstack = np.zeros((array.shape[0],array.shape[1]))
                    gfft = np.fft.fftshift(np.fft.fft(array))           
                    gfft[:,:began,i] = reduction_factor*gfft[:,:began,i]
                    gfft[:,end:,i] = reduction_factor*gfft[:,end:,i]
                    gfilt = abs(np.fft.ifft(np.fft.fftshift(gfft)))
                    gfilt_dstack = np.dstack((gfilt_vstack,gfilt))
                    greturn = np.multiply(gfilt_dstack[:,:,1:int(array.shape[2]+1)],array)
                if mode == 1:
                    began = int(array.shape[1]/2)-int(red_tup[1]/2)
                    end = int(array.shape[1]/2)+int(red_tup[1]/2)
                    gfilt = np.zeros((array.shape[0],array.shape[1]))
                    gfilt_vstack = np.zeros((array.shape[0],array.shape[1]))
                    gfft[:began,:,i] = reduction_factor*gfft[:began,:,i]
                    gfft[end:,:,i] = reduction_factor*gfft[end:,:,i]
                    gfilt = abs(np.fft.ifft(np.fft.fftshift(gfft)))
                    gfilt = abs(np.fft.ifft(np.fft.fftshift(gfft)))
                    gfilt_dstack = np.dstack((gfilt_vstack,gfilt))
                    greturn = np.multiply(gfilt_dstack[:,:,1:int(array.shape[2]+1)],array)
                    
                'end if'
        else:
            raise Exception("mode must be either selected as int or 'tuple'")
        return greturn
    'end def'
    
    def _fourier_exp(array,n):
        """
        takes an array and uses fourier interpolation to expand the image resolution by n times 
        """
        if array.shape[0]%2 != 0 or array.shape[1]%2 != 0:
            array = array[0:array.shape[0]-(array.shape[0]%2),0:array.shape[1]-(array.shape[1]%2)]
        'end if'
        shap0 = array.shape[0]
        shap1 = array.shape[1]
        newarray = np.zeros((shap0*n,shap1*n))
        fftarray = np.fft.fft(array)
        newarray[int(n*shap0/2-shap0/2):int(n*shap0/2+shap0/2),int(n*shap1/2-shap1/2):int(n*shap1/2+shap1/2)] = fftarray
        newarray = np.fft.ifft(newarray).astype(float)
        
    
    def _create_circle(radius,angle,fill=False):
        """creates a circle of radius and segments it by angle
        to be used in curve detection
        radius = int (must be odd)
        angle = float (radians)"""
        if radius%2 == 0:
            raise ValueError("Radius must be odd")
        'end if'
        circle = np.zeros((radius,radius))
        N = int(radius/2)
        x_val = list(range(0,radius))
        y_val = list(range(0,radius))
        for x in x_val:
            for y in y_val:
                circle[int(x),int(y)] = np.sqrt((x-N)**2+(y-N)**2)
            'end for'
        'end for'
        circle_bool = np.logical_not(np.add(circle>(radius/2),circle<(radius-2)/2))
#        rowcount = 0
        if fill == True:
            circle_bool = circle<(radius/2)
            '''
            for row in circle_bool:
                edges = [i for i,x in enumerate(row) if x == True]
                if int(np.unique(np.diff(edges))) == 1:
                    pass
                else:
                for i in range(len(row)-1):
                    if row[i] == True and row[i+1] == False:
                        circle_bool[rowcount,i+1] = True
                    elif row[i] == True and row[i+1] == True:
                        break
                    'end if'
                'end for'
                rowcount += 1
                'end for'
            'end for'
            '''
        'end if'
        return circle_bool
    
    def Hough_Transform():
        """blank"""
        
    'end def'
    
    
    def range_cont(array,layers,maxlayers,shift = 0,reverse = True):
        """
        generates a list of "thresholds" to be used for _contours.
        uses the max and min of the array as bounds and the number of layers as
        segmentation.
        array = np.array()
        layers = number of differentials (int)
        maxlayers = number of layers kept (int)
        shfit = a variable that shifts the sampling up and down based on maxlayers (int)
        reverse = start at the min or end at max of the array (True == end at max) (bool)
        """
        arraymax = np.max(array)+(np.max(array)/layers)
        arraymin = np.min(array)
        incr = arraymax/layers
        maxincr = arraymin+(incr*(maxlayers))
        if reverse == True:
            thresh = [arraymax-maxincr-shift*(incr*(maxlayers+1))]
            arraystart = arraymax-maxincr - shift*(incr*(maxlayers+1))
            maxincr = arraymax - shift*(incr*(maxlayers+1))
            
        elif reverse == False:
            thresh = [arraymin + shift*(incr*(maxlayers))]
            arraystart = arraymin + shift*(incr*(maxlayers))
            if shift > 0:
                maxincr = arraystart + shift*(incr*(maxlayers-1))
            'end if'
        try:
            while thresh[-1] < maxincr:
                thresh.append(np.float32(arraystart+incr))
                incr = incr + arraymax/layers
        except:
            raise ValueError("Max and Min == 0")
        'end try'
        return thresh
            
    
    def _contours(array,thresh):
        """takes an array and creates a topographical map based on layers and maxlayers. Useful in creating
        differential sections of images"""
        contours = [None]*(len(thresh))
        sums = [None]*(len(thresh))
        avgweight= [None]*(len(thresh))
        avgnan= [None]*(len(thresh)) 
        count = 0
        maxlayers = len(thresh)
        #############
        contours[count] = np.logical_not(np.add(array<thresh[count],array>thresh[count+1]))
        sums[count] = np.sum(contours[count])
        overim = contours[count]*array
        avgweight[count] = np.average(overim)
        overim[overim == 0] = 'nan'
        avgnan[count] = np.nanmean(overim)
        while count < maxlayers-2:
            count += 1
            contours[count] = np.logical_not(np.add(array<thresh[count],array>thresh[count+1]))
            sums[count] = np.sum(contours[count])
            overim = contours[count]*array
            avgweight[count] = np.average(overim)
            overim[overim == 0] = 'nan'
            avgnan[count] = np.nanmean(overim)
        'end while'
        sums = sums[0:count]
        contours = contours[0:count]
        avgnan = avgnan[0:count]
        avgweight = avgweight[0:count]
        return [contours,sums,avgnan,avgweight]
    'end def'
    
    def centers(array,circle):
        """
        inserts a circle for every True element in an array inspired by the Hough-Transform.
        """
#        circle = np.array([[0,0,1,1,1,0,0],[0,1,0,0,0,1,0],[1,0,0,0,0,0,1],[1,0,0,0,0,0,1],[1,0,0,0,0,0,1],[0,1,0,0,0,1,0],[0,0,1,1,1,0,0]])
        r = circle.shape[0]
        arraynew = np.zeros((array.shape[0],array.shape[1]))
        for i in range(0,array.shape[0]):
            for j in range(0,array.shape[1]):
                if array[i,j] == 1:
                    if r%2 != 0:
                        ex = 1
                    else:
                        ex = 0
                    if i >= r and array.shape[0]-i > int(r/2)+ex:
                        if j >= r and array.shape[1]-j > r:
                            added = np.add(circle[:,:],array[i-(int(r/2)+ex):i+(int(r/2)),j-(int(r/2)+ex):j+int(r/2)])
                            arraynew[i-(int(r/2)+ex):i+(int(r/2)),j-(int(r/2)+ex):j+int(r/2)] = np.add(added,arraynew[i-(int(r/2)+ex):i+(int(r/2)),j-(int(r/2)+ex):j+int(r/2)])
                        elif j >= r and array.shape[1]-j < int(r/2)+ex:
                            added = np.add(circle[:,0:(array.shape[1]-j)+(int(r/2)+ex)],array[i-(int(r/2)+ex):i+(int(r/2)),j-(int(r/2)+ex):array.shape[1]])
                            arraynew[i-(int(r/2)+ex):i+(int(r/2)),j-(int(r/2)+ex):array.shape[1]] = np.add(added,arraynew[i-(int(r/2)+ex):i+(int(r/2)),j-(int(r/2)+ex):array.shape[1]])
                        elif j < int(r/2)+ex:
                            added = np.add(circle[:,int(r/2)-j:r],array[i-(int(r/2)+ex):i+(int(r/2)),0:j+int(r/2)+ex])
                            arraynew[i-(int(r/2)+ex):i+(int(r/2)),0:j+int(r/2)+ex] = np.add(added,arraynew[i-(int(r/2)+ex):i+(int(r/2)),0:j+int(r/2)+ex])
                            
                    elif i >= r and array.shape[0]-i < int(r/2)+ex:
                        if j >= r and array.shape[1]-j > r:
                            added = np.add(circle[0:(array.shape[0]-i)+int(r/2)+ex,:],array[i-(int(r/2)+ex):array.shape[0],j-(int(r/2)+ex):j+int(r/2)])
                            arraynew[i-(int(r/2)+ex):array.shape[0],j-(int(r/2)+ex):j+int(r/2)] = np.add(added,arraynew[i-(int(r/2)+ex):array.shape[0],j-(int(r/2)+ex):j+int(r/2)])
                        elif j >= r and array.shape[1]-j < int(r/2)+ex:
                            added = np.add(circle[0:(array.shape[0]-i)+int(r/2)+ex,0:(array.shape[1]-j)+(int(r/2)+ex)],array[i-(int(r/2)+ex):array.shape[0],j-(int(r/2)+ex):array.shape[1]])
                            arraynew[i-(int(r/2)+ex):array.shape[0],j-(int(r/2)+ex):array.shape[1]] = np.add(added,arraynew[i-(int(r/2)+ex):array.shape[0],j-(int(r/2)+ex):array.shape[1]])
                        elif j < int(r/2)+ex:
                            added = np.add(circle[0:(array.shape[0]-i)+int(r/2)+ex,int(r/2)-j:r],array[i-(int(r/2)+ex):array.shape[0],0:j+int(r/2)+ex])
                            arraynew[i-(int(r/2)+ex):array.shape[0],0:j+int(r/2)+ex] = np.add(added,arraynew[i-(int(r/2)+ex):array.shape[0],0:j+int(r/2)+ex])
                        
                    elif i < int(r/2)+ex:
                        if j >= r and array.shape[1]-j > r:
                            added = np.add(circle[int(r/2)-i:r,:],array[0:i+int((r/2)+ex),j-(int(r/2)+ex):j+int(r/2)])
                            arraynew[0:i+int((r/2)+ex),j-(int(r/2)+ex):j+int(r/2)] = np.add(added,arraynew[0:i+int((r/2)+ex),j-(int(r/2)+ex):j+int(r/2)] )
                        elif j >= r and array.shape[1]-j < int(r/2)+ex:
                            added = np.add(circle[int(r/2)-i:r,0:(array.shape[1]-j)+(int(r/2)+ex)],array[0:i+int((r/2)+ex),j-(int(r/2)+ex):array.shape[1]])
                            arraynew[0:i+int((r/2)+ex),j-(int(r/2)+ex):array.shape[1]] = np.add(added,arraynew[0:i+int((r/2)+ex),j-(int(r/2)+ex):array.shape[1]])
                        elif j < int(r/2)+ex:
                            added = np.add(circle[int(r/2)-i:r,int(r/2)-j:r],array[0:i+int((r/2)+ex),0:j+(int(r/2)+ex)])
                            arraynew[0:i+int((r/2)+ex),0:j+(int(r/2)+ex)] = np.add(added,arraynew[0:i+int((r/2)+ex),0:j+(int(r/2)+ex)])
                        
                    
                    'end if'
                else:
                    continue
                'end if'
            'end for'
        'end for'
        return arraynew
    

    def add_layers(contours,shape):
        added = np.zeros((shape[0],shape[1]))
        for i in contours:
            added = np.add(added,i)
        'end for'
        return added
    'end def'
    
    def cell_process(self,width,intensity,diffwidth,layers,shift=0,store_filtered_img=False):
        """finds cells in an image:
            width:int, width of the fourier filter built into method
            intensity: int, intensity of reduction, most reduction being intenstiy = 0
            store_filtered_img: boolean, turns on or off the feature of storing the filtered
            image in the object as an attribute"""
        self.filt_width = width
        self.filt_intensity = intensity
        im = settings.images[self.objectcount]
        
#        while 
        #R1 images being under thresholded everything else looks okay
        filtim_four = internal_calc._FourierFilt(im,width,intensity,'2D')
#        filtim = filtim_four > threshold_otsu(filtim_four)
#        filtim = filtim*filtim_four
        if store_filtered_img == True:
#            self.filt_img = filtim
            self.filt_img_four = filtim_four
        'end if'
        
        diff_filt = internal_calc._diffmat(filtim_four,np.arange(0,2*np.pi,(2*np.pi)/12),diffwidth)
        self.diff_filt = diff_filt
#        diff_filt = diff_filt > threshold_otsu(diff_filt)
        #dgaus seems to make borders wide on detected ceils so recurssion may help with tighetening borderspace, butt this
        #would also increase processing time, so way consequence. 
        dgaus = internal_calc.onediv_d3gaussian(7,1.5,1.5)
        dgaus_diff_filt = convolve(diff_filt,dgaus)
        self.dgaus_diff_filt = dgaus_diff_filt
        thresh = internal_calc.range_cont(dgaus_diff_filt,layers,10,shift,False)
        print(thresh)
        self.thresh = thresh
        [dgaus_contour,sums,_,_] = internal_calc._contours(dgaus_diff_filt,thresh)
        self.dgaus_contour = dgaus_contour
        negative = np.logical_not(dgaus_contour[0])
        closed = binary_closing(negative)
        added = remove_small_objects(closed,min_size=20)
        self.cell_bool = added
#        dist = distance_transform_edt(negative)
#        peaks = peak_local_max(dist,indices=False)
        
        
#        self.dist = dist
#        markers = label(negative)
#        self.markers = markers
#        watershed_img = watershed(negative,markers)
        
#        count = len(np.unique(watershed_img))
        
#        newim = im*dgaus_contour[0]
        
        
        #None Gaussian filt
        '''
        radii = list(range(5,20,2))
        center_finder = [None]*len(radii)*len(dgaus_contour)
        count = 0
        for i in range(len(dgaus_contour)):
            for rad in radii:
                circ = internal_calc._create_circle(rad,0)
                center_finder[count] = internal_calc.centers(dgaus_contour[i],circ)
                count += 1
            'end for'
        'end for'
        '''
#        added = center_finder
        
        
        #add recursion that watersheds the cells and cross compares the detected waveforms. Markov sequencing?
#        self.cell_bool = added
    'end def'
    
    def _number_jacobs(self):
        self.jacobsnum = np.sum(self.cell_bool)/np.sum(self.body_bool)
        return self.jacobsnum
    'end def'
    
    def fill_img(array):
        #isolate main body based on 
        markers = label(array)
        labels = np.unique(markers)
        sums = []
        for number in labels:
            sums.append(np.sum(markers == number))
        'end for'
#        sums = sums[1:]
        max_ind = [i for i,x in enumerate(sums == np.max(sums)) if x == True][0]
        markers = (markers == max_ind)
        #fill image
        for indexr, row in enumerate(markers):
            vals = [i for i,x in enumerate(row) if x == True]
            try:
                row_max = np.max(vals)
                row_min = np.min(vals)
                markers[indexr,row_min:row_max] = [True]*int(row_max-row_min)
            except ValueError:
                continue            
        'end for'
        return markers
    'end def'
    
    def remove_other(im_array):
        labeled = label(im_array)
        layers = np.unique(labeled)
        boolesum = [None]*(len(layers))
        count = 0
        for layer in layers:
            boolesum[count] = np.sum(labeled == layer)
            count+=1
        'end for'
        boolesum = boolesum[1:]
        for layer in layers:
            if np.sum(labeled == layer) < np.max(boolesum):
                labeled[labeled == layer] = 0
            'end if'
        'end for'
        return labeled
    'end def'
    
    def back_process(self):
        #find tissue Body
        circle = np.array([[0,0,1,1,0,0],[0,1,1,1,1,0],[1,1,1,1,1,1],[1,1,1,1,1,1],[0,1,1,1,1,0],[0,0,1,1,0,0]])
        filtim = internal_calc._FourierFilt(settings.images[self.objectcount],(500,1000),0,'2D')
        self.body_for_filt = filtim
        flatimi = internal_calc._diffmat(filtim,np.arange(0,np.pi,(np.pi)/12),(10,10))
        self.body_diff_filt = flatimi
        convolved = convolve(flatimi,internal_calc.onediv_d3gaussian(100,1,1))
        self.body_gaus_filt = convolved
        thresh = internal_calc.range_cont(convolved,10_100_000,10,0,False)
        [contours,sums,avgnan,avgweight] = internal_calc._contours(convolved,thresh)
        self.body_dgaus_contour = contours
        count = 0
        for count in range(0,len(sums)):
            if np.sum(sums[count:]) < 800_000: #and abs(avgnan[count+1]-avgnan[count])/avgnan[count+1] > 0.5:
                threshfin = thresh[count]
                break
            else: 
                continue
            'end if'
        'end while'
        backim = flatimi>threshfin
        
        #Fill image
        backim = binary_closing(backim,circle)
        self.body_close1 = backim
        backadd = np.divide(np.add(backim,settings.images[self.objectcount]),settings.images[self.objectcount])
        backadd[backadd == 1] = np.max(backadd)
        self.body_backadd2 = backadd
        thresh = internal_calc.range_cont(backadd,100,20,0,False)
        [contours,sums,avgnan,avgweight] = internal_calc._contours(backadd,thresh)
        
        count = 0
        self.body_dgaus_contour2 = contours
        for i in range(0,len(sums)):
            if np.sum(sums[:i]) > (settings.images[self.objectcount].shape[0]*settings.images[self.objectcount].shape[1])/2:
                break
            else:
                continue
        'end for'
        backadd = backadd < thresh[i]
        backadd = remove_small_objects(backadd,min_size=40)
        backadd = binary_closing(backadd,circle)
        self.body_closedandre = backadd
        backaddfin = internal_calc.remove_other(backadd)
        self.body_bool = backaddfin
        return
    'end def'
    def remove_straight_lines():
        """removes stitching artifacts based on criteria that 1.) is a straight line of 1's 2.) is bordered almost immediately by 0's"""
    
    def _count_cells(self):
        markers = label(self.cell_bool)
        watershed_img = watershed(self.cell_bool,markers)
        count = len(np.unique(watershed_img))
        self.cellcount = count
        return count
    
    def convert_bw(savepath=r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\unet-master\unet-master\data\membrane\train\image",
                   impath=r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\unet-master\unet-master\data\membrane\train\image_872019"):
        count=0
        for root, dirs, files in walk(impath):
            for f in files:
                im= Image.open(join(root,f))
                im.convert('L')
                im.save(join(savepath,"%d.png"%count))
                count += 1 
    'end def'
    
    def save_binaries(rootdir=r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\Detect_A_Cell",
                      datfile = "imagedata.dat",
                      savedir=r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\unet-master\unet-master\data\membrane\train\label"):
        count = 0
        with open(join(rootdir,datfile),'rb') as f:
            objs = pickle.load(f)
        'end with'
        for obj in objs:
            binary_img = Image.fromarray(obj.cell_bool)
            binary_img.save(join(savedir,"%d.png"%count))
            count+=1
        'end for'
    'end def'
    
    def error_btwn(array1,array2,savefig=True):
        shape0err = array1.shape[0]-array2.shape[0]
        if shape0err > 0:
            newcolum = np.zeros((shape0err,array2.shape[1]))
            array2 = np.concatenate((array2,newcolum),axis=0)
        elif shape0err < 0:
            newcolum = np.zeros((abs(shape0err),array1.shape[1]))
            array1 = np.concatenate((array1,newcolum),axis=0)
        else:
            pass
        shape1err = array1.shape[1]-array2.shape[1]
        if shape1err > 0:
            newcolum = np.zeros((array2.shape[0],shape1err))
            array2 = np.concatenate((array2,newcolum),axis=1)
        elif shape1err < 0:
            newcolum = np.zeros((array1.shape[0],abs(shape1err)))
            array1 = np.concatenate((array1,newcolum),axis=1)
        else:
            pass
        overlapimg = np.add(array1*2,array2*1)
        if savefig == True:
            plt.figure('overlapimg')
            overlapimg[overlapimg==0]='nan'
#            overlapimg[overlapimg==4]=5
            plt.imshow(overlapimg,cmap='Paired')
            savedir = r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\Detect_A_Cell\Saved_Figs\Alg_figs"
            plt.savefig(join(savedir,"overlapimg"+".tif"),dpi=600,quality=95,pad_inches=0)
        'end if'
        sumoverlap = np.sum(overlapimg==3)
        sumunet = np.sum(overlapimg==1)
        sumagl = np.sum(overlapimg==2)
        figoverlap = sumoverlap/abs(sumunet+sumagl+sumoverlap)
        print(figoverlap)
        return figoverlap
'end class'

if __name__ == '__main__':
    dirpath = __main__.__file__
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (10,10)
    get_ipython().run_line_magic('matplotlib','qt5')

#    internal_calc.save_multiple_f(dirpath,objs)
#    internal_calc.save_multiple_f(dirpath,objs)
#    chdir(r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\Detect_A_Cell")
#    import settings
#    internal_calc.save_binaries()
#    internal_calc.convert_image()
#    internal_calc.convert_bw()
#    internal_calc.save_image_predict()
#    internal_calc.return_results_CNN(0.28)

    start = time.time()
    "11:17pm 9/7/2019: pretty good detection tested it on 3 subjects and 2 were valid test more also need border tightening procedure"
    settings.init()
    jool = internal_calc()
    settings.images.append(internal_calc.load_image(path.abspath(join(dirpath,"..","..","MergedandStiched/Slide#12_Row#2_Column#8.tif"))))
    jool._change_channel(1)
#    im = settings.images[0]

    
    """using slide "MergedandStiched/Slide#07_Row#3_Column#7.tif" for images """
    jool.cell_process((500,1000),0,(4,4),900,0,True)
    jool.back_process()
    end = time.time()
    print('time to calc: ' + (end-start)*1349/60)
    print(jool._number_jacobs())
    print(jool._count_cells())
    #Cell Figs
#    brightness = 3
#    internal_calc.figfig(settings.images[0],"cell_Originial",True)
#    internal_calc.figfig(jool.filt_img_four*brightness,"cell_FourierFilt",True)
#    internal_calc.figfig(jool.diff_filt*brightness,"cell_diff_filt",True)
#    internal_calc.figfig(jool.dgaus_diff_filt*brightness,"cell_dgaus",True)
#    internal_calc.figfig(np.logical_not(jool.dgaus_contour[0]),"cell_contour",True)
    internal_calc.figfig(jool.cell_bool,"cell_bool",True)
    
    #body Figs
#    internal_calc.figfig(jool.body_for_filt*brightness,"body_FourierFilt",True)
#    internal_calc.figfig(jool.body_diff_filt*5000,"body_diff_filt",True)
#    internal_calc.figfig(jool.body_gaus_filt*5000,"body_dgaus",True)
#    internal_calc.figfig(jool.body_close1,"body_close1",True)
#    internal_calc.figfig(jool.body_backadd2,"body_backadd2",True)
#    internal_calc.figfig(jool.body_closedandre,"body_close2",True)
#    internal_calc.figfig(np.logical_not(jool.dgaus_contour[0]),"body_contour",True)
    internal_calc.figfig(jool.body_bool,"body_bool",True)

#    internal_calc.figfig_cont(jool.dgaus_contour,"contours1t9",True)
#    internal_calc.figfig_cont(jool.body_dgaus_contour,"contoursbody1t9",True)
#    internal_calc.figfig_cont(jool.body_dgaus_contour2,"contours2body1t9",True)
#    internal_calc.imshow_overlay(settings.images[0],jool.cell_bool,True)
    
#    fig = plt.figure('fourier filtered')
#    ax = plt.axes(projection='3d')
#    filtimg = jool.filt_img
    unetdir = r'C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\unet-master\unet-master\data\membrane\saved_predict'
    image = plt.imread(join(unetdir,'330_predict_final.png'))
    internal_calc.figfig(image*10,'unetimg',True)
    internal_calc.figfig(image>=0.0392,'unetimgbool',True)
    internal_calc.imshow_overlay2(settings.images[0],jool.cell_bool,image>=0.0392,True)
    img2 = internal_calc.error_btwn(jool.cell_bool,image>=0.0392)
#    internal_calc.figfig(img2,'overlaps:-2 = alg, 1 = unet',True)
#    sumuneto = np.sum(image>=0.0392)
#    sumaglo = np.sum(jool.cell_bool)
    sumoverlap = np.sum(img2==3)
    sumunet = np.sum(img2==1)
    sumagl = np.sum(img2==2)
    figoverlap = sumoverlap/abs(sumunet+sumagl+sumoverlap)
    print(figoverlap)
#    figunet = abs((sumunet-sumuneto))/(sumuneto)
#    figagl = abs((sumagl-sumaglo))/(sumaglo)
    
'end if'



