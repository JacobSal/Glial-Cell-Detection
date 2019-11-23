# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:25:02 2019

@author: jsalm
"""

from Internal_Calc import internal_calc
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from os import walk,path
from PIL import Image
import numpy as np
import Graphing
import pickle
from IPython import get_ipython
from skimage import img_as_float32
from skimage.morphology import remove_small_objects,binary_closing

def save_image_predict(rootdir=r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\unet-master\unet-master\data\membrane\test",
                           originalimgdir = r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\MergedandStiched",
                           savedir=r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\unet-master\unet-master\data\membrane\saved_predict"):
        count = 0
        countfin = 0
        for root, dirs, files in walk(originalimgdir):
            for f in files:
                im = plt.imread(path.join(root,f))
                width,height,z = im.shape
                redvar = 256
                slicvar = 0
                im_0 = width - width%redvar
                im_1 = height - height%redvar
                subx = int(im_0/redvar)
                suby = int(im_1/redvar)
                newim = np.zeros((im_0,im_1))
                for i in range(0,subx):
                    for j in range(0,suby):
                        subim = plt.imread(path.join(rootdir,"%d_predict.png"%count))
                        subim = subim[slicvar:subim.shape[0]-slicvar,slicvar:subim.shape[1]-slicvar]
                        newim[int((im_0/subx)*i):int((im_0/subx)*(i+1)),int((im_1/suby)*j):int((im_1/suby)*(j+1))] = subim
                        count += 1
                    'end for'
                'end i'
#                newim = (newim*255).astype('uint8')
                newim = (newim*255).astype('uint8')
                binary_img = Image.fromarray(newim)
                binary_img.save(path.join(savedir,"%d_predict_final.png"%countfin))
#                plt.imsave(path.join(savedir,"%d_predict_final.png"%countfin),newim)
#                plt.imshow(newim)
                countfin += 1
            'end for'
        'end for'
                
                        
def convert_image(rootdir, savedir,
                  channel=1):
    
   
    count = 0
    for root, dirs, files in walk(rootdir):
        for f in files:
            impath = path.join(root,f)
            im = plt.imread(impath)
            if len(im.shape) > 2:
                im = im[:,:,channel]
            'end if'
            width = im.shape[0]
            height = im.shape[1]
            im_0 = width - width%256
            im_1 = height - height%256
            im = im[:im_0,:im_1]
            subx = int(im_0/256)
            suby = int(im_1/256)
            if im_0%subx != 0 or im_1%suby != 0:
                raise ValueError("%d and %d must be factors of 256"%(im_0,im_1))
            'end if'
            for i in range(0,subx):
                for j in range(0,suby):
                    imsub = im[int((im_0/subx)*i):int((im_0/subx)*(i+1)),int((im_1/suby)*j):int((im_1/suby)*(j+1))]
                    imsub = (imsub*255).astype('uint8')
                    binary_img = Image.fromarray(imsub)
#                        binary_img = binary_img.convert('')
                    binary_img.save(path.join(savedir,"%d.png"%count))
                    count+=1                      
                'end for'
            'end for'
'end def'

def return_results_CNN(threshold,
                       imagepath=r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\unet-master\unet-master\data\membrane\saved_predict",
                       rootdir=r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\Detect_A_Cell",
                       datfile = r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\Detect_A_Cell\saved_data",
                       savepath = r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\unet-master\unet-master\data\membrane\saved_binaries"):
    """takes reconstitued images from save_image_predict() and converts images over to binaries that can
    be compared to algorithmic calculations"""
#    with open(path.join(rootdir,datfile),'rb') as f:
#        objs = pickle.load(f)
#    'end with'    
    objs = Graphing._import_data(datfile)
    objsnew = [internal_calc() for i in range(len(objs))]
    for root, dirs, files in walk(imagepath):
        for f in files:
            index = int(f.split("_")[0])
            objsnew[index] = objs[index]
            im = plt.imread(path.join(root,f))
            if len(im.shape) == 3:
                im = im[:,:,1]
            'end if'
            im_bool = (im >= threshold)
            objsnew[index].overlap = internal_calc.error_btwn(objs[index].cell_bool,im_bool,False)
            objsnew[index].cell_bool = im_bool
            objsnew[index]._count_cells()
            objsnew[index]._number_jacobs()
            print(index)
        'end for'
    'end for'
    internal_calc.save_multiple_f(savepath,objsnew)
#    hand = path.abspath(path.join(rootdir,"imagedata_CNN.dat"))
#    with open(hand,'wb') as f:
#        pickle.dump(objs,f)
#    'end with'
'end def'

def graph_thresh_and_wait(im,filename):

    fig1 = plt.figure('raw image %s'%(filename))
    plt.imshow(im)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (10,10)
    plt.show()
    
    fig2 = plt.figure('press to continue')
    plt.show()
    plt.waitforbuttonpress()
    
    
    cell_thr = input("threshold cell: ")
    back_thr = input("threshold back: ")
    
    plt.close(fig1)
    plt.close(fig2)
    
    cell_bool = im >= float(cell_thr)
    cell_bool = remove_small_objects(cell_bool,min_size=25)
    back_bool = im >= float(back_thr)
    
    fig3 = plt.figure('cell_bool')
    plt.imshow(cell_bool)
    
    fig4 = plt.figure('back_bool')
    plt.imshow(back_bool)
    plt.waitforbuttonpress()

    plt.close(fig3)
    plt.close(fig4)
    
    return (cell_bool,back_bool)
'end def'

def thresh_and_check(imagedir,savedir,channel):
    """
    shows image for thresholding, determine appropriate threshold to be used
    create training data, check bitmap, and recalculate as needed using input
    """
    count = 0
    filestart = str(input("specific file to start at? [image name]/[n]: "))
    for root, dirs, files in walk(imagedir):
        index = [i for i,x in enumerate(files) if x==filestart][0]
        count = index
        for f in sorted(files[index:]):
            
            im = plt.imread(path.join(root,f))
            if len(im.shape) == 3:            
                im = im[:,:,channel]
            'end if'
            im = img_as_float32(im)
#            im_filt = internal_calc._FourierFilt(im,(500,1000),0,0)
            im_filt = convolve(im,internal_calc.d3gaussian(10))
            im = internal_calc._diffmat(im_filt,np.arange(0,np.pi*2,(np.pi*2)/12),(4,4))
#            im = np.multiply(im,im_diff)
            ans = True
            (cell_bool,back_bool) = graph_thresh_and_wait(im,f)
            ans = input("recalculate dat bitch doh? [y/n]: ")
            while ans:
                if ans == "y":
                   (cell_bool,back_bool) = graph_thresh_and_wait(im,f)
                   ans = input("recalculate dat bitch doh? [y/n]: ")
                else:
                    break
            'end while'
            cell_bool = (cell_bool*255).astype('uint8')
            binary_img = Image.fromarray(cell_bool)
            binary_img.save(path.join(savedir,"%d.png"%count))
            count += 1
        'end for'
    'end for'
'end def'
            
if __name__ == "__main__":
    get_ipython().run_line_magic('matplotlib','qt5')
#    imagedir = r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\MergedandStiched"
#    savedir = r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\unet-master\unet-master\data\membrane\train\label_862019_2"
#    thresh_and_check(imagedir,savedir,1)
#    savedir = r"C:\Users\jsalm\Docum1ents\Python Scripts\Automated_Histo\unet-master\unet-master\data\membrane\test"
#    imagedir = r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\MergedandStiched"
#    convert_image(imagedir,savedir)
#    save_image_predict() #used 3 as slicvar
    objs = return_results_CNN(0.0392)
    

                
                    
                
    
    