
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 15:42:48 2018

@author: jsalm
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pickle
import os
from Internal_Calc import internal_calc
from IPython import get_ipython
from matplotlib.colors import ListedColormap
import settings
from os import path
import operator
import __main__

print(__main__.__file__)

plt.close('all')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (10,10)
get_ipython().run_line_magic('matplotlib','qt5')


def imshow_overlay(im, mask, alpha=0.5, color='red', **kwargs):
    """Show semi-transparent red mask over an image"""
    mask = mask > 0
    mask = np.ma.masked_where(~mask, mask)        
    plt.imshow(im, **kwargs)
    plt.imshow(mask, alpha=alpha, cmap=ListedColormap([color]))
'end def'


dirpath = __main__.__file__
hand = path.abspath(path.join(dirpath,"..","imagedata.dat"))
savepath = path.abspath(path.join(dirpath,"..","Saved_Figs"))
imagedir = path.abspath(path.join(dirpath,"..","..","MergedAndStiched"))
settings.init()
internal_calc.load_file(imagedir)

#try objs:
#with open(hand,'rb') as f:
#    objs = pickle.load(f)

#try: objs
#except NameError: 
    
numobj = len(objs)


slidesnum = []
for i in range(0,len(objs)):
    slidesnum.append(int(objs[i].slide_num))
'end for'
slidesnum = np.unique(slidesnum)
numsli = len(slidesnum)

cv.destroyAllWindows()
green_counts = np.zeros((5,numsli))

count1 = 0
count2 = 0
count3 = 0
count4 = 0
count5 = 0

#for j in range(0,len(objs)):  #slides: 1,4*,7,10,13,16,19*,22,24,25,26,28,31,34,37,39,40,43*
#    for k in slidesnum:
#        if int(objs[j].slide_num) == k:
#            if float(objs[j].row_num[1]) == 1:
#                green_counts[0,count1] = float(objs[j].cellcount)
#                count1 += 1
#            elif float(objs[j].row_num[1]) == 2:
#                green_counts[1,count2] = float(objs[j].cellcount)
#                count2 += 1
#            elif float(objs[j].row_num[1]) == 3:
#                green_counts[2,count3] = float(objs[j].cellcount)
#                count3 += 1
#            elif float(objs[j].row_num[1]) == 4:
#                green_counts[3,count4] = float(objs[j].cellcount)
#                count4 += 1
#            elif float(objs[j].row_num[1]) == 5:
#                green_counts[4,count5] = float(objs[j].cellcount)                
#                count5 += 1
#            else:
#                pass
#            'end if'
#        'end if'
#    'end for k'
#'end for j'

for j in range(0,len(objs)):  #slides: 1,4*,7,10,13,16,19*,22,24,25,26,28,31,34,37,39,40,43*
    for k in slidesnum:
        if int(objs[j].slide_num) == k:
            if float(objs[j].row_num[1]) == 1:
                green_counts[0,count1] = float(objs[j].jacobsnum)
                count1 += 1
            elif float(objs[j].row_num[1]) == 2:
                green_counts[1,count2] = float(objs[j].jacobsnum)
                count2 += 1
            elif float(objs[j].row_num[1]) == 3:
                green_counts[2,count3] = float(objs[j].jacobsnum)
                count3 += 1
            elif float(objs[j].row_num[1]) == 4:
                green_counts[3,count4] = float(objs[j].jacobsnum)
                count4 += 1
            elif float(objs[j].row_num[1]) == 5:
                green_counts[4,count5] = float(objs[j].jacobsnum)                
                count5 += 1
            else:
                pass
            'end if'
        'end if'
    'end for k'
'end for j'

num = slidesnum[slidesnum != 0]
slides = np.zeros((1,numsli))
slides[0,0:numsli+1] = num

green_counts[green_counts == 0] = np.nan

#normalize green_counts using lowest fig
#for i in range(0,green_counts.shape[0]):
#    ming = np.nanmin(green_counts[i,:])
#    green_counts[i,:] = green_counts[i,:]/ming
#'end for'



indexg = np.zeros((5,1))
valueg = np.zeros((5,1))
for i in range(0,5):
    indexg[i,0],valueg[i,0] = max(enumerate(green_counts[i,:]),key=operator.itemgetter(1))
'end for'
maxindg = int(np.max(indexg))
minindg = int(np.min(indexg))

zcounts = np.zeros((len(indexg),len(green_counts[0,:])+int(maxindg)))
for i in range(0,5):
    zerq = int(maxindg - indexg[i])
    zcounts[None,i,0:len(green_counts[i,:])+zerq] = np.insert(green_counts[None,i,:],[0]*zerq,0,axis=1)
'end for'
green_counts = zcounts
green_counts[green_counts == 0] = np.nan

boxvalg1 = np.zeros((4,15))
boxvalg2 = np.zeros((4,15))

boxvalg1 = green_counts[0:3,:]
boxvalg2 = green_counts[3:5,:]




mask = ~np.isnan(boxvalg1)
boxvalg1 = [d[m] for d, m in zip(boxvalg1.T,mask.T)]

mask = ~np.isnan(boxvalg2)
boxvalg2 = [d[m] for d, m in zip(boxvalg2.T,mask.T)]

#outliersr1 = red_counts1[red_counts1 > 500]


yaxis = np.zeros((len(slidesnum)+1))
yaxis[1::] = slidesnum*20
zaxis = yaxis[::-1]
yaxis = yaxis[1::]
yaxis = np.concatenate((zaxis*-1,yaxis),axis=0)
midl = [i for i, e in enumerate(yaxis[:]) if e==0]
boolarray = np.nanmax(green_counts,axis=1)
imidlg = [i for i,x in enumerate(green_counts[4] == boolarray[4]) if x]
iendg = len(green_counts[0,:])
yaxis2 = yaxis[int(midl[0])-1-int(imidlg[0]):int(midl[0])+int(iendg)-int(imidlg[0])]
yaxis = yaxis[int(midl[0])-int(imidlg[0]):int(midl[0])+int(iendg)-int(imidlg[0])]


averageghbo = np.zeros((1,len(green_counts[0,:])))
stddevghbo = np.zeros((1,len(green_counts[0,:])))

averageg = np.zeros((1,len(green_counts[0,:])))
stddevg = np.zeros((1,len(green_counts[0,:])))


#y = np.transpose(y)
for i in range(0,green_counts.shape[0]):
    for k in range(0,green_counts.shape[1]):
        averageghbo[0,k] = np.nanmean(green_counts[0:3,k])
        stddevghbo[0,k] = np.nanstd(green_counts[0:3,k])
        averageg[0,k] = np.nanmean(green_counts[3:5,k])
        stddevg[0,k] = np.nanstd(green_counts[3:5,k])
    'end for'
'end for'


max1 = np.nanmax(np.concatenate(boxvalg1[:]))
max2 = np.nanmax(np.concatenate(boxvalg2[:]))
maxfin = np.max([max1,max2])

x = np.transpose(yaxis2)
xi = [i for i in range(0,len(x))]
box1 = plt.figure('Microglia green dpi_boxplot group 1')
plt.boxplot(boxvalg1)
plt.yticks(list(np.arange(0,maxfin,maxfin/5)))
plt.xticks(xi,x)
box1.savefig(savepath+"/box1.tif",dpi=200,format='tif')
box2 = plt.figure('Microglia green dpi_boxplot group 2')
plt.boxplot(boxvalg2)
plt.yticks(list(np.arange(0,maxfin,maxfin/5)))
plt.xticks(xi,x)
box2.savefig(savepath+"/box2.tif",dpi=200,format='tif')


y = yaxis
error1 = plt.figure('Microglia green dpi')
plt.errorbar(y,averageghbo[0,:],stddevghbo[0,:], linestyle='--',marker='o')
plt.errorbar(y,averageg[0,:],stddevg[0,:], linestyle='-',marker='^')
plt.xlabel('distance (um)')
plt.ylabel('jacobs number (number of pixels in cells/number of pixels in body)')
error1.savefig(savepath+"/error1.tif",dpi=200,format='tif')
#plt.errorbar(y,averageg,stddevg, linestyle='--',marker='o')
error2 = plt.figure('Microglia counts green dpi')
plt.plot(y,green_counts[0,:],'bs',y,green_counts[1,:],'b^',y,green_counts[2,:],'bo',y,green_counts[3,:],'rs',y,green_counts[4,:],'r^')
plt.xlabel('distance (um)')
plt.ylabel('jacobs number (number of pixels in cells/number of pixels in body)')
error2.savefig(savepath+"/error2.tif",dpi=200,format='tif')
'''
plt.figure('Astrocytes red dpi')
plt.errorbar(y,averager,stddevr, linestyle='--',marker='o')
plt.errorbar(y,averagerhbo,stddevrhbo, linestyle='-',marker='^')
plt.xlabel('distance (um)')
plt.ylabel('jacobs number (number of pixels in cells/number of pixels in body)')
plt.figure('Astrocytes counts red dpi')
plt.plot(y,red_counts[None,0,:],'bs',y,red_counts[None,1,:],'b^',y,red_counts[None,2,:],'bo',y,red_counts[None,3,:],'rs',y,red_counts[None,4,:],'r^')
plt.xlabel('distance (um)')
plt.ylabel('jacobs number (number of pixels in cells/number of pixels in body)')
'''
#plt.figure('rattos cell counts')
#fig, axs = plt.subplots(nrows = 2, ncols=2, sharex=True)
#ax = axs[0,0]
#ax.errorbar()

rootdir = path.abspath(path.join(dirpath,"..","..","MergedAndStitched"))

f = []
for subdirs,dirs,filenames in os.walk(rootdir):
    for k in filenames:
        f.append(k)
    'end for'
'end for'
# f[2] = float('nan')
# f = [x for x in f if str(x) !='nan']
saveall = input('save figures?' )
if saveall == 'y':
    for j in range(0,len(objs)):
        for i in range(0,2):
            slide = objs[j].slide_num
            row = objs[j].row_num
            imi = settings.images[j]
            overlay = objs[j].cell_bool
            numcel = objs[j].jacobsnum
            body = objs[j].body_bool
            fig = plt.figure('slide '+str(slide)+' animal '+str(row)+' number of cells '+str(numcel))
            fig.suptitle('pixel density: ' + str(numcel),fontsize=12)
            plt.subplot(1,3,1)
            plt.title('Orignial', fontsize=10)
            plt.imshow(imi)
            plt.axis('off')
            plt.subplot(1,3,2)
            plt.title('Cells', fontsize=10)
            plt.imshow(overlay)
            plt.axis('off')
            plt.subplot(1,3,3)
            plt.title('body', fontsize=10)
            plt.imshow(body)
            plt.axis('off')
            plt.close('all')
            fig.savefig(r'D:\LabWork\Automated Histology\Detect_A_Cell\Saved_Compare'+'\\'+'slide'+str(slide)+'_animal'+str(row)+'_numcel'+str(numcel)+'.png',dpi=450)
        'end for'
    'end for'
else: pass
 
dec = input('would you like to see overlays? [y/n]')
if dec == 'y':
    ans = True
    j = 0
    plt.ion()
    objs[j].show_combinefig(j,1)
    while ans:
        print("""
              "N": Next
              "B": Back
              "S": Save
              "Recalc": Recalculate image
              "specific obj#": ##
              "E": Exit
              """)
        ans = input("Choose an Option: ") 
        plt.close()
        if ans == "N":
            objs[j].show_combinefig(j,1)
            plt.close()
            j += 1
        elif ans == "B":
            j -= 1
            objs[j].show_combinefig(j,1)
            plt.close("all")
        elif ans == "S":
            fig.savefig(path.abspath(path.join(dirpath,"..","/Saved_Compare/slide_{}_anm_{}_jacobs_{}.png".format(slide,row,numcel))),dpi=450)
        elif ans == "Recalc":
            filtersize = input("filter size type(tuple): ")
            
            settings.images[j]
            objs[j]._change_channel(1)
            objs[j].cell_process(filtersize,0,False)
            objs[j].back_process()
            print(objs[j]._number_jacobs())
            print(objs[j]._count_cells())
        elif type(ans) == int:
            j = ans
            objs[j].show_combinefig(j,1)
        elif ans == "E":
            break
        "end if"
    "end while"
else: pass

