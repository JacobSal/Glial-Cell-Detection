# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 20:26:53 2019

@author: Das_Gooche
"""
import pickle
from Internal_Calc import internal_calc
import settings
import matplotlib.pyplot as plt
import __main__
import Graphing
import operator
import xlwings as xw
import os
import numpy as np
from IPython import get_ipython
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.graphics.factorplots import interaction_plot
import matplotlib.pyplot as plt
from scipy import stats
#from pylab import *
#from matplotlib.cbook import get_sample_data
#from matplotlib._png import read_png



global slidesnum
slidesnum = []

    
def _import_data(rootdir):
    "Imports data from a .dat file stored from previous calculations"
    objs = []
    if os.path.getsize(rootdir) > 0:
        for root, dirs, files in os.walk(rootdir):
            for f in files:
                with open(os.path.join(root,f),'rb') as f:
                    objs.extend(pickle.load(f))
        return objs
    'end if'
'end def'

def _organize_data(dat,overlap = False,sort="automatic",seq = 3):
    """organizes data based on group number, animal number ("manual" mode), or the ("automatic") labeling given
    from filenames
    
    dat = Graphing._import_data(r"filename.dat")
    sort = string: automatic/manual
    """
    
    slidesnum = []
    groupsnum = []
    rowsnum = []
    columsnum = []
    groupsnum = []
    animalsnum = []
    
    for i in range(0,len(dat)):
        slidesnum.append(int(dat[i].slide_num))
        groupsnum.append(int(dat[i].group_num))
        rowsnum.append(int(dat[i].row_num))
        groupsnum.append(int(dat[i].group_num))
        animalsnum.append(int(dat[i].animal_num))
        try: int(dat[i].colum_num)
        except ValueError:
            continue
        else:
            columsnum.append(int(dat[i].colum_num))
    'end for'

    slidesnum = np.unique(slidesnum)
    groupsnum = np.unique(groupsnum)
    rowsnum = np.unique(rowsnum)
    columsnum = np.unique(columsnum)
    animalsnum = np.unique(animalsnum)
    groupsnum = np.unique(groupsnum)
    if overlap == True:
        rowsnum = rowsnum[1:]
        columsnum = columsnum[1:]
        slidesnum = slidesnum[1:]
        
    'end if'
    print(len(rowsnum))
    print(slidesnum)
    print(columsnum)
           
    slic = 0
    rowc = 0
    colc = 0
    
    counts = np.zeros((len(rowsnum),len(slidesnum),len(columsnum)))
    sequenced = np.zeros((len(rowsnum),(len(slidesnum)+1)*len(columsnum)))
    overlapval = np.zeros((len(rowsnum),len(slidesnum),len(columsnum)))
    sequencedov = np.zeros((len(rowsnum),(len(slidesnum)+1)*len(columsnum)))
    for slide in slidesnum:
        currentslide = [x for x in dat if int(x.slide_num)== slide]
        for row in rowsnum:
            currentrow = [x for x in currentslide if int(x.row_num) == row]
            for col in columsnum:
                for x in currentrow:
                    try: int(x.colum_num)
                    except ValueError:
                        continue
                    else:
                        if int(x.colum_num) == col:
                            counts[rowc,slic,colc] = x.jacobsnum
                            if overlap == True:
                                overlapval[rowc,slic,colc] = x.overlap
                            'end if'
                        'end if'
                colc += 1
            'end for'
            colc = 0
            rowc += 1
        'end for'
        rowc = 0
        slic += 1
    'end for'
#    print(overlapval.shape[0])
#    print(counts.shape[0])
    """sequencing the data to adapt to cutting schedule"""
    new_hash = [0]*seq
    for j in range(len(new_hash)):
        new_hash[j] = np.arange(j+1,(seq*counts.shape[2]+1)-(seq-(j+1)),3)
    'end for'
    slide_hash = new_hash        
    remd = int((counts.shape[1] - counts.shape[1]%seq)/seq)
    re_count = seq*counts.shape[2]  
    for i in range(remd):
        slide_hash = np.concatenate((slide_hash,new_hash))
        slide_hash[-3:,:] = slide_hash[-3:,:]+re_count
        re_count += seq*counts.shape[2]
    'end for'
    slide_hash = slide_hash[:counts.shape[1],:]
    for i in range(0,counts.shape[0]):
        for j in range(0,counts.shape[1]):
            sequenced[i,slide_hash[j,:]] = counts[i,j,:]
        'end for'
    'end for'
    
    if overlap == True:
        new_hashov = [0]*seq
        for j in range(len(new_hashov)):
            new_hashov[j] = np.arange(j+1,(seq*overlapval.shape[2]+1)-(seq-(j+1)),3) 
        'end for'
        slide_hashov = new_hashov
        remdov = int((overlapval.shape[1] - overlapval.shape[1]%seq)/seq)
        re_countov = seq*overlapval.shape[2]
        for i in range(remdov):
            slide_hashov = np.concatenate((slide_hashov,new_hashov))
            slide_hashov[-3:,:] = slide_hashov[-3:,:]+re_countov
            re_countov += seq*overlapval.shape[2]
        'end for'
        slide_hashov = slide_hashov[:overlapval.shape[1],:]
        for i in range(0,overlapval.shape[0]):
            for j in range(0,overlapval.shape[1]):
                sequencedov[i,slide_hashov[j,:]] = overlapval[i,j,:]
            'end for'
        'end for'
    'end if'
    
    Graphing.slidesnum = slidesnum
    return (sequenced,sequencedov)
        
def _average_zaxis(sorted_data):
    sorted_data_new = np.zeros((sorted_data.shape[0],sorted_data.shape[1]))
    sorted_data_std = np.zeros((sorted_data.shape[0],sorted_data.shape[1]))
    for i in range(0,int(sorted_data.shape[0])):
        for j in range(0,int(sorted_data.shape[1])):
            sorted_data[sorted_data == 0] = 'nan'
            sorted_data_std[i,j] = np.nanstd(sorted_data[i,j,:])
            sorted_data_new[i,j] = np.nanmean(sorted_data[i,j,:])            
        'end for'
    'end for'
    return (sorted_data_new,sorted_data_std)

def remove_anm(shifted_data,anm):
    if type(anm) == int:
        print('working...')
        shifted_data_reduced = np.concatenate((shifted_data[:(anm),:],shifted_data[(anm+1):,:]),axis=0)
    'end if'
    return shifted_data_reduced

def _average_group(shifted_data,grouping):
    averageg = np.zeros((len(grouping),shifted_data.shape[1]))
    stddevg = np.zeros((len(grouping),shifted_data.shape[1]))
    shifted_data[shifted_data==0] = 'nan'
    grouping.insert(0,0)
    for j in range(0,len(grouping)-1):
        for k in range(0,shifted_data.shape[1]):
            averageg[j,k] = np.nanmean(shifted_data[grouping[j]:int(grouping[j]+grouping[j+1]),k])
            stddevg[j,k] = np.nanstd(shifted_data[grouping[j]:int(grouping[j]+grouping[j+1]),k])
        'end for'
    'end for'
    averageg[averageg == 0] = 'nan'
    stddevg[stddevg == 0] = 'nan'
    return (averageg,stddevg)

def _normalize_data(sorted_data):
    sorted_data[sorted_data==0] = 'nan'
    normalized_data = np.zeros((sorted_data.shape[0],sorted_data.shape[1]))
    for i in range(0,sorted_data.shape[0]):
        avgg = np.nanmean(sorted_data[i,:])
        normalized_data[i,:] = sorted_data[i,:]/avgg
    'end for'
    return normalized_data

def center_data_on_max(sorted_data,overlapdata,interval):
    """finds max values in each row (axis = 0) and centers the data in a new array along the max"""
    
    sorted_data[sorted_data == 0] = 'nan'
    overlapdata[overlapdata == 0] = 'nan'
    iterates = int((sorted_data.shape[1]-(sorted_data.shape[1]%interval))/interval)
    new_sorted = np.zeros((sorted_data.shape[0],iterates+1))
    new_sortedov = np.zeros((overlapdata.shape[0],iterates+1))
    for i in range(0,int(sorted_data.shape[0])):
        count = 0
        while count <= iterates:
            if int(count*interval)+int(interval) > sorted_data.shape[1]:
                remain = int((sorted_data.shape[1]-count*interval))
                print("interval did not divide evenly into data. appending average of %d vals"%(remain))
                new_sorted[i,count] = np.nanmean(sorted_data[i,int(count*interval):int(count*interval)+remain])
                new_sortedov[i,count] = np.nanmean(overlapdata[i,int(count*interval):int(count*interval)+remain])
                break
            'end if'
            new_sorted[i,count] = np.nanmean(sorted_data[i,int(count*interval):int(count*interval)+int(interval)])
            new_sortedov[i,count] = np.nanmean(overlapdata[i,int(count*interval):int(count*interval)+int(interval)])
            count += 1
        'end while'
    'end for'
    
    new_sorted[np.isnan(new_sorted)] = 0
    indexg = np.zeros((int(new_sorted.shape[0]),1))
    valueg = np.zeros((int(new_sorted.shape[0]),1))
    "find max in each row"
    for i in range(0,int(sorted_data.shape[0])):
        indexg[i,0],valueg[i,0] = max(enumerate(new_sorted[i,:]),key=operator.itemgetter(1))
    'end for'
    
    "find max in each subset"
    maxindg = int(np.max(indexg))
    
    "shift data"
    shifted_data = np.zeros((int(new_sorted.shape[0]),int(new_sorted.shape[1]+maxindg)))
    shifted_dataov = np.zeros((int(new_sortedov.shape[0]),int(new_sortedov.shape[1]+maxindg)))
    for i in range(0,sorted_data.shape[0]):
        zerq = int(maxindg - indexg[i])
        shifted_data[None,i,0:len(new_sorted[i,:])+zerq] = np.insert(new_sorted[None,i,:],[0]*zerq,0,axis=1)
        shifted_dataov[None,i,0:len(new_sortedov[i,:])+zerq] = np.insert(new_sortedov[None,i,:],[0]*zerq,0,axis=1)
    'end for'
    
    return (shifted_data,shifted_dataov)

def get_axis(sorted_data,interval):
    sorted_data[sorted_data==0] = 'nan'
    yaxis = np.zeros((sorted_data.shape[1]+1))
    tics = list(range(1,(sorted_data.shape[1]+1)))
    yaxis[1::] = np.multiply(tics,20*interval)
    yaxis = np.concatenate((yaxis[::-1]*-1,yaxis[1::]),axis=0)
    midl = [i for i, e in enumerate(yaxis[:]) if e==0]
    boolarray = np.nanmax(sorted_data,axis=1)
    imidlg = [i for i,x in enumerate(sorted_data[3] == boolarray[3]) if x]
    iendg = len(sorted_data[0,:])
    yaxisbox = yaxis[int(midl[0])-1-int(imidlg[0]):int(midl[0])+int(iendg)-int(imidlg[0])]
    yaxiserror = yaxis[int(midl[0])-int(imidlg[0]):int(midl[0])+int(iendg)-int(imidlg[0])]
    yaxisbox = [int(i) for i in yaxisbox]
    yaxiserror = [int(i) for i in yaxiserror]
    for i,x in enumerate(yaxisbox):
        if x%1000 != 0:
            yaxisbox[i] = None
        'end if'
    'end for'
    
    return (yaxiserror,yaxisbox)

def box_plot(shifted_data,grouping,interval):
    """takes a 2D array of data and creates box plots using only the non-zero values
    
    sorted_data = np.array([])
    grouping = list, values must represent the number of objects in each group
    ex./ grouping = [2,3], therefore, there are 2 objects in first group and 3 objects in second group"""
    
    grouping.insert(0,0)
    shifted_data[shifted_data==0] = 'nan'
    (yaxis,yaxis2) = get_axis(shifted_data,interval)
    for i in range(0,len(grouping)-1):
        plt.figure('Microglia green dpi_boxplot group %d'%(i))
        group_boxval = shifted_data[grouping[i]:grouping[i]+grouping[i+1],:]
        mask = ~np.isnan(group_boxval)
        group_boxval = [d[m] for d,m in zip(group_boxval.T,mask.T)]
#        group_max = []
#        for i in range(0,len(grouping)):
#            group_max.append(np.nanmax(np.concatenate(group_boxval)))
#        'end for'
        x = np.transpose(yaxis2)
        xi = [i for i in range(0,len(x))]
        plt.boxplot(group_boxval)
        plt.xticks(xi,x)
        plt.xlabel('distance along spinal cord (um)')
        plt.ylabel('fraction of gliosis present in the spinal cord section')
        plt.show()
    'end for'
    
def error_plot(average_data,std_data,shifted_data,overlapdata,interval,overlap=False):
    (yaxis,yaxis2) = get_axis(shifted_data,interval)
    shifted_data[shifted_data == 0] = 'nan'
    
    
    plt.figure('Microglia green dpi-average/std')           
    plt.xlabel('distance along spinal cord (um)')
    plt.ylabel('fraction of gliosis present in the spinal cord section')
    labels = ['HBO','NonHBO']
    for i in range(average_data.shape[0]):
        plt.errorbar(yaxis,average_data[i,:],std_data[i,:],label='group %s'%(labels[i]),linestyle='-',marker='o')
    'end for'
    plt.legend(loc='upper left')
    plt.show()

    plt.figure('Microglia counts green dpi-scattehttps://i.stack.imgur.com/cuhnS.pngr')
    plt.xlabel('distance along spinal cord (um)')
    plt.ylabel('fraction of gliosis present in the spinal cord section')
    for i in range(shifted_data.shape[0]):
        plt.scatter(yaxis,shifted_data[i,:],label='animal %d'%(i))
    'end for'
    plt.legend(loc='upper left')
    plt.show()
    if overlap == True:
        plt.figure('Overlap between Unet and Algorithmic detection of Gliosis')
        plt.xlabel('distance along spinal cord (um)')
        plt.ylabel('percent amount of overlap')
        for i in range(overlapdata.shape[0]):
            plt.scatter(yaxis,overlapdata[i,:],label='animal %d'%(i))
        'end for'
        plt.legend(loc='upper left')
        plt.show()
'end def'

def _3dgraph(objs,animal):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    count=1
    plt.draw()
    for obj in objs:
        if int(obj.animal_num) == int(animal):
            img = obj.cell_bool
            x, y = ogrid[0:img.shape[0], 0:img.shape[1]]
            ax.plot_surface(x, y, img*count)
            count += 1
        else:
            continue
    'end for'       
    return (fig,ax)
'end def'
def remove_nan_groups(shifted_data,average_dat,col):
    (axis,notwanted) = get_axis(shifted_data,4)
    while True in np.isnan(shifted_data):
        if True in np.isnan(shifted_data[:,col]):
            average_dat  = np.concatenate((average_dat[:,:col],average_dat[:,int(col+1):]),axis=1)
            shifted_data  = np.concatenate((shifted_data[:,:col],shifted_data[:,int(col+1):]),axis=1)
            axis = np.concatenate((axis[:col],axis[int(col+1):]))
        'end if'
        col += 1        
    return (axis,average_dat)

def _2way_anova(shifted_data,grouping):
    data_set = {}
    groupHBO = {}
    groupNonHBO = {}
    distval = {}
    count = 0    
    grouping.insert(0,0)
    group = [None]*(len(grouping)-1)
    (axis,_) = get_axis(shifted_data)
#    for i in range(0,len(grouping)-1):
#        group[i] = shifted_data[grouping[i]:grouping[i]+grouping[i+1],:]
#    'end'\
    shifted_data_new = []
    axis_new = []
    i = 0
    for i in range(shifted_data.shape[1]):
        if not(np.isnan(shifted_data[0:2,i])) and not(np.isnan(shifted_data[2:4,i])):
#            average_dat_new = np.concatenate((average_dat[:,:i],average_dat[:,int(i+1):]),axis=1)
            shifted_data_new.append(shifted_data[:,i])
            axis_new.append(axis[i])
        'end if'
        i += 1
    'end while'
    shifted_data = np.transpose(np.array(shifted_data_new))
    axis = np.transpose(np.array(axis_new))
    
#    remove_nan_groups(shifted_data,average_dat,0)
            
    groupHBO["HBO"] = shifted_data[0:2]
    groupNonHBO["NonHBO"] = shifted_data[2:4]
    
    for dist in axis:
        group1["%d"%(count)] = group[0][count]
        group2["%d"%(count)] = group[1][count]
        distval["%d"%(count)] = dist
        count += 1
    'end for'
    
    data_set["HBO"] = group1
    data_set["NonHBO"] = group2
    data_set["Distance"] = distval
    
    d = pd.DataFrame(data_set)
    d = d.where((pd.notnull(d)),'drop')
    d_melt = pd.melt(d,id_vars=['Distance'],value_vars=['HBO','NonHBO'])
    d_melt.columns = ['Distance','Group','value']
    model = ols('value ~ C(Distance) + C(Group) + C(Distance,Group)', data = d_melt, missing = 'drop').fit()
    anova_table = anova_lm(model,typ=2)
    return anova_table

def export2excel(shifted_data,sheet,
                 wbdir=r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\Exported_Data_overlap.xlsx"):
    (dist,_) = Graphing.get_axis(shifted_data,4)
    wb = xw.Book(wbdir)
    sht = wb.sheets[sheet]
    for j in range(len(dist)):
        sht.range((j+3,2)).value = dist[j]
    for i in range(0,shifted_data.shape[0]):
        for j in range(0,shifted_data.shape[1]):
            if np.isnan(shifted_data[i,j]):
                shifted_data[i,j] = 0
            'end if'
            sht.range((j+3,i+3)).value = shifted_data[i,j]
        'end j'
    'end i'
'end def'
#
#def importfromexcel(sheet,
#                    wbdir=r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\Exported_Data_overlap.xlsx")
#    (dist,_) = Graphing.get_axis(shifted_data,4)
#    wb = xw.Book(wbdir)
#    sht = wb.sheets['Sheet1']
#    for j in range(len(dist)):
#        sht.range((j+3,2)).value = dist[j]
#    for i in range(0,shifted_data.shape[0]):
#        for j in range(0,shifted_data.shape[1]):
#            if np.isnan(shifted_data[i,j]):
#                shifted_data[i,j] = 0
#            'end if'
#            sht.range((j+3,i+3)).value = shifted_data[i,j]
#        'end j'
#    'end i'
#'end def'
    
def see_overlays(dat):
    dec = input('would you like to see overlays? [y/n]')
    if dec == 'y':
        ans = True
        j = 0
        plt.ion()
        dat[j].show_combinefig(j,1)
        while ans:
            print("""
                  "N": Next
                  "B": Back
                  "Recalc": Recalculate image
                  "specific obj#": ##
                  "E": Exit
                  """)
            ans = input("Choose an Option: ") 
            plt.close()
            if ans == "N":
                dat[j].show_combinefig(j,1)
                plt.close()
                j += 1
            elif ans == "B":
                j -= 1
                dat[j].show_combinefig(j,1)
                plt.close("all")
            elif ans == "Recalc":
                filtersize = input("filter size type(tuple): ")
                settings.images[j]
                dat[j]._change_channel(1)
                dat[j].cell_process(filtersize,0,False)
                dat[j].back_process()
                print(dat[j]._number_jacobs())
                print(dat[j]._count_cells())
            elif type(ans) == int:
                j = ans
                objs[j].show_combinefig(j,1)
            elif ans == "E":
                break
            "end if"
        "end while"
    else: pass 
     
def save_combfigs(dat):
    for i in range(0,len(dat)):
        fig = dat[i].show_combinefig(i,1)
        fig.savefig(r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\Detect_A_Cell\Saved_Figs\Comb_figs\%s_%s_%s.tif"%(dat[i].slide_num,dat[i].row_num,dat[i].colum_num),dpi=500, pad_inches=0.05,format='tif')
    'end for'


if __name__ == '__main__':
    plt.rcParams['figure.dpi'] = 400
    plt.rcParams['figure.figsize'] = (10,10)
    get_ipython().run_line_magic('matplotlib','qt5')
    dirpath = __main__.__file__

#    hand = os.path.abspath(os.path.join(dirpath,"..","saved_data"))
#    hand = r"E:\1LabWork\Automated Histology\Detect_A_Cell\saved_data"
    hand = r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\unet-master\unet-master\data\membrane\saved_data"
    objs = _import_data(hand)
    (array,overlapval) = _organize_data(objs,True)
    array_rem = remove_anm(array,0)
    overlapval_rem = remove_anm(overlapval,0)
    (shifted,shiftedov) = center_data_on_max(array_rem,overlapval_rem,4)
#    normalized = _normalize_data(shifted)
    (average_g,std_g) = _average_group(shifted,[2,2])
    box_plot(shifted,[2,2],4)
    error_plot(average_g,std_g,shifted,shiftedov,4,True)
#    export2excel(shifted,'Unet')
#    export2excel(shiftedov,'Overlap')
    
    
    
#    normalizedov = _normalize_data(shiftedov)
#    (average_gov,std_gov) = _average_group(shiftedov,[2,2])
#    box_plot(shiftedov,[2,2],4)
#    error_plot(average_gov,std_gov,shiftedov,4)
    
#    anovatable = _2way_anova(shifted,average_g,[2,2])
#    wbdir = r"C:\Users\jsalm\Documents\Python Scripts\Automated_Histo\unet-master\unet-master\data\membrane\Exported_DataCNN.xlsx"
##    export2excel(shifted,wbdir)
#    settings.init()
#    imagedir = os.path.abspath(os.path.join(dirpath,"..","..","MergedandStiched"))
#    internal_calc.load_file(imagedir)
#    save_combfigs(objs)
#    see_overlays(objs)
    '''
    array = _organize_data(objs)
    array = remove_anm(array,0)
    shifted = center_data_on_max(array)
    (average_g,std_g) = _average_group(shifted,[3,2])
    box_plot(shifted,[3,2])
    error_plot(average_g,std_g,shifted)
    '''