# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 13:12:05 2018

@author: jsalm
"""


import __main__
import settings
import numpy as np
from os import path
import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(True)
from matplotlib.colors import ListedColormap
from IPython import get_ipython
from Internal_Calc import internal_calc
import Graphing

plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.figsize'] = (10,10)

plt.close('all')
get_ipython().run_line_magic('matplotlib','qt5')


def imshow_overlay(im, mask, alpha=0.2, color='red', **kwargs):
    """Show semi-transparent red mask over an image"""
    mask = mask > 0
    mask = np.ma.masked_where(~mask, mask)
    fig = plt.figure('overlayed')        
    plt.imshow(im, **kwargs)
    plt.imshow(mask, alpha=alpha, cmap=ListedColormap([color]))
    return fig
'end def'
'''
if __name__ == '__main__':
    dirpath = __main__.__file__
    internal_calc.save_multiple_f(dirpath,objs)
'end for'
'''
if __name__ == '__main__':
    dirpath = __main__.__file__
    imagedir = path.abspath(path.join(dirpath,"..","..","MergedandStiched"))
#    imagedir = r"D:\LabWork\Automated Histology\Gabby Keyence imaged\MergedandStiched"
    settings.init()
    internal_calc.load_file(imagedir)
    objs = [internal_calc() for i in range(len(settings.images))]
        
    #start loop through Images    
    for j in range(0,len(settings.images)):
        plt.close('all')
        objs[j].slide_num = settings.filename[j][6:8]
        objs[j].row_num = settings.filename[j][13:14]
        objs[j].animal_num = settings.filename[j][13:14]
        objs[j].colum_num = settings.filename[j][22:23]
        objs[j]._change_channel(1)
        objs[j].cell_process((400,1000),0,(4,4),False)
        objs[j].back_process()
        print(objs[j]._count_cells())
        objs[j]._number_jacobs()
        
        print(settings.filename[j])
        print(objs[j].jacobsnum)
        
        i = 1
        
        'end if'   
    'end for'
    dirpath = r"E:\LabWork\Automated Histology\Detect_A_Cell\saved_data"
    internal_calc.save_multiple_f(dirpath,objs)
 
'end if'

for j in objs:
    j.cell_count = j._count_cells()
'end for'














