# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 19:13:37 2019

@author: Das_Gooche
"""

from Internal_Calc import internal_calc as ic
import Tk
from matplotlib.colors import ListedColormap
import settings

class gui_process(ic):
    
    def __init__(self,master,animal_num,slide_num,row_num,colum_num,cut_width,channel,group_num):
        self.master = master
        master.title = ""
        
        self.animal_num = animal_num
        self.slide_num = slide_num
        self.row_num = row_num
        self.colum_num = colum_num
        self.cut_width = cut_width
        self.channel = channel
        self.group_num = group_num

        master.title("Automatic Lesion Detection")
        
    def _load_file():
        "loads file into objects that can be passed to internal_calc"
    def display_img():
        "dispalys image to GUI contained within internal calc storage includes(orignal,cells,backim,filtered)"
    'end def'
    def dipslay_stats():
        "takes data from class Internal_calc and dispalys it ontop GUI"
    'end def'
    def save_img():
        "takes data from class Internal_calc and saves it to external .dat file"
    'end def'
    def next_img():
        "load next img in input file"
    'end def'
    def display_graphs():
        """opens new window which contains graphs for lines graphs and boxplots
        based on group_num, slide_num, cut_width, channel. Takes 
        internal_storage.calc"""
    'end def'
    def load_program():
        """loads start up window of GUI"""
        top = Tkinter.Tk()
        top.mainloop()
       
    def imshow_overlay(im, mask, alpha=0.35, color='red', **kwargs):
        """Show semi-transparent red mask over an image"""
        try: mask = mask > 0            
        except:
            return np.array([[1,1,1],[1,1,1]])
        mask = np.ma.masked_where(~mask, mask)      
        plt.imshow(im, **kwargs)
        plt.imshow(mask, alpha=alpha, cmap=ListedColormap([color]))
    'end def'
    
if __name__ == "__main__":
    root = Tk()
    my_gui = gui_process(root)
    root.mainloop()
'end if'