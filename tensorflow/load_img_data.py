import os
from PIL import Image
from array import *
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#dir = 'C:/Users/Tony/Desktop/images'
#dir = 'C:/Users/Tony/Downloads/python/py_study/images/'

def load_img(dir):
    data_image_total = []
    data_label = array('B')
    FileList = []
    for subdir in os.listdir(dir): 
        print('dirname', subdir)
        path = os.path.join(dir, subdir)
        print(path)
        for filename in os.listdir(path):
            if filename.endswith(".bmp"):
                FileList.append(os.path.join(dir, subdir, filename)) 
    img_num = 0           
    for filename in FileList:
        label = 2 
        Im = Image.open(filename)
        pixel = Im.load()
        width, height = Im.size
        print('width and height:', width, height)
        if width != 28 or  height != 28:
            continue    
        
        img_num += 1
        data_image = []
        for x in range(0,width):
            for y in range(0,height):
                data_image.append(pixel[y,x])

        data_image_total.append(data_image)        

    print(np.array(data_image_total).shape)
    '''
    for data in np.array(data_image_total):
        plt.imshow(data.reshape((28, 28)))
        plt.show()
    ''' 
    return np.array(data_image_total)

#load_img('C:/Users/Tony/Downloads/python/py_study/images/')
