import os
from PIL import Image
from array import *
from random import shuffle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load from and save to

os.chdir('C:/Users/Tony/Downloads/python/py_study/')
Names = [['./training-images','train'], ['./test-images', 'test']]
for name in Names:
    
    data_image = array('B')
    data_label = array('B')

    FileList = []
    for dirname in os.listdir(name[0]): # [1:] Excludes .DS_Store from Mac OS
        print('dirname',dirname)
        path = os.path.join(name[0],dirname)
        print(path)
        for filename in os.listdir(path):
            if filename.endswith(".bmp"):
                FileList.append(os.path.join(name[0],dirname,filename))

    shuffle(FileList)
    for filename in FileList:

        label = 2 #int(filename.split('/')[2])

        Im = Image.open(filename)

        pixel = Im.load()

        width, height = Im.size

        print('pixel:', pixel)
        print('width and height:', width, height)
        for x in range(0,width):
            for y in range(0,height):
                data_image.append(pixel[y,x])

        data_label.append(label) # labels start (one unsigned byte each)

    hexval = "{0:#0{1}x}".format(len(FileList),6) # number of files in HEX

    # header for label array

    header = array('B')
    header.extend([0,0,8,1,0,0])
    header.append(int('0x'+hexval[2:][:2],16))
    header.append(int('0x'+hexval[2:][2:],16))
    
    data_label = header + data_label

    # additional header for images array
    if max([width,height]) <= 256:
        header.extend([0,0,0,width,0,0,0,height])
    else:
        raise ValueError('Image exceeds maximum size: 256x256 pixels');
    
    header[3] = 3 # Changing MSB for image data (0x00000803)
    
    model = tf.keras.models.load_model('C:\\Users\\Tony\\Downloads\\python\\py_study\\tensorflow\\my_model_test.h5')
    print('data_image', np.array(data_image).reshape(1, 28, 28))

    result = model.predict(np.array(data_image).reshape(1, 28, 28) / 255, batch_size=1)

    print('result', result)
    predict = np.argmax(result,axis=1)  #axis = 1是取行的最大值的索引，0是列的最大值的索引
    plt.title(predict[0])
    plt.imshow(np.array(data_image).reshape(28, 28))
    plt.show()

    data_image = header + data_image

    output_file = open(name[1]+'-images-idx3-ubyte', 'wb')
    data_image.tofile(output_file)
    output_file.close()

    output_file = open(name[1]+'-labels-idx1-ubyte', 'wb')
    data_label.tofile(output_file)
    output_file.close()
