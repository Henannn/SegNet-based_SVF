import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import numpy.matlib
import glob as gb
import PIL.Image as Image
import os
img_path = gb.glob("pred\\*.png") 
counter = 0
for path in img_path:
    counter += 1
    #file = str(path)
    img = io.imread(path)#skimage中的图像展示
    row, col, channel = img.shape
    img = img * 1.0
    center_x = (col-1)/2.0
    center_y = (row-1)/2.0
    xx = np.arange (col) #array([0,1,2.....col-1])col列
    yy = np.arange (row)#array([0,1,2.....row-1])row行
    e = col*1.0/row
    x_mask = numpy.matlib.repmat (xx, row, 1)#由传入的矩阵复制产生新的矩阵，行为xx的row倍,列为xx数组的1倍
    y_mask = numpy.matlib.repmat (yy, col, 1)
    y_mask = np.transpose(y_mask)#将y_mask矩阵x,y轴互换
    xx_dif = x_mask - center_x
    yy_dif = center_y - y_mask
    theta = np.arctan2(yy_dif*e, xx_dif+0.0001) #fisheye中的某一点的角度
    mask_1 = yy_dif < 0
    theta = theta * (1 - mask_1) + (theta + 3.1415926 * 2) * mask_1
    r1 = yy_dif / np.sin(theta)
    y1_mask = r1 * 1.0
    x1_mask = theta * col / (2 * 3.1415926)
    mask = x1_mask < 0  
    x1_mask = x1_mask * (1 - mask) 
    mask = x1_mask > (col - 1) 
    x1_mask = x1_mask * (1 - mask) + (x1_mask * 0 + col -2) * mask
    mask = y1_mask < 0
    y1_mask = y1_mask * (1 - mask)
    mask = y1_mask > (row -1)
    y1_mask = y1_mask * (1 - mask) + (y1_mask * 0 + row -2) * mask
    int_x = np.floor (x1_mask)
    int_x = int_x.astype(int)
    int_y = np.floor (y1_mask)
    int_y = int_y.astype(int)
    p_mask = x1_mask - int_x
    q_mask = y1_mask - int_y
    img_out = img * 2.0

    for ii in range(row):
        for jj in range (col):
            new_xx = int_x [ii, jj]
            new_yy = int_y [ii, jj]
            p = p_mask[ii, jj]    
            q = q_mask[ii, jj] 

            img_out[ii, jj, :] = img[new_yy, new_xx, :]

    img_out = img_out / 255.0
    plt.figure(2)
    plt.imshow(img_out)
    plt.axis('off')
    plt.savefig('fit\\' + str(counter) + '.png', dpi=500 , bbox_inches='tight')

foreground = Image.open("utils\\x.png")
img_path = gb.glob("fit\\*.png") 
counter = 0
for path in img_path:
    #file = str(path)
    filepath, tempfilename = os.path.split(path)
    filename, extension = os.path.splitext(tempfilename)
    #print('filename:',filename)
    #print('extension:',extension)
    counter += 1
    background = Image.open(path)
    background = background.resize((480, 480))
    background.paste(foreground, (0, 0), foreground)
    background.save('result\\' + str(counter) + '.png',quality=95,subsampling=0)

img_path = gb.glob("result\\*.png")
counter_128 = 0
counter_black = 0
counter_all = 0
for path in img_path:
    #file = str(path)
    im = Image.open(path)
    rgb_im = im.convert('RGB')
    width = im.size[0]
    height = im.size[1]
    #print('width',width)
    #print('height',height)
    for i in range(0,width):
        for j in range(0,height):
            r, g, b = rgb_im.getpixel((i, j))
            counter_all += 1 
            if(r == 128 and g == 128 and b == 128):
                counter_128 += 1
            if(r == 0 and g == 0 and b == 0):
                counter_black += 1
    svf = counter_128/(counter_all - counter_black)
    print(svf)
