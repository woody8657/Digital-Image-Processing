import os
import argparse
from cv2 import _OutputArray_DEPTH_MASK_ALL_BUT_8S, norm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy

def rotate(img, theta):
    output = np.zeros(img.shape)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
                output[i,j] = 1
def scaling(img, s_x, s_y):
    output = np.zeros(img.shape)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            try:
                p, q = (i-img.shape[0]-0.5)/s_y + img.shape[0] + 0.5, (j-0.5)/s_x + 0.5
                if p>=0 and q>=0:
                
                    a, b = p - np.floor(p), q - np.floor(q)
                    output[i, j] = (1-a)*(1-b)*img[int(np.floor(p)),int(np.floor(q))] + (1-a)*(b)*img[int(np.floor(p)),int(np.floor(q))+1] \
                        +  (a)*(1-b)*img[int(np.floor(p))+1,int(np.floor(q))] + (a)*(b)*img[int(np.floor(p))+1,int(np.floor(q))+1]
            except:
                pass
    return output



def translate(img, t_x, t_y):
    output = np.zeros(img.shape)
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            try:
                p, q = i+t_y, j-t_x
                a, b = p - np.floor(p), q - np.floor(q)
                output[i, j] = (1-a)*(1-b)*img[int(np.floor(p)),int(np.floor(q))] + (1-a)*(b)*img[int(np.floor(p)),int(np.floor(q))+1] \
                    +  (a)*(1-b)*img[int(np.floor(p))+1,int(np.floor(q))] + (a)*(b)*img[int(np.floor(p))+1,int(np.floor(q))+1]
            except:
                pass
    return output

        
def HE(img):
    hist, _ = np.histogram(img[img!=0].ravel(), bins = np.arange(257), density=True)
    cdf = np.cumsum(hist)
    f = lambda x : cdf[x]
    output = f(img) * 255
    return output


def main(opt):
    file_path = os.path.dirname(os.path.abspath(__file__))
    img_sample3 = cv2.imread(os.path.join(file_path, 'imgs_2022', opt.input), cv2.IMREAD_GRAYSCALE)
    org_img = copy.deepcopy(img_sample3)

    print('delete dege...')
    for i in range(img_sample3.shape[0]):
        count = 0
        for j in range(img_sample3.shape[1]):
            if img_sample3[i,j] != 0 :
                img_sample3[i,j] = 0
                count = count + 1
            if count == 3 :
                break
    for j in range(img_sample3.shape[1]):
        count = 0
        for i in range(img_sample3.shape[0]):
            if img_sample3[i,j] != 0 :
                count = count + 1
                img_sample3[i,j] = 0
            if count == 3 :
                break

    for i in range(img_sample3.shape[0]):
        count = 0
        for j in range(img_sample3.shape[1]):
            if img_sample3[i,img_sample3.shape[1]-j-1] != 0 :
                count = count + 1
                img_sample3[i,img_sample3.shape[1]-j-1] = 0
            if count == 3 :
                break

    for j in range(img_sample3.shape[1]):
        count = 0
        for i in range(img_sample3.shape[0]):
            if img_sample3[img_sample3.shape[0]-1-i,j] != 0 :
                count = count + 1
                img_sample3[img_sample3.shape[0]-1-i,j] = 0
            if count == 3 :
                break
    # tmp = HE(img_sample3)
    # tmp = img_sample3
    # cv2.imwrite(os.path.join(file_path, '2_a.png'), tmp)
    img_sample3 = scaling(img_sample3, 0.5, 0.5)
    img_sample3 = translate(img_sample3, 150, 260)
    result7 = np.transpose(img_sample3) + img_sample3
    result7 =  translate(result7, -10, 0)
    result7 = result7 + translate(result7[::-1, ::-1],-10,-10)
    cv2.imwrite(os.path.join(file_path, opt.output1), result7)

         




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='sample3.png', help='input image')
    parser.add_argument('--output1', default='result7.png', help='output image')
    opt = parser.parse_args()
    
    main(opt)