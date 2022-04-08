import os
import argparse
from cv2 import _OutputArray_DEPTH_MASK_ALL_BUT_8S, norm, resize
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

        


def main(opt):
    file_path = os.path.dirname(os.path.abspath(__file__))
    img_sample3 = cv2.imread(os.path.join(file_path, 'imgs_2022', opt.input), cv2.IMREAD_GRAYSCALE)
    result8 = np.zeros(img_sample3.shape)
    for i in range(result8.shape[0]):
        for j in range(result8.shape[1]):
            try:

                result8[i,j] = img_sample3[i-int(30*np.sin((j+35)*2*np.pi/150)),j+int(30*np.sin((i+30)*2*np.pi/150))]
            except:
                pass
    # result8 = copy.deepcopy(img_sample3)
    # for j in range(result8.shape[1]):
    #     tmp = np.zeros((result8.shape[0],1))
    #     t = int(40*np.sin(j*np.pi/80))
    #     if t > 0:
    #         tmp[0:result8.shape[1]-t,0] = np.squeeze(result8[t::,j])
    #         result8[:,j] = np.squeeze(tmp)
    #     elif t < 0:
    #         tmp[-t::,0] = np.squeeze(result8[:result8.shape[1]+t,j])
    #         result8[:,j] = np.squeeze(tmp)
    #     else:
    #         pass

    # for i in range(result8.shape[0]):
    #     tmp = np.zeros((1,result8.shape[1]))
    #     t = int(20*np.sin(i*np.pi/80))
    #     if t > 0:
    #         tmp[0,0:result8.shape[0]-t] = np.squeeze(result8[i,t::])
    #         result8[i,:] = np.squeeze(tmp)
    #         pass
    #     elif t < 0:
    #         tmp[0,-t::] = np.squeeze(result8[i,:result8.shape[0]+t])
    #         result8[i,:] = np.squeeze(tmp)
    #     else:
    #         pass

    
    # cv2.imwrite(os.path.join(file_path, opt.output1), np.concatenate((img_sample3,result8),axis=1))
    cv2.imwrite(os.path.join(file_path, opt.output1), result8)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='sample5.png', help='input image')
    parser.add_argument('--output1', default='result8.png', help='output image')
    opt = parser.parse_args()
    
    main(opt)