import os
import argparse
from cv2 import _OutputArray_DEPTH_MASK_ALL_BUT_8S, norm
import numpy as np
import cv2
import matplotlib.pyplot as plt
import copy



def normalize(img):
    img = (img - img.min()) / (img.max() - img.min())
    img = img * 255
    img.astype(np.uint8)
    
    return img

def convolution(img, kernel):
    filter_size = kernel.shape[0]
    img_padded = np.pad(img,  [(int((filter_size-1)/2),int((filter_size-1)/2)),(int((filter_size-1)/2),int((filter_size-1)/2))], 'symmetric')
    result = np.zeros(img.shape)
    y_start, y_end = int((filter_size-1)/2), int(img.shape[0]+(filter_size-1)/2)
    x_start, x_end = int((filter_size-1)/2), int(img.shape[1]+(filter_size-1)/2)
    for i in range(y_start, y_end):
        for j in range(x_start, x_end):
            result[i-y_start, j-x_start] = np.sum(np.multiply(img_padded[i-y_start:i+y_start+1, j-y_start:j+y_start+1],kernel))

    return result

def thresholding(img, T=125):
    binary_map = copy.deepcopy(img)
    binary_map[img>=T] = 255
    binary_map[img<T] = 0
    
    return binary_map 

def double_thresholding(img, T_H, T_L):
    binary_map = copy.deepcopy(img)
    binary_map[img>T_H] = 255
    binary_map[(img<=T_H) & (img>=T_L)] = 128
    binary_map[img<T_L] = 0
    
    return binary_map 

# def connected(img_mask):
#     img = copy.deepcopy(img_mask)
#     flag = False
#     idx_edge = np.where(img==255)
#     tmp = 0
#     try:
#         for i in range(len(idx_edge[0])):
#             idx_candidate = np.where(img[idx_edge[0][i]-1:idx_edge[0][i]+2,idx_edge[1][i]-1:idx_edge[1][i]+2]==128)
#             if len(idx_candidate[0]) != 0:
#                 tmp = tmp + len(idx_candidate[0])
#                 flag = True
#                 for j in range(len(idx_candidate[0])):
#                     img[idx_edge[0][i]+idx_candidate[0][j],idx_edge[1][i]+idx_candidate[1][j]] = 255
#     except:
#         pass
#     print(f'{tmp} candidate are founded!!')
#     if not flag:
#         img[img==128]=0
#         return img
#     else:
#         return connected(img)

def connected(img_mask):
    img = copy.deepcopy(img_mask)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[0]-1):
            if (img[i,j+1] == 128) and (img[i,j] == 255):
                img[i,j+1]=255
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[0]-1):
            if (img[img.shape[0]-1-i,img.shape[0]-2-j] == 128) and (img[img.shape[0]-1-i,img.shape[0]-1-j] == 255):
                img[img.shape[0]-1-i,img.shape[0]-2-j] = 255
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[0]-1):
            if (img[i+1,j] == 128) and (img[i,j] == 255):
                img[i+1,j]=255
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[0]-1):
            if (img[img.shape[0]-2-i,img.shape[0]-1-j] == 128) and (img[img.shape[0]-1-i,img.shape[0]-1-j] == 255):
                img[img.shape[0]-2-i,img.shape[0]-1-j] = 255
    img[img==128] = 0
    return img


        



def main(opt):
    file_path = os.path.dirname(os.path.abspath(__file__))
    img_sample1 = cv2.imread(os.path.join(file_path, 'imgs_2022', opt.input), cv2.IMREAD_GRAYSCALE)

    # 1.Gaussian low pass filter
    print('Denoising...\(>u<)/')
    G = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/273
    LP = convolution(img_sample1, G)
    # 2. Unsharp masking
    print('Unsharp masking...\(>u<)/')
    c = 3/5
    result5 = (c/(2*c-1))*img_sample1 - ((1-c)/(2*c-1))*normalize(LP)
    cv2.imwrite(os.path.join(file_path, opt.output1), np.concatenate((normalize(img_sample1),result5.astype(np.uint8)),axis=1))  
         




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='x-ray.png', help='input image')
    parser.add_argument('--output1', default='result5.png', help='output image')
    opt = parser.parse_args()
    
    main(opt)