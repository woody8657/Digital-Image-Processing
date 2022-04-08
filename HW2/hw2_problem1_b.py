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

    # 1.Noise reduction
    print('Denoising...')
    F = np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]]) / 159
    output_denoise = convolution(img_sample1, F)
    
    # 2.Gradient map and orientation 

    K = 2 # Sobel mask
    # Row gradient
    print('Sobel...')
    k_r = np.array([[-1, 0, 1], [-K, 0, K], [-1, 0, 1]])/(K+2)
    row_gdmap = convolution(output_denoise, k_r)
    k_c = np.array([[1, K, 1], [0, 0, 0], [-1, -K, -1]])/(K+2)
    col_gdmap = convolution(output_denoise, k_c)
    output_edge = np.sqrt(np.square(row_gdmap)+np.square(col_gdmap))


    # 3.NMS
    print('NMS...')
    output_NMS = copy.deepcopy(output_edge)
    theta = np.arctan(col_gdmap/row_gdmap)
    for i in range(1,output_edge.shape[0]-1):
        for j in range(1,output_edge.shape[1]-1):
            if theta[i,j] >= 0:
                if (output_edge[i,j]<=output_edge[i-1,j+1]) or (output_edge[i,j]<=output_edge[i+1,j-1]):
                    output_NMS[i,j] = 0
            else:
                if (output_edge[i,j]<=output_edge[i+1,j+1]) or (output_edge[i,j]<=output_edge[i-1,j-1]):
                    output_NMS[i,j] = 0
    # distribution of output_NMS
    # plt.hist(output_NMS.ravel(), alpha=0.5)
    # plt.savefig(os.path.join(file_path, '1_b_hist.png'))
    # plt.clf()

    
    # 4.Thresholding
    print('thresholding...')
    output_thresholding = double_thresholding(output_NMS, np.percentile(output_NMS.ravel(), 96), np.percentile(output_NMS.ravel(), 90))
   

    # Connected
    print('labeling...')
    output_connected = connected(output_thresholding)
    # cv2.imwrite(os.path.join(file_path, opt.output1), np.concatenate((img_sample1,normalize(output_denoise),normalize(output_edge),normalize(output_NMS),normalize(output_thresholding),normalize(output_connected)),axis=1))
    cv2.imwrite(os.path.join(file_path, opt.output1), output_connected)
    








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='sample1.png', help='input image')
    parser.add_argument('--output1', default='result3.png', help='output image')
    opt = parser.parse_args()
    
    main(opt)