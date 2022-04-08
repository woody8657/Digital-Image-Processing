import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
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
    tmp_img = copy.deepcopy(img)
    tmp_img[img>=T] = 255
    tmp_img[img<T] = 0
    
    return tmp_img



def main(opt):
    file_path = os.path.dirname(os.path.abspath(__file__))
    img_sample1 = cv2.imread(os.path.join(file_path, 'imgs_2022', opt.input), cv2.IMREAD_GRAYSCALE)

    K = 2 # Sobel mask
    # Row gradient
    print('row...')
    k_r = np.array([[-1, 0, 1], [-K, 0, K], [-1, 0, 1]])/(K+2)
    row_gdmap = convolution(img_sample1, k_r)
    print('col...')
    k_c = np.array([[1, K, 1], [0, 0, 0], [-1, -K, -1]])/(K+2)
    col_gdmap = convolution(img_sample1, k_c)
    result1 = np.sqrt(np.square(row_gdmap)+np.square(col_gdmap))


    # normalize to 0~255
    result1 = normalize(result1)
    cv2.imwrite(os.path.join(file_path, opt.output1), result1)


    # distribution of gradient
    # bins = np.linspace(0, 255, 256)
    # plt.hist(result1.ravel(), bins, alpha=0.5)
    # plt.savefig(os.path.join(file_path, '1_a_hist.png'))
    # plt.clf()

    # thresholding to binary map
    # tmp = np.zeros((img_sample1.shape[0]*2,img_sample1.shape[1]*5))
    # for i in range(10):
    #     percentile = (i+1)*10
    #     tmp_img = thresholding(result1, T=np.percentile(result1.ravel(), percentile)).astype(np.uint8)
    #     if i < 5:
    #         tmp[0:img_sample1.shape[0],i*(img_sample1.shape[1]):(i+1)*(img_sample1.shape[1])] = tmp_img
    #     else:
    #         tmp[img_sample1.shape[0]::,(i-5)*(img_sample1.shape[1]):(i+1-5)*(img_sample1.shape[1])] = tmp_img
    #     cv2.imwrite(os.path.join(file_path, 'thresholding.png'),tmp)



    percentile = 80
    result2 = thresholding(result1, T=np.percentile(result1.ravel(), percentile)).astype(np.uint8)
    cv2.imwrite(os.path.join(file_path, opt.output2), result2)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='sample1.png', help='input image')
    parser.add_argument('--output1', default='result1.png', help='output image')
    parser.add_argument('--output2', default='result2.png', help='output image')
    opt = parser.parse_args()
    
    main(opt)