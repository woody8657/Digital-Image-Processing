from curses import window
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

def PSNR(img1, img2):
        MSE = np.sum(np.multiply(img1-img2,img1-img2))/(img1.shape[0]*img1.shape[1])
        PSNR = 10 * np.log10(255*255/MSE)
        return PSNR

def main(opt):
    file_path = os.path.dirname(os.path.abspath(__file__))
    img_sample3 = cv2.imread(os.path.join(file_path, 'hw1_sample_images', opt.input1), cv2.IMREAD_GRAYSCALE)
    img_sample3 = img_sample3.astype('float32')
    # (a)
    # uniform noise
    print('producing images of problem 2.a...')
    img_sample4 = cv2.imread(os.path.join(file_path, 'hw1_sample_images', opt.input2), cv2.IMREAD_GRAYSCALE)
    img_sample4 = img_sample4.astype('float32')
    # H = np.ones((3,3)) / (3**2)
    H = np.array([[1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]])/273
    filter_size = H.shape[0]
    img_padded = np.pad(img_sample4,  [(int((filter_size-1)/2),int((filter_size-1)/2)),(int((filter_size-1)/2),int((filter_size-1)/2))], 'symmetric')
    result10 = np.zeros(img_sample3.shape)
    y_start, y_end = int((filter_size-1)/2), int(img_sample3.shape[0]+(filter_size-1)/2)
    x_start, x_end = int((filter_size-1)/2), int(img_sample3.shape[1]+(filter_size-1)/2)
    for i in range(y_start, y_end):
        for j in range(x_start, x_end):
            result10[i-y_start, j-x_start] = np.sum(np.multiply(img_padded[i-y_start:i+y_start+1, j-y_start:j+y_start+1],H))
    cv2.imwrite(os.path.join(file_path, opt.output1), result10)
    # impulse noise
    img_sample5 = cv2.imread(os.path.join(file_path, 'hw1_sample_images', opt.input3), cv2.IMREAD_GRAYSCALE)
    img_sample5 = img_sample5.astype('float32')
    filter_size = 3
    img_padded = np.pad(img_sample4,  [(int((filter_size-1)/2),int((filter_size-1)/2)),(int((filter_size-1)/2),int((filter_size-1)/2))], 'symmetric')
    result11 = np.zeros(img_sample3.shape)
    y_start, y_end = int((filter_size-1)/2), int(img_sample3.shape[0]+(filter_size-1)/2)
    x_start, x_end = int((filter_size-1)/2), int(img_sample3.shape[1]+(filter_size-1)/2)
    for i in range(y_start, y_end):
        for j in range(x_start, x_end):
            result11[i-y_start, j-x_start] = np.median(img_padded[i-y_start:i+y_start+1, j-y_start:j+y_start+1])
    cv2.imwrite(os.path.join(file_path, opt.output2), result11)

    #(b)
    print(f"PSNR between sample3.png and sample4.png : {PSNR(img_sample3,img_sample4)}")
    print(f"PSNR after denoising : {PSNR(img_sample3,result10)}")
    print(f"PSNR between sample3.png and sample5.png : {PSNR(img_sample3,img_sample5)}")
    print(f"PSNR after denoising : {PSNR(img_sample3,result11)}")
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input1', default='sample3.png', help='input image')
    parser.add_argument('--input2', default='sample4.png', help='input image')
    parser.add_argument('--input3', default='sample5.png', help='input image')
    parser.add_argument('--output1', default='result10.png', help='input image')
    parser.add_argument('--output2', default='result11.png', help='input image')
    opt = parser.parse_args()

    main(opt)