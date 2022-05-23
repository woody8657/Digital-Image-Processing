import os
import argparse
import numpy as np
import cv2
import math

def get_filter(threshold, size, type='ideal', n=2):
    center_i, center_j = (size[0]-1)/2, (size[1]-1)/2
    filter = np.zeros(size)
    if type == 'ideal':
        for i in range(filter.shape[0]):
            for j in range(filter.shape[1]):
                D = ((i-center_i)**2 + (j-center_j)**2)**0.5
                D0 = threshold
                if D > D0:
                    filter[i,j] = 1
    elif type == 'butterworth':
        for i in range(filter.shape[0]):
            for j in range(filter.shape[1]):
                D = ((i-center_i)**2 + (j-center_j)**2)**0.5
                D0 = threshold
                filter[i,j] = 1 / (1 + (D0/D)**(2*n))
    elif type == 'gaussian':
        for i in range(filter.shape[0]):
            for j in range(filter.shape[1]):
                D = ((i-center_i)**2 + (j-center_j)**2)**0.5
                D0 = threshold
                filter[i,j] = 1 - np.exp(-(D**2)/(2*(D0**2)))
    else:
        pass

    return filter

def main(opt):
    file_path = os.path.dirname(os.path.abspath(__file__))
    sample2 = cv2.imread(os.path.join(file_path, 'hw4_sample_images', opt.input1), cv2.IMREAD_GRAYSCALE)
    # sample3 = cv2.imread(os.path.join(file_path, 'hw4_sample_images', opt.input2), cv2.IMREAD_GRAYSCALE)
    sample3 = cv2.imread('/home/u/woody8657/tmp/OXR_utils/tmp1.png' , cv2.IMREAD_GRAYSCALE)
   
    
    # 2(a)
    # f_image = np.fft.fftshift(np.fft.fft2(sample2))
    # result5 = np.log(np.abs(f_image)+1)
    # result5 = (result5 - result5.min()) / (result5.max() - result5.min())*255
    # result5 = cv2.resize(sample2, (300, 300), interpolation=cv2.INTER_NEAREST)
    # result5 = cv2.resize(result5, (600, 600), interpolation=cv2.INTER_NEAREST)
    # cv2.imwrite(os.path.join(file_path, opt.output1), result5)

    # 2(b)
    f_image = np.fft.fftshift(np.fft.fft2(sample3))
    # result6 = np.abs(f_image)
    # result6 = np.log(np.abs(f_image)+1)
    # result6 = (result6 - result6.min()) / (result6.max() - result6.min())*255
    # cv2.imwrite('fft.png', result6)
    algo = 'butterworth' # 'ideal', 'butterworth' or 'gaussian'
    D0 = 1
    filter = get_filter(D0, f_image.shape, type=algo, n=2)
    f_image = f_image * filter
    print(sample3.shape,f_image.shape)
    result6 = abs(np.fft.ifft2(f_image))
    print(result6.max(), result6.min())
    cv2.imwrite(os.path.join(file_path, opt.output2), result6)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input1', default='sample2.png', help='input image')
    parser.add_argument('--input2', default='sample3.png', help='input image')
    parser.add_argument('--output1', default='result5.png', help='output image')
    parser.add_argument('--output2', default='result6.png', help='output image')
    opt = parser.parse_args()

    main(opt)