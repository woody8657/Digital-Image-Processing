import os
import argparse
import numpy as np
import cv2
import math

def get_threshold_matrix(N=2):
    I = np.array([[1, 2],[3, 0]])
    for _ in range(int(math.log(N,2)-1)):
        I = np.block([[4*I+1, 4*I+2],[4*I+3, 4*I+0]])
    T = ((I+0.5)/(N**2)) * 255
    return T


def main(opt):
    file_path = os.path.dirname(os.path.abspath(__file__))
    sample1 = cv2.imread(os.path.join(file_path, 'hw4_sample_images', opt.input), cv2.IMREAD_GRAYSCALE)
    
    # 1(a) 
    result1 = np.zeros(sample1.shape)
    T = get_threshold_matrix(N=2)
    for i in range(0,sample1.shape[0],2):
        for j in range(0,sample1.shape[1],2):
            result1[i:i+2,j:j+2] = (sample1[i:i+2,j:j+2]>T)*255
    cv2.imwrite(os.path.join(file_path, opt.output1), result1)
    
    # 1(b) 
    result2 = np.zeros(sample1.shape)
    T = get_threshold_matrix(N=256)
    result2 = (sample1>T)*255
    cv2.imwrite(os.path.join(file_path, opt.output2), result2)

    # 1(c)
    # Floyd Steinberg
    E = np.zeros(sample1.shape)
    G = np.zeros(sample1.shape)
    for i in range(0,sample1.shape[0]-1):
        for j in range(1,sample1.shape[1]-1):
            G[i,j] = ((sample1[i,j]+E[i,j]) >= 255/2) * 255
            tmp = (sample1[i,j]+E[i,j]) - G[i,j]
            E[i:i+2,j-1:j+2] = E[i:i+2,j-1:j+2] + np.array([[0,0,7],[3,5,1]])/16*tmp
    result3 = G
    cv2.imwrite(os.path.join(file_path, opt.output3), result3)
    # Jarvis’
    E = np.zeros((256,256))
    G = np.zeros((256,256))
    for i in range(0,sample1.shape[0]-2):
        for j in range(2,sample1.shape[0]-2):
            G[i,j] = ((sample1[i,j]+E[i,j]) >= 255/2) * 255
            tmp = (sample1[i,j]+E[i,j]) - G[i,j]
            E[i:i+3,j-2:j+3] = E[i:i+3,j-2:j+3] + np.array([[0,0,0,7,5],[3,5,7,5,3],[1,3,5,3,1]])/48*tmp
    result4 = G
    cv2.imwrite(os.path.join(file_path, opt.output4), result4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='sample1.png', help='input image')
    parser.add_argument('--output1', default='result1.png', help='output image')
    parser.add_argument('--output2', default='result2.png', help='output image')
    parser.add_argument('--output3', default='result3.png', help='output image')
    parser.add_argument('--output4', default='result4.png', help='output image')
    opt = parser.parse_args()

    main(opt)