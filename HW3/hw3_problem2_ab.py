import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import copy
import random





def normalize(img):
    img = (img - img.min()) / (img.max() - img.min())
    img = img * 255
    
    return img.astype(np.uint8)

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

def energy(img, filter_size=15):
    img_padded = np.pad(img,  [(int((filter_size-1)/2),int((filter_size-1)/2)),(int((filter_size-1)/2),int((filter_size-1)/2))], 'symmetric')
    result = np.zeros(img.shape)
    y_start, y_end = int((filter_size-1)/2), int(img.shape[0]+(filter_size-1)/2)
    x_start, x_end = int((filter_size-1)/2), int(img.shape[1]+(filter_size-1)/2)
    for i in range(y_start, y_end):
        for j in range(x_start, x_end):
            result[i-y_start, j-x_start] = np.sum(np.multiply(img_padded[i-y_start:i+y_start+1, j-y_start:j+y_start+1],img_padded[i-y_start:i+y_start+1, j-y_start:j+y_start+1]))

    return result

def kmeans(tensor, iteration):
    np.random.seed(21)
    label = np.zeros((tensor.shape[0],tensor.shape[1]))
    init_idx = np.random.randint(256, size=6)
    c0, c1, c2 = tensor[init_idx[0],init_idx[1],:], tensor[init_idx[2],init_idx[3],:], tensor[init_idx[4],init_idx[5],:]
    for _ in range(iteration):
        d0 = ((tensor - c0)*(tensor - c0)).sum(axis=2)
        d1 = ((tensor - c1)*(tensor - c1)).sum(axis=2)
        d2 = ((tensor - c2)*(tensor - c2)).sum(axis=2)
        label[(d2<=d1)&(d2<=d0)] = 2
        label[(d1<=d0)&(d1<=d2)] = 1
        label[(d0<=d2)&(d0<=d1)] = 0
        tmp = []
        for i,j in zip(np.where(label==0)[0],np.where(label==0)[1]):
            tmp.append(tensor[i,j,:])
        if len(tmp) !=0:     
            c0 = sum(tmp)/len(tmp)
        tmp = []
        for i,j in zip(np.where(label==1)[0],np.where(label==1)[1]):
            tmp.append(tensor[i,j,:])
        if len(tmp) !=0:     
            c1 = sum(tmp)/len(tmp)
        tmp = []
        for i,j in zip(np.where(label==2)[0],np.where(label==2)[1]):
            tmp.append(tensor[i,j,:])
        if len(tmp) !=0:     
            c2 = sum(tmp)/len(tmp)
    
    return label

    



def main(opt):
    file_path = os.path.dirname(os.path.abspath(__file__))
    img_sample2 = cv2.imread(os.path.join(file_path, 'hw3_sample_images', opt.input), cv2.IMREAD_GRAYSCALE)

    L = []
    L.append(np.array([[1,2,1],[2,4,2],[1,2,1]])/36)
    L.append(np.array([[1,0,-1],[2,0,-2],[1,0,-1]])/12)
    L.append(np.array([[-1,2,-1],[-2,4,-2],[-1,2,-1]])/12)
    L.append(np.array([[-1,-2,-1],[0,0,0],[1,2,1]])/12)
    L.append(np.array([[1,0,-1],[0,0,0],[-1,0,1]])/4)
    L.append(np.array([[-1,2,-1],[0,0,0],[1,-2,1]])/4)
    L.append(np.array([[-1,-2,-1],[2,4,2],[-1,-2,-1]])/12)
    L.append(np.array([[-1,0,1],[2,0,-2],[-1,0,1]])/4)
    L.append(np.array([[1,-2,1],[-2,4,-2],[1,-2,1]])/4)
    print('1 stage...')
    M = []
    for i in range(len(L)):
        M.append(convolution(img_sample2,L[i]))
    print('2 stage...')
    T = []
    for i in range(len(L)):
        tmp = energy(M[i],filter_size = 15)
        T.append(tmp)
        # cv2.imwrite(f'test{i}.png',normalize(tmp))
    
    tensor = np.stack(T,axis=2)
    print('kmeans...')
    label = kmeans(tensor, 15)
    cv2.imwrite(os.path.join(file_path, opt.output1), np.invert((label*255/2).astype(np.uint8)))
    

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='sample2.png', help='input image')
    parser.add_argument('--output1', default='result6.png', help='output image')
    # parser.add_argument('--output2', default='result2.png', help='output image')
    opt = parser.parse_args()
    
    main(opt)