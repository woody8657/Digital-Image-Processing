import os
import argparse
import numpy as np
import cv2
import time
import copy

class Img():
    def __init__(self, img):
        self.img = img
        
    def translation(self, y, x):
        tmp = np.zeros(self.img.shape)
        idx = np.where(self.img==255)
        tmp[(idx[0]+y,idx[1]+x)] = 255
        
        return tmp

    def dilation(self, H, origin = (1,1)):
        output = np.zeros(self.img.shape).astype(np.uint8)
        for i,j in zip(np.where(H==1)[0],np.where(H==1)[1]):
            output = output | self.translation(i-origin[0],j-origin[1]).astype(np.uint8)
        
        return output

    def erosion(self, H, origin = (1,1)):
        output = np.ones(self.img.shape).astype(np.uint8)*255
        for i,j in zip(np.where(H==1)[0],np.where(H==1)[1]):
            output = output & self.translation(i-origin[0],j-origin[1]).astype(np.uint8)
          
        return output

    def complement(self):
        return np.ones(self.img.shape)*255 - self.img


def main(opt):
    file_path = os.path.dirname(os.path.abspath(__file__))
    img_sample1 = cv2.imread(os.path.join(file_path, 'hw3_sample_images', opt.input), cv2.IMREAD_GRAYSCALE)
    img_sample1 = Img(img_sample1)
    
    H = np.ones((3,3))
    result1 = img_sample1.img - img_sample1.erosion(H)
    cv2.imwrite(os.path.join(file_path, opt.output1),result1)

    
 
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='sample1.png', help='input image')
    parser.add_argument('--output1', default='result1.png', help='output image')
    opt = parser.parse_args()
    
    main(opt)