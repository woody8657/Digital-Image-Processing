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
        return np.ones(self.img.shape).astype(np.uint8)*255 - self.img


def main(opt):
    file_path = os.path.dirname(os.path.abspath(__file__))
    img_sample1 = cv2.imread(os.path.join(file_path, 'hw3_sample_images', opt.input), cv2.IMREAD_GRAYSCALE)
    org = Img(img_sample1)
    tmp = np.zeros(img_sample1.shape)
    tmp[242,103] = tmp[234,132] = tmp[269,149] = \
    tmp[475,300] = tmp[462,295] = tmp[487,305] = \
    tmp[400,290] = tmp[425,205] = tmp[500,235] = 255
    result2 = Img(tmp)
    
    
    
    H = np.array([[0,1,0],[1,1,1],[0,1,0]])
    for _ in range(100):
        result2.img = result2.dilation(H) & org.complement()
    result2.img = result2.img | org.img
    cv2.imwrite(os.path.join(file_path, opt.output1),result2.img)
    


    # Searching G_0
    # i,j = 475,300
    # w = 25
    # cv2.imwrite(os.path.join(file_path, 'test.png'),org.img[i-w:i+w+1,j-w:j+w+1])
    # print(org.img[475,300])
    # print(org.img[462,295])
    # print(org.img[487,305])
    # print(org.img[400,290])
    # print(org.img[425,205])
    # print(org.img[500,235])
    # print(org.img[269,149])
    
 
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='sample1.png', help='input image')
    parser.add_argument('--output1', default='result2.png', help='output image')
    opt = parser.parse_args()
    
    main(opt)
