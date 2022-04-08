import os
import argparse
import numpy as np
import cv2

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
    img_sample1 = Img(img_sample1)
    
    

    H = np.array([[1,1,1],[1,1,1],[1,1,1]])
    tmp = np.zeros(img_sample1.img.shape)
    component_list = []
    for i,j in zip(np.where(img_sample1.img==255)[0],np.where(img_sample1.img==255)[1]):
        flag = False
        if len(component_list) > 0:
            for k in range(len(component_list)):
                if component_list[k][i,j]==255:
                    flag = True
        if flag:
            continue
        tmp = np.zeros(img_sample1.img.shape)
        tmp[i,j] = 255
        result5 = Img(tmp)
        for dil in range(100000):
            tmp_img = result5.img
            result5.img = result5.dilation(H) & org.img
            if np.array_equal(tmp_img,result5.img):
                break
        component_list.append(result5.img)

    result5 = np.zeros((img_sample1.img.shape[0],img_sample1.img.shape[1],3))
    for component in component_list:
        color = np.random.randint(256, size=3)
        for i,j in zip(np.where(component==255)[0],np.where(component==255)[1]):
            result5[i,j,:] = color

    cv2.imwrite(os.path.join(file_path, opt.output1),result5)
        
    
    


    
 
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='sample1.png', help='input image')
    parser.add_argument('--output1', default='result5.png', help='output image')
    opt = parser.parse_args()
    
    main(opt)
