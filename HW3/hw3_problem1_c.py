import os
import argparse
import numpy as np
import cv2
import copy
import time
def rotation(lst, array):
    lst.append(array)
    lst.append(array[::-1,:])
    lst.append(array[:,::-1])
    lst.append(array[::-1,::-1])

    return lst


def sk(img):
    output = copy.deepcopy(img)
    LUT1 = []
    # B = 4
    tmp = np.array([[0,1,0],[0,1,1],[0,0,0]])
    LUT1 = rotation(LUT1,tmp)
    tmp = np.array([[0,0,1],[0,1,1],[0,0,1]])
    LUT1 = rotation(LUT1,tmp)
    # B = 6
    tmp = np.array([[1,1,1],[0,1,1],[0,0,0]])
    LUT1 = rotation(LUT1,tmp)
    tmp = np.array([[0,1,1],[0,1,1],[0,0,1]])
    LUT1 = rotation(LUT1,tmp)
    # B = 7
    tmp = np.array([[1,1,1],[0,1,1],[0,0,1]])
    LUT1 = rotation(LUT1,tmp)
    # B = 8
    tmp = np.array([[0,1,1],[0,1,1],[0,1,1]])
    LUT1 = rotation(LUT1,tmp)
    # B = 9
    tmp = np.array([[1,1,1],[0,1,1],[0,1,1]])
    LUT1 = rotation(LUT1,tmp)
    tmp = np.array([[0,1,1],[0,1,1],[1,1,1]])
    LUT1 = rotation(LUT1,tmp)
    # B = 10
    tmp = np.array([[1,1,1],[0,1,1],[1,1,1]])
    LUT1 = rotation(LUT1,tmp)
    # B = 11
    tmp = np.array([[1,1,1],[1,1,1],[0,1,1]])
    LUT1 = rotation(LUT1,tmp)


    print('first stage...')
    output_stage1 = np.zeros(img.shape)
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            tmp = img[i-1:i+2,j-1:j+2]
            
            if np.array_equal(tmp, np.zeros((3,3))):
                continue
            
            for pattern in LUT1:
                if np.array_equal(tmp, pattern*255):
                    output_stage1[i,j] = 255
                    break
                

    LUT2 = []
    # spur
    tmp = np.array([[0,0,0],[0,1,0],[0,0,1]])
    LUT2 = rotation(LUT2,tmp)
    # single 4-connectioni
    tmp = np.array([[0,0,0],[0,1,0],[0,1,0]])
    LUT2 = rotation(LUT2,tmp)
    # L corner
    tmp = np.array([[0,1,0],[0,1,1],[0,0,0]])
    LUT2 = rotation(LUT2,tmp)
    # corner cluster
    
    
    print('second stage...')
    for i in range(1,output_stage1.shape[0]-1):
        for j in range(1,output_stage1.shape[1]-1):
            tmp = output_stage1[i-1:i+2,j-1:j+2]
            flag = False
            if np.array_equal(tmp, np.zeros((3,3))):
                continue
            for pattern in LUT2:
                if np.array_equal(tmp, pattern*255):
                    flag = True
                    break
            if flag:
                print('hihihihhihihihih')
                continue
            # elif np.array_equal(tmp[:2,:2], np.ones((2,2))*255) or np.array_equal(tmp[1:,:2], np.ones((2,2))*255) or np.array_equal(tmp[:2,1:], np.ones((2,2))*255) or np.array_equal(tmp[1:,1:], np.ones((2,2))*255):
            #     continue
            # elif (tmp[0,1]==tmp[1,0]==tmp[1,1]==tmp[1,2]==255) or (tmp[1,0]==tmp[0,1]==tmp[1,1]==tmp[2,1]==255) or (tmp[1,0]==tmp[1,1]==tmp[1,2]==tmp[2,1]==255) or (tmp[0,1]==tmp[1,1]==tmp[2,1]==tmp[1,2]==255):
            #     continue
            # elif ((tmp[0,0]==tmp[1,1]==tmp[0,2]==255) and (tmp[2,:].sum()>0)) or ((tmp[0,0]==tmp[1,1]==tmp[2,0]==255) and (tmp[:,2].sum()>0)) or ((tmp[2,0]==tmp[1,1]==tmp[2,2]==255) and (tmp[0,:].sum()>0)) or ((tmp[0,2]==tmp[1,1]==tmp[2,2]==255) and (tmp[:,0].sum()>0)): 
            #     continue
            # elif ((tmp[0,1]==tmp[1,1]==tmp[1,2]==tmp[2,0]==255) and (tmp[0,2]==tmp[1,0]==tmp[2,1]==0)) or ((tmp[0,1]==tmp[1,1]==tmp[1,0]==tmp[2,2]==255) and (tmp[0,0]==tmp[1,2]==tmp[2,1]==0)) or ((tmp[0,2]==tmp[1,1]==tmp[1,0]==tmp[2,1]==255) and (tmp[0,1]==tmp[1,2]==tmp[2,0]==0)) or ((tmp[0,0]==tmp[1,1]==tmp[1,2]==tmp[2,1]==255) and (tmp[0,1]==tmp[1,0]==tmp[2,2]==0)):
            #     continue
            # else:
            #     output[i,j] = 0
            elif (np.array_equal(tmp[:2,1:], np.ones((2,2))*255) and (tmp[0,0]==tmp[1,0]==tmp[2,0]==tmp[2,1]==tmp[2,2])) or \
                 (np.array_equal(tmp[1:,:2], np.ones((2,2))*255) and (tmp[0,0]==tmp[0,1]==tmp[0,2]==tmp[1,2]==tmp[2,2])) or \
                 (np.array_equal(tmp[:2,:2], np.ones((2,2))*255) and (tmp[2,0]==tmp[2,1]==tmp[2,2]==tmp[0,2]==tmp[1,2])) or \
                 (np.array_equal(tmp[1:,1:], np.ones((2,2))*255) and (tmp[0,0]==tmp[1,0]==tmp[2,0]==tmp[0,1]==tmp[0,2])):
                print('aaaaaa')
                continue
            elif ((tmp[0,1]==tmp[1,0]==tmp[1,1]==tmp[1,2]==255) and (tmp[0,0]==tmp[0,2]==tmp[2,0]==tmp[2,1]==tmp[2,2]))or \
                 ((tmp[1,0]==tmp[0,1]==tmp[1,1]==tmp[2,1]==255) and (tmp[0,0]==tmp[2,0]==tmp[0,2]==tmp[1,2]==tmp[2,2])) or \
                 ((tmp[1,0]==tmp[1,1]==tmp[1,2]==tmp[2,1]==255) and (tmp[0,0]==tmp[0,1]==tmp[0,2]==tmp[2,0]==tmp[2,2])) or \
                 ((tmp[0,1]==tmp[1,1]==tmp[2,1]==tmp[1,2]==255) and (tmp[0,0]==tmp[1,0]==tmp[2,0]==tmp[0,2]==tmp[2,2])):
                print('bbbbbb')
                continue
            elif (((tmp[0,0]==tmp[1,1]==tmp[0,2]==255) and (tmp[2,:].sum()>0)) and (tmp[1,0]==tmp[0,1]==tmp[1,2])) or \
                 (((tmp[0,0]==tmp[1,1]==tmp[2,0]==255) and (tmp[:,2].sum()>0)) and (tmp[1,0]==tmp[0,1]==tmp[2,1])) or \
                 (((tmp[2,0]==tmp[1,1]==tmp[2,2]==255) and (tmp[0,:].sum()>0)) and (tmp[1,0]==tmp[2,1]==tmp[1,2])) or \
                 (((tmp[0,2]==tmp[1,1]==tmp[2,2]==255) and (tmp[:,0].sum()>0)) and (tmp[0,1]==tmp[1,2]==tmp[2,1])): 
                print('ccccccc')
                continue
            elif (((tmp[0,1]==tmp[1,1]==tmp[1,2]==tmp[2,0]==255) and (tmp[0,2]==tmp[1,0]==tmp[2,1]==0)) and (tmp[0,0]==tmp[2,2])) or \
                 (((tmp[0,1]==tmp[1,1]==tmp[1,0]==tmp[2,2]==255) and (tmp[0,0]==tmp[1,2]==tmp[2,1]==0)) and (tmp[2,0]==tmp[0,2])) or \
                 (((tmp[0,2]==tmp[1,1]==tmp[1,0]==tmp[2,1]==255) and (tmp[0,1]==tmp[1,2]==tmp[2,0]==0)) and (tmp[0,0]==tmp[2,2])) or \
                 (((tmp[0,0]==tmp[1,1]==tmp[1,2]==tmp[2,1]==255) and (tmp[0,1]==tmp[1,0]==tmp[2,2]==0)) and (tmp[2,0]==tmp[0,2])):
                print('ddddddd')
                continue
            else:
                output[i,j] = 0
                
                     
                     
     
    return output

def sk_ZS(img):
    output = img
    idx = []
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
           
            A = 0
            B = 0
            tmp = img[i-1:i+2,j-1:j+2]
            if np.array_equal(tmp, np.zeros((3,3))):
                continue
            p = [tmp[1,1], tmp[0,1], tmp[0,2], tmp[1,2], tmp[2,2], tmp[2,1], tmp[2,0], tmp[1,0], tmp[0,0]] 
            for k in range(1,len(p)):
                B = B + p[k]
                if k < len(p)-1:
                    if (p[k]==0) and (p[k+1]==255):
                        A = A+1
                else:
                    if (p[k]==0) and (p[1]==255):
                        A = A+1
            
            B = B / 255
            if ((B<=6) and (B>=2)) and (A==1) and ((p[1]*p[3]*p[5])==0) and ((p[3]*p[5]*p[7])==0):
                idx.append([i,j])
    for i,j in idx:
        output[i,j]=0

    idx = []
    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):
            A = 0
            B = 0
            tmp = img[i-1:i+2,j-1:j+2]
            if np.array_equal(tmp, np.zeros((3,3))):
                continue
            p = [tmp[1,1], tmp[0,1], tmp[0,2], tmp[1,2], tmp[2,2], tmp[2,1], tmp[2,0], tmp[1,0], tmp[0,0]] 
            for k in range(1,len(p)):
                B = B + p[k]
                if k < len(p)-1:
                    if (p[k]==0) and (p[k+1]==255):
                        A = A+1
                else:
                    if (p[k]==0) and (p[1]==255):
                        A = A+1
            
            B = B / 255
            if ((B<=6) and (B>=2)) and (A==1) and ((p[1]*p[3]*p[7])==0) and ((p[1]*p[5]*p[7])==0):
                idx.append([i,j])
    for i,j in idx:
        output[i,j]=0
    return output
    



def main(opt):
    file_path = os.path.dirname(os.path.abspath(__file__))
    img_sample1 = cv2.imread(os.path.join(file_path, 'hw3_sample_images', opt.input), cv2.IMREAD_GRAYSCALE)
    result3 = img_sample1.astype(np.uint32)
    
    for it in range(1000):
        print(f'{it}th iteration...')
        tmp = copy.deepcopy(result3)
        a = time.time()
        result3 = sk_ZS(result3)
        b = time.time()
        cv2.imwrite(os.path.join(file_path, opt.output1),result3.astype(np.uint8))
        if np.array_equal(tmp,result3):
            break
    # print(it)
    

    result4 = np.invert(img_sample1).astype(np.uint32)
    for it in range(1000):
        print(f'{it}th iteration...')
        tmp = copy.deepcopy(result4)
        result3 = sk_ZS(result4)
        cv2.imwrite(os.path.join(file_path, opt.output2),result4.astype(np.uint8))
        if np.array_equal(tmp,result4):
            break
    # print(it)

    # result3 = img_sample1
    # resut4 = np.invert(img_sample1)
    # for i in range(100):
    #     print(f'{i}th iteration...')
    #     tmp = copy.deepcopy(result3)
    #     result3 = sk(result3)
    #     cv2.imwrite(os.path.join(file_path, opt.output1),result3)
    #     if np.array_equal(tmp,result3):
    #         break
    # for i in range(100):
    #     print(f'{i}th iteration...')
    #     tmp = copy.deepcopy(result4)
    #     result4 = sk(result4)
    #     cv2.imwrite(os.path.join(file_path, opt.output2),result4)
    #     if np.array_equal(tmp,result3):
    #         break
    
    # result3 = cv2.ximgproc.thinning(img_sample1, thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    # result4 = cv2.ximgproc.thinning(np.invert(img_sample1), thinningType=cv2.ximgproc.THINNING_ZHANGSUEN)
    # cv2.imwrite(os.path.join(file_path, 'test.png'),result3.astype(np.uint8))
    
    


    
 
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='sample1.png', help='input image')
    parser.add_argument('--output1', default='result3.png', help='output image')
    parser.add_argument('--output2', default='result4.png', help='output image')
    opt = parser.parse_args()
    
    main(opt)
