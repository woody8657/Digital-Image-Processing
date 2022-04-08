import os
import argparse
import numpy as np
import cv2

def main(opt):
    file_path = os.path.dirname(os.path.abspath(__file__))
    img_sample1 = cv2.imread(os.path.join(file_path, 'hw1_sample_images', opt.input))
    # (a)
    print('producing images of problem 0.a...')
    result1 = img_sample1[:,::-1,:]
    cv2.imwrite(os.path.join(file_path, opt.output1), result1)
    # (b)
    # Gray = R*0.299 + G*0.587 + B*0.114
    print('producing images of problem 0.b...')
    result2 = img_sample1[:,:,0] * 0.114 + img_sample1[:,:,1] * 0.587 + img_sample1[:,:,0] * 0.299
    cv2.imwrite(os.path.join(file_path, opt.output2), result2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='sample1.png', help='input image')
    parser.add_argument('--output1', default='result1.png', help='output image')
    parser.add_argument('--output2', default='result2.png', help='output image')
    opt = parser.parse_args()
    main(opt)