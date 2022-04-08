from curses import window
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

def HE(img):
    hist, _ = np.histogram(img.ravel(), bins = np.arange(257), density=True)
    cdf = np.cumsum(hist)
    f = lambda x : cdf[x]
    output = f(img) * 255
    return output

def LHE(img, window_size=5):
    padded_img = np.zeros((img.shape[0]+(window_size-1), img.shape[1]+(window_size-1)))
    output_img = np.zeros(img.shape)
    # index of image in padded image
    y_start, y_end = int((window_size-1)/2), int(img.shape[0]+(window_size-1)/2)
    x_start, x_end = int((window_size-1)/2), int(img.shape[1]+(window_size-1)/2)
    padded_img[y_start:y_end,x_start:x_end] = img
    for i in range(y_start, y_end):
        for j in range(x_start, x_end):
            hist, _ = np.histogram(padded_img[i-y_start:i+y_start+1, j-y_start:j+y_start+1].ravel(), bins = np.arange(257), density=True)
            cdf = np.cumsum(hist)
            output_img[i-y_start, j-x_start] = cdf[int(padded_img[i,j])] * 255 
    return output_img


def main(opt):
    file_path = os.path.dirname(os.path.abspath(__file__))
    img_sample2 = cv2.imread(os.path.join(file_path, 'hw1_sample_images', opt.input), cv2.IMREAD_GRAYSCALE)
    img_sample2 = img_sample2.astype('float32')
    # (a)
    print('producing images of problem 1.a...')
    result3 = img_sample2 / 2
    cv2.imwrite(os.path.join(file_path, opt.output1), result3)
    # (b)
    print('producing images of problem 1.b...')
    result4 = result3 * 3 
    result4[result4>255] = 255
    cv2.imwrite(os.path.join(file_path, opt.output2), result4)
    # (c)
    print('producing images of problem 1.c...')
    bins = np.linspace(0, 255, 256)
    plt.hist(img_sample2.ravel(), bins, alpha=0.5, label='sample2')
    plt.hist(result3.ravel(), bins, alpha=0.5, label='result3')
    plt.hist(result4.ravel(), bins, alpha=0.5, label='result4')
    plt.ylim(0, 25000)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(file_path, 'problem1_c.png'))
    plt.clf()
    # (d)
    print('producing images of problem 1.d...')
    result3 = cv2.imread(os.path.join(file_path, 'result3.png'),cv2.IMREAD_GRAYSCALE)
    result5 = HE(result3)
    cv2.imwrite(os.path.join(file_path, opt.output3), result5)

    result4 = cv2.imread(os.path.join(file_path, 'result4.png'),cv2.IMREAD_GRAYSCALE)
    result6 = HE(result4)
    cv2.imwrite(os.path.join(file_path, opt.output4), result6)
    plt.hist(result3.ravel(), bins, alpha=0.5, label='result3')
    plt.hist(result4.ravel(), bins, alpha=0.5, label='result4')
    plt.hist(result5.ravel(), bins, alpha=0.5, label='result5')
    plt.hist(result6.ravel(), bins, alpha=0.5, label='result6')
    plt.ylim(0, 25000)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(file_path, 'problem1_d.png'))
    plt.clf()
    # (e)
    print('producing images of problem 1.e...')
    result3 = cv2.imread(os.path.join(file_path, 'result3.png'),cv2.IMREAD_GRAYSCALE)
    result7 = LHE(result3, 21)
    cv2.imwrite(os.path.join(file_path, opt.output5), result7)

    result4 = cv2.imread(os.path.join(file_path, 'result4.png'),cv2.IMREAD_GRAYSCALE)
    result8 = LHE(result4, 21)
    cv2.imwrite(os.path.join(file_path, opt.output6), result8)

    plt.hist(result3.ravel(), bins, alpha=0.3, label='result3')
    plt.hist(result7.ravel(), bins, alpha=0.3, label='result7')
    plt.hist(result4.ravel(), bins, alpha=0.3, label='result4')
    plt.hist(result8.ravel(), bins, alpha=0.3, label='result8')
    plt.ylim(0, 25000)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(file_path, 'problem1_e.png'))
    plt.clf()

    #(f)
    print('producing images of problem 1.f...')
    # obeservation of image histogram
    # plt.hist(img_sample2[0:int(img_sample2.shape[0]/2),:].ravel(), bins, alpha=0.5, label='upper')
    # plt.hist(img_sample2.ravel(), bins, alpha=0.5, label='org_img')
    # plt.legend(loc='upper right')
    # plt.savefig(os.path.join(file_path, 'problem1_f.png'))

    # exponent = 3
    # f = lambda x : (((((x/255)-0.5)**exponent)+0.5**exponent)/(0.5**exponent)/2)*255
    plt.plot([0,255],[0,255])
    plt.plot([0,50,100,255],[0,100,100,255])
    plt.savefig(os.path.join(file_path, 'problem1_f_transfer_function.png'))
    plt.clf()
    def piecewise(x):
        if x<=50:
            return x*2
        elif x>50 and x<=100:
            return 100
        else :
            return x
            # return (x-1)*180/155+75

    result9  = np.zeros(img_sample2.shape)
    for i in range(img_sample2.shape[0]):
        for j in range(img_sample2.shape[1]):
            result9[i,j] = piecewise(img_sample2[i,j])
    cv2.imwrite(os.path.join(file_path, opt.output7), result9)

    plt.hist(img_sample2.ravel(), bins, alpha=0.5, label='org_img')
    plt.hist(result9.ravel(), bins, alpha=0.5, label='result9')
    plt.ylim(0, 25000)
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(file_path, 'problem1_f.png'))
    
    
 
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='sample2.png', help='input image')
    parser.add_argument('--output1', default='result3.png', help='output image')
    parser.add_argument('--output2', default='result4.png', help='output image')
    parser.add_argument('--output3', default='result5.png', help='output image')
    parser.add_argument('--output4', default='result6.png', help='output image')
    parser.add_argument('--output5', default='result7.png', help='output image')
    parser.add_argument('--output6', default='result8.png', help='output image')
    parser.add_argument('--output7', default='result9.png', help='output image')
    opt = parser.parse_args()
    main(opt)