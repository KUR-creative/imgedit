import cv2, random
import numpy as np

import utils
from utils import path2img_path

def sqr_origin_yx(h,w, size):
    return random.randrange(h-size+1), random.randrange(w-size+1)
def img2squares(size, img):
    return img

import unittest
class Test(unittest.TestCase):
    def test_sqr_origin_yx(self):
        size = 3
        h,w = 7,7
        img = np.ones((h,w))[:h-size+1,:w-size+1]
        while np.any(img): # zeros -> ones
            y,x = sqr_origin_yx(h,w,size)
            img[y,x] = 0

        size = 15
        h,w = 150,70
        img = np.ones((h,w))[:h-size+1,:w-size+1]
        while np.any(img): # zeros -> ones
            y,x = sqr_origin_yx(h,w,size)
            img[y,x] = 0
            


if __name__ == '__main__':
    unittest.main()
    src_imgs_path = 'examples'
    dst_imgs_path = 'squares'
    utils.safe_copytree(src_imgs_path, dst_imgs_path,
                        ['*.jpg', '*.jpeg', '*.png'])

    img,_ = path2img_path('./examples/Kagamigami/Chapter 001 - RAW/004.jpg')
    cv2.imshow('img',img); cv2.waitKey(0)
