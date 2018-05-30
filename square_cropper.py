import cv2, random
import numpy as np
from pymonad.Reader import curry

import utils

def sqr_origin_yx(h, w, size):
    return random.randrange(h-size+1), random.randrange(w-size+1)
@curry
def img2sqr_piece(size, img):
    h,w = img.shape[:2]
    y,x = sqr_origin_yx(h, w, size)
    return img[y:y+size,x:x+size]
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
        h,w = 100,70
        img = np.ones((h,w))[:h-size+1,:w-size+1]
        while np.any(img): # zeros -> ones
            y,x = sqr_origin_yx(h,w,size)
            img[y,x] = 0
            
    def test_img2sqr_piece(self):
        img = np.ones((400,200,3))
        size = 5
        for _ in range(500):
            square = img2sqr_piece(size,img)
            self.assertEqual(square.shape[:2], (size,size))
        size = 15
        for _ in range(1000):
            square = img2sqr_piece(size,img)
            self.assertEqual(square.shape[:2], (size,size))

    def test_curried_img2sqr_piece(self):
        img = np.ones((400,200,3))
        img2_5x5piece = img2sqr_piece(5)
        for _ in range(500):
            square = img2_5x5piece(img)
            self.assertEqual(square.shape[:2], (5,5))
        img2_15x15piece = img2sqr_piece(15)
        for _ in range(1000):
            square = img2_15x15piece(img)
            self.assertEqual(square.shape[:2], (15,15))



if __name__ == '__main__':
    unittest.main()
    src_imgs_path = 'examples'
    dst_imgs_path = 'squares'
    utils.safe_copytree(src_imgs_path, dst_imgs_path,
                        ['*.jpg', '*.jpeg', '*.png'])

    img,_ = path2img_path('./examples/Kagamigami/Chapter 001 - RAW/004.jpg')
    cv2.imshow('img',img); cv2.waitKey(0)
