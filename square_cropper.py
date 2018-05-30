import random, os, cv2
import numpy as np
from pymonad.Reader import curry

import utils
from itertools import repeat, cycle
from fp import pipe, cmap, cfilter, flatten, crepeat, cflatMap

def sqr_origin_yx(h, w, size):
    return random.randrange(h-size+1), random.randrange(w-size+1)
@curry
def img2sqr_crop(size, img):
    h,w = img.shape[:2]
    y,x = sqr_origin_yx(h, w, size)
    return img[y:y+size,x:x+size]
def path2crop_path(path, num, delimiter='_', ext='png'):
    name, _ = os.path.splitext(path)
    name = name + delimiter + str(num)
    return name + '.' + ext

import unittest
class Test(unittest.TestCase):
    def test_path2piece_path(self):
        #gen_path = utils.file_paths('./examples/')
        self.assertEqual(
          path2crop_path(
            '/examples/Kake Gurui/Chapter 013 - RAW/016.jpg', 0),
          '/examples/Kake Gurui/Chapter 013 - RAW/016_0.png')
        self.assertEqual(
          path2crop_path(
            '/examples/Kake Gurui/Chapter 013 - RAW/016.jpg', 12),
          '/examples/Kake Gurui/Chapter 013 - RAW/016_12.png')

    
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
            square = img2sqr_crop(size,img)
            self.assertEqual(square.shape[:2], (size,size))
        size = 15
        for _ in range(1000):
            square = img2sqr_crop(size,img)
            self.assertEqual(square.shape[:2], (size,size))

    def test_curried_img2sqr_piece(self):
        img = np.ones((400,200,3))
        img2_5x5piece = img2sqr_crop(5)
        for _ in range(500):
            square = img2_5x5piece(img)
            self.assertEqual(square.shape[:2], (5,5))
        img2_15x15piece = img2sqr_crop(15)
        for _ in range(1000):
            square = img2_15x15piece(img)
            self.assertEqual(square.shape[:2], (15,15))

    def test_img2sqr_pieces(self):
        img = cv2.imread('./examples/Kagamigami/Chapter 001 - RAW/004.jpg')
        img2_128x128piece = img2sqr_crop(128)
        imgs = repeat(img,5)
        '''
        for square in map(img2_128x128piece, imgs):
            cv2.imshow('sqr',square); cv2.waitKey(0)
        '''



if __name__ == '__main__':
    #unittest.main()
    '''
    src_imgs_path = 'examples'
    dst_imgs_path = 'squares'
    utils.safe_copytree(src_imgs_path, dst_imgs_path,
                        ['*.jpg', '*.jpeg', '*.png'])
    num_crop = 3
    gen_crop_id = cycle(range(num_crop))
    img2_128x128crop = img2sqr_crop(128)
    gen = \
    pipe(utils.file_paths,
         cmap(lambda path: (cv2.imread(path), path)),
         cfilter(lambda img_path: img_path[0] is not None),
         cflatMap(lambda img_path: repeat(img_path,num_crop)),
         cmap(lambda img_path: \
                (img2_128x128crop(img_path[0]), 
                 path2crop_path(img_path[1],next(gen_crop_id)))),
         cmap(lambda img_path: \
                (img_path[0], 
                 utils.make_dstpath(img_path[1],
                                    src_imgs_path,
                                    dst_imgs_path))))

    for img,path in gen(src_imgs_path):
        cv2.imwrite(path,img)
        cv2.imshow('img',img);cv2.waitKey(0)
        print(path)
    '''
    #src_imgs_path = 'examples'
    src_imgs_path = 'e'
    num_crop = 3
    img2_128x128crop = img2sqr_crop(128)
    gen = pipe(utils.file_paths,
               cmap(lambda path: cv2.imread(path)),
               cfilter(lambda img: img is not None),
               cflatMap(crepeat(num_crop)),
               cmap(lambda img:img2_128x128crop(img)))

    print(len(list(gen(src_imgs_path))))
    for img in gen(src_imgs_path):
        #cv2.imwrite(path,img)
        cv2.imshow('img',img);cv2.waitKey(0)

             
