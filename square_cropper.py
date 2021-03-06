import unittest
import utils
import argparse
import random, os, cv2, h5py, sys
import numpy as np
from pymonad.Reader import curry

from tqdm import tqdm
from itertools import repeat, cycle, islice
from fp import pipe, cmap, cfilter, flatten, crepeat, cflatMap

#np.set_printoptions(threshold=np.nan, linewidth=np.nan)

def path2rgbimg(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def sqr_origin_yx(h, w, size):
    return random.randrange(h-size+1), random.randrange(w-size+1)

def is_cuttable(img, size):
    h,w = img.shape[:2]
    return (h - size + 1 > 0) and (w - size + 1 > 0)

@curry
def img2rand_sqr_crops(size, img):
    h,w,c = img.shape
    y,x = sqr_origin_yx(h, w, size)
    return img[y:y+size,x:x+size].reshape((size,size,c))

def hw2start_yxs(origin_yx, img_hw, piece_hw):
    ''' 
    Yield left-top coodinate of crops 
    But coord can excess range of image..
    '''
    org_y,org_x = origin_yx  
    img_h,img_w = img_hw
    piece_h,piece_w = piece_hw
    for y in range(org_y, img_h, piece_h):
        for x in range(org_x, img_w, piece_w):
            yield (y,x)
def hw2not_excess_start_yxs(origin_yx, img_hw, piece_hw):
    ''' 
    Yield left-top coodinate of crops 
    Coord cannot excess range of image
    '''
    img_h,img_w = img_hw
    piece_h,piece_w = piece_hw
    def not_excess(yx):
        y,x = yx
        return (y + piece_h - 1 < img_h) and (x + piece_w - 1< img_w)
    return filter(not_excess, 
                  hw2start_yxs(origin_yx, img_hw, piece_hw))

import itertools 
def img2sqr_crops(img, size):
    i_h, i_w = img.shape[:2]
    h = w = size
    for r_y,r_x in sorted(itertools.product([0, 0.5],
                                     [0, 0.5])):
        org_yx = int(h*r_y),int(w*r_x)
        for y,x in hw2not_excess_start_yxs( org_yx, (i_h,i_w), (h,w) ):
            #print(y,x)
            #cv2.imshow('img',img[y:y+h, x:x+w]); cv2.waitKey(0)
            yield img[y:y+h, x:x+w]

def path2crop_path(path, num, delimiter='_', ext='png'):
    name, _ = os.path.splitext(path)
    name = name + delimiter + str(num)
    return name + '.' + ext

def gen_chunk(iterable, chk_size):
    iters = [iter(iterable)] * chk_size
    return zip(*iters)

def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))

def iter_mean(prev_mean,prev_size, now_sum,now_size):
    total = prev_size + now_size
    return (prev_mean*prev_size + now_sum)/total


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--src_imgs_path',
                    help='path of source images') 
parser.add_argument('-d', '--dataset_name',
                    help='name or path of dataset to be created') 
parser.add_argument('-s', '--crop_size', type=int, 
                    help='size of crop(=128 or 256 ...).') 
parser.add_argument('-n', '--num_crop', type=int, 
                    help='number of crops per 1 original image. \
                    If 0, then it crops adaptive to img size.') 
parser.add_argument('-c', '--chk_size', type=int, 
                    help='size of chunk of data for performance.') 

def main(src_imgs_path, dataset_name, num_crop, crop_size, chk_size):
    if num_crop != 0:
        rand_sqr_crop = img2rand_sqr_crops(crop_size)
        gen \
        = pipe(utils.file_paths,
               cmap(lambda path: cv2.imread(path)),
               cfilter(lambda img: img is not None),
               cfilter(lambda img: is_cuttable(img, crop_size)),
               cmap(utils.slice1channel),
               cflatMap(crepeat(num_crop)),
               cmap(lambda img: rand_sqr_crop(img)),
               cmap(lambda img: (img / 255).astype(np.float32)),
               lambda imgs: split_every(chk_size, imgs))
    else:
        print('!')
        num_crop = 100 # big enough value..
        gen \
        = pipe(utils.file_paths,
               cmap(lambda path: cv2.imread(path)),
               cfilter(lambda img: img is not None),
               cfilter(lambda img: is_cuttable(img, crop_size)),
               cmap(utils.slice1channel),
               cflatMap(lambda img: img2sqr_crops(img, crop_size)),
               cmap(lambda img: (img / 255).astype(np.float32)),
               lambda imgs: split_every(chk_size, imgs))

    print(src_imgs_path)
    expected_num_imgs = len(list(utils.file_paths(src_imgs_path))) * num_crop
    print('-------------- SUMARY --------------')
    print('      dataset name = ', dataset_name)
    print('      size of crop = ', crop_size)
    print(' num crops per img = ', num_crop)
    print(' expected num imgs = ', expected_num_imgs)
    print('        chunk size = ', chk_size)

    f = h5py.File(dataset_name,'w')
    timer = utils.ElapsedTimer()
    #-------------------------------------------------------------
    f.create_dataset('images', 
                     (expected_num_imgs,crop_size,crop_size,1),
                       maxshape = (None,crop_size,crop_size,1),
                     chunks = (chk_size,crop_size,crop_size,1))

    mean = 0
    num_img_elems = (crop_size**2)
    for chk_no, chunk in tqdm(enumerate(gen(src_imgs_path)),
                              total=expected_num_imgs//chk_size):
        beg_idx = chk_no * chk_size 
        f['images'][beg_idx:beg_idx+len(chunk)] = chunk
        mean = iter_mean(mean, beg_idx*num_img_elems,
                         np.sum(chunk), len(chunk)*num_img_elems)
    f.create_dataset('mean_pixel_value', data=mean)

    last_chunk_size = len(chunk)
    actual_num_img = chk_no * chk_size + last_chunk_size
    if actual_num_img != expected_num_imgs:
        print(expected_num_imgs,' != ',actual_num_img)
        print('dataset resized!')
        f['images'].resize((actual_num_img,crop_size,crop_size,1))

    # [mean test code]
    #li = list(flatten(gen(src_imgs_path)))
    #real_mean = np.mean(li)
    #print('real MEAN:', real_mean)
    #print(len(li))
    #print('saved MEAN:', f['mean_pixel_value'][()])
    #-------------------------------------------------------------
    f.close()
    print('------------------------------------')
    print('dataset generated successfully.')
    msg = timer.elapsed_time()

    '''
    import mailing
    mailing.send_mail_to_kur(
        'Dataset generated successfully.',msg
    )
    '''
    

    # [load test code]
    f = h5py.File(dataset_name,'r')
    #-------------------------------------------------------------
    print('f', f['images'].shape)
    print('loaded MEAN:', f['mean_pixel_value'][()])
    #for i in range(f['images'].shape[0] ):
        #cv2.imshow('img',f['images'][i]);cv2.waitKey(0)
    cv2.imshow('img',f['images'][-1]);cv2.waitKey(0)
    #-------------------------------------------------------------
    f.close()
    '''
    '''


    '''
    src_imgs_path = 'examples'
    dst_imgs_path = 'squares'
    utils.safe_copytree(src_imgs_path, dst_imgs_path,
                        ['*.jpg', '*.jpeg', '*.png'])
    num_crop = 3
    gen_crop_id = cycle(range(num_crop))
    img2_128x128crop = img2rand_sqr_crops(128)
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

class Test(unittest.TestCase):
    @unittest.skip('no dir')
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
            square = img2rand_sqr_crops(size,img)
            self.assertEqual(square.shape[:2], (size,size))
        size = 15
        for _ in range(1000):
            square = img2rand_sqr_crops(size,img)
            self.assertEqual(square.shape[:2], (size,size))

    def test_curried_img2sqr_piece(self):
        img = np.ones((400,200,3))
        img2_5x5piece = img2rand_sqr_crops(5)
        for _ in range(500):
            square = img2_5x5piece(img)
            self.assertEqual(square.shape[:2], (5,5))
        img2_15x15piece = img2rand_sqr_crops(15)
        for _ in range(1000):
            square = img2_15x15piece(img)
            self.assertEqual(square.shape[:2], (15,15))

    @unittest.skip('no dir')
    def test_img2sqr_pieces(self):
        img = cv2.imread('./examples/Kagamigami/Chapter 001 - RAW/004.jpg')
        img2_128x128piece = img2rand_sqr_crops(128)
        imgs = repeat(img,5)
        '''
        for square in map(img2_128x128piece, imgs):
            cv2.imshow('sqr',square); cv2.waitKey(0)
        '''

    def test_iter_mean(self):
        expected = np.mean(np.arange(10))
        mean = 0
        chk_size = 2
        for idx, chk in enumerate(split_every(chk_size,np.arange(10))):
            beg_idx = idx * chk_size
            print(beg_idx)
            mean = iter_mean(mean, beg_idx, 
                             np.sum(chk),chk_size)
        self.assertAlmostEqual(expected,mean)

        expected = np.mean([np.arange(10,20),
                            np.arange(20,30),
                            np.arange(30,40),
                            np.arange(20,30),
                            np.arange(30,40),
                            np.arange(20,30),
                            np.arange(30,40),
                            np.arange(40,50),])
        mean = 0
        chk_size = 2
        for idx, chk in enumerate(split_every(chk_size,
                                              [np.arange(10,20),
                                               np.arange(20,30),
                                               np.arange(30,40),
                                               np.arange(20,30),
                                               np.arange(30,40),
                                               np.arange(20,30),
                                               np.arange(30,40),
                                               np.arange(40,50),])):
            beg_idx = idx * chk_size
            print(beg_idx)
            mean = iter_mean(mean, beg_idx*10, 
                             np.sum(chk),len(chk)*10)
        self.assertAlmostEqual(expected,mean)

class hw2start_yxsTest(unittest.TestCase):
    def test_mod0cases(self):
        self.assertEqual(list(hw2start_yxs((0,0),(100,60),(20,20))),
                         [( 0,0),( 0,20),( 0,40),
                          (20,0),(20,20),(20,40),
                          (40,0),(40,20),(40,40),
                          (60,0),(60,20),(60,40),
                          (80,0),(80,20),(80,40),])
    def test_not_mod0cases(self):
        self.assertEqual(list(hw2start_yxs((0,0),(90,49),(20,20))),
                         [( 0,0),( 0,20),( 0,40),
                          (20,0),(20,20),(20,40),
                          (40,0),(40,20),(40,40),
                          (60,0),(60,20),(60,40),
                          (80,0),(80,20),(80,40),])
    def test_not_excess(self):
        self.assertEqual(list(hw2not_excess_start_yxs(( 0, 0),
                                                      (91,49),
                                                      (20,20))),
                         [( 0,0),( 0,20),
                          (20,0),(20,20),
                          (40,0),(40,20),
                          (60,0),(60,20),])
    def test_origin_specified(self):
        self.assertEqual(list(hw2start_yxs((40,20),
                                           (90,49), 
                                           (20,20))),
                         [(40,20),(40,40),
                          (60,20),(60,40),
                          (80,20),(80,40),])

        self.assertEqual(list(hw2start_yxs((30,10),
                                           (90,49), 
                                           (20,20))),
                         [(30,10),(30,30), 
                          (50,10),(50,30), 
                          (70,10),(70,30)])
        self.assertEqual(list(hw2not_excess_start_yxs((40,20),
                                                      (91,49),
                                                      (20,20))),
                         [(40,20),(60,20)])
if __name__ == '__main__':
    #unittest.main()
    args = parser.parse_args()
    src_imgs_path = args.src_imgs_path#'H:\\DATA2\\f'
    dataset_name = args.dataset_name#'gray128.h5'
    num_crop = args.num_crop# 3
    crop_size = args.crop_size#128
    chk_size = args.chk_size#100 #00 
    main(src_imgs_path, dataset_name, num_crop, crop_size, chk_size)
