# separate rgb/grayscale file.

import pathlib, shutil, cv2
import numpy as np

import utils
from fp import pipe, cmap, cfilter


mixed_imgs_path = './data/examples'
rgb_imgs_path = './data/rgb'

utils.safe_copytree(mixed_imgs_path, rgb_imgs_path, '*.*')

path2img_path = lambda path: (cv2.imread(path), path)
is_grayscale = (lambda img: 
                  np.all(img[:,:,0] == img[:,:,1]) and 
                  np.all(img[:,:,1] == img[:,:,2]))

f = pipe(utils.file_paths, 
         cmap(path2img_path),
         cfilter(lambda img_path: img_path[0] is not None),
         cfilter(lambda img_path: not is_grayscale(img_path[0])))

old_parent_dir = pathlib.Path(mixed_imgs_path).parts[-1]
for img, img_path in f(mixed_imgs_path):
    new_path = utils.make_dstpath(img_path, old_parent_dir,
                                  rgb_imgs_path)
    #print(img_path, old_parent_dir, new_path)
    shutil.move(img_path,new_path)
