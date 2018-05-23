# separate rgb/grayscale file.

import pathlib, shutil, cv2
import numpy as np

import utils
from fp import pipe, cmap, cfilter


path2img_path = lambda path: (cv2.imread(path), path)
is_grayscale = (lambda img: 
                  np.all(img[:,:,0] == img[:,:,1]) and 
                  np.all(img[:,:,1] == img[:,:,2]))

utils.safe_copytree('./data/examples', './data/rgb', '*.*')
f = pipe(utils.file_paths, 
         cmap(path2img_path),
         cfilter(lambda img_path: img_path[0] is not None),
         cfilter(lambda img_path: not is_grayscale(img_path[0])))

for img, path in f('./data/examples'):
    new_path = utils.make_dstpath(path, 'examples', './data/rgb')
    print(path, new_path)
    shutil.move(path,new_path)
