# separate rgb/grayscale file.
import cv2, pathlib, shutil
import numpy as np
from ui import get_mask_ui
import utils

from pymonad.Reader import curry
import functools
def pipe(*functions):
    def pipe2(f, g):
        return lambda x: g(f(x))
    return functools.reduce(pipe2, functions, lambda x: x)

cfilter = curry(lambda f,xs: filter(f,xs))
cmap = curry(lambda f,xs: map(f,xs))

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
