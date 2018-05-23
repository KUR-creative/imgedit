import pathlib, shutil, sys, cv2
import numpy as np

import utils
from fp import pipe, cmap, cfilter

utils.help_option(
'''
separator: separate rgb/grayscale image files.

create new argv[2] directory(tree structure preserved), 
separate images in argv[1], and move rgb imgs to new directory.

ex)
python separator.py ./data/examples/ ./data/rgb
                    ^~~~~~~~~~~~~~~~ ^~~~~~~~~~  
                    origin img dir   new directory for rgb imgs.
'''
)

path2img_path = lambda path: (cv2.imread(path), path)
is_grayscale = (lambda img: 
                  np.all(img[:,:,0] == img[:,:,1]) and 
                  np.all(img[:,:,1] == img[:,:,2]))

if __name__ == '__main__':
    mixed_imgs_path = sys.argv[1]
    rgb_imgs_path = sys.argv[2]

    utils.safe_copytree(mixed_imgs_path, rgb_imgs_path, '*.*')
    f = pipe(utils.file_paths, 
             cmap(path2img_path),
             cfilter(lambda img_path: img_path[0] is not None),
             cfilter(lambda img_path: not is_grayscale(img_path[0])))
    old_parent_dir = pathlib.Path(mixed_imgs_path).parts[-1]

    timer = utils.ElapsedTimer('moving in')
    for img, img_path in f(mixed_imgs_path):
        new_path = utils.make_dstpath(img_path, old_parent_dir,
                                      rgb_imgs_path)
        #print(img_path, old_parent_dir, new_path)
        shutil.move(img_path,new_path)
    timer.elapsed_time()
