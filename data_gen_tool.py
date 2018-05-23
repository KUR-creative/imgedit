# origin --(handwork)--> mask, answer
import cv2, pathlib
import numpy as np
from ui import get_mask_ui
import utils

utils.safe_copytree('./data/examples/', './data/new', '*.*')
print(list(utils.file_paths('./data/examples/')))
path1 = list(utils.file_paths('./data/examples/'))[0]
print(path1)

print(utils.replace_part_of(path1, 'examples', 'new'))
