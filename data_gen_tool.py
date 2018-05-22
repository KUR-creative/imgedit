# origin --(handwork)--> mask, answer
import cv2
import numpy as np
from ui import get_mask_ui
import utils

utils.safe_copytree('./data/examples/', './data/new', '*.*')
print(list(utils.file_paths('./data/examples/')))
print(list(utils.file_paths('./data/new/')))
