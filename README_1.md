# Ultra HD Segmentation using RandomWalk

The main idea behind this project is simple - There are two aspects of instance segmentation - (i) Boundary Detection and (ii) Object Detection. Object Detection can happen well even in lower-resolution images. However, boundary detection upscaled from lower scales do not work since boundaries get distorted. Moroever, detection of boundary is a local process (i.e local information is sufficient) while object detection is a global process (i.e global information is required.) 

We use the above inductive bias as follows:

1. Using state-of-art boundary detections we obtain boundaries on the high-res image.
2. Using state-of-art object detection we obtain seeds using low-res instance segmentations.
3. Finally combine the information using RandomWalk on high-res images. 

**Note 1:** This repo is dependent on the following git-repos [edges](https://github.com/pdollar/edges.git) and [toolbox](https://github.com/pdollar/toolbox.git). So, run the following commands after cloning.

```
git submodule add https://github.com/pdollar/edges.git ./edges/
git submodule add https://github.com/pdollar/toolbox.git ./dollar_toolbox
```

**Note 2:** This repo also uses MATLAB(R2021a) using the `matlab.engine` module.

The code in `test.py` provides the demo of our method.
```
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from tqdm.auto import tqdm
import cv2

from Main import get_UHD_segmentation
from Evaluate import iou_score
from Datasets import generate_BIG_dataset
from Segmentation import map_label_to_class

# Define the parameters here
param = {}
param['scale'] = 0.5
param['thresh_unsure'] = 0.5

BIG_test, _ = generate_BIG_dataset()
idx = 1
img_code_tmp, label, gt, img_fname, img_fname_gt = BIG_test[idx, 1:]
class_no=map_label_to_class[label]
img_pred = get_UHD_segmentation(img_fname, class_no, **param)
```