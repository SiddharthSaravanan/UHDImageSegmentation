"""
Recall there are two parts of the process -

1) Obtain bondaries for high-res image
2) Downscale the image and obtain low-res semantic information. This low-res-semantic information is
further upscaled to obtain seeds.

Here we are interested in obtaining boundaries for the high-res image.
"""

import numpy as np
import matlab.engine


def get_dollar_gradient(img_fname):
    """
    Uses the matlab functions to obtain dollar gradient.
    """

    eng = matlab.engine.start_matlab()
    eng.addpath('./edges/')
    eng.addpath('./dollar_toolbox/matlab')
    eng.addpath('./dollar_toolbox/channels')
    prob_boundary = eng.edgesDemo_man(str("."+img_fname))
    prob_boundary = np.array(prob_boundary, dtype=np.float64)
    eng.quit()
    return prob_boundary
