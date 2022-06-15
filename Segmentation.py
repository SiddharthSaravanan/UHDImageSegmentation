"""
Recall there are two parts of the process -

1) Obtain bondaries for high-res image
2) Downscale the image and obtain low-res semantic information. This low-res-semantic information is
further upscaled to obtain seeds.

Here we are interested in obtaining the low-res semantic segmentation. We use the pytorch DeepLabV3
for this purpose.
"""
import numpy as np
import pdb
import cv2
from scipy.special import softmax

labels = ["none", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
          "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tv", "tvmonitor"]
class_label = np.arange(22)
class_label[21] = 20
map_label_to_class = dict([(x, y) for (x, y) in zip(labels, list(class_label))])
map_class_to_label = dict([(y, x) for (x, y) in zip(labels, list(class_label))])

# array of SE used for thinning
b = np.zeros((8, 3, 3))
b[0] = np.array((
    [[-1, -1, -1],
     [0, 1, 0],
        [1, 1, 1]]), dtype="int")
b[1] = np.array((
    [[0, -1, -1],
     [1, 1, -1],
        [1, 1, 0]]), dtype="int")
b[2] = np.array((
    [[1, 0, -1],
     [1, 1, -1],
        [1, 0, -1]]), dtype="int")
b[3] = np.array((
    [[1, 1, 0],
     [1, 1, -1],
        [0, -1, -1]]), dtype="int")
b[4] = np.array((
    [[1, 1, 1],
     [0, 1, 0],
        [-1, -1, -1]]), dtype="int")
b[5] = np.array((
    [[0, 1, 1],
     [-1, 1, 1],
        [-1, -1, 0]]), dtype="int")
b[6] = np.array((
    [[-1, 0, 1],
     [-1, 1, 1],
        [-1, 0, 1]]), dtype="int")
b[7] = np.array((
    [[-1, -1, 0],
     [-1, 1, 1],
        [0, 1, 1]]), dtype="int")


def get_probs(prob_arr):
    if np.abs(np.sum(prob_arr) - (prob_arr.shape[0])*(prob_arr.shape[1])) < 2.0:
        return prob_arr
    print("soft")
    return softmax(prob_arr,axis=2)

def thinning(img, iterations):
    img = img.astype('uint8')
    for i in range(iterations):
        for j in range(8):
            temp = np.copy(img)
            temp = cv2.morphologyEx(src=temp, op=cv2.MORPH_HITMISS, kernel=b[j], borderType=cv2.BORDER_REPLICATE)
            img = img - temp  # center of the SE always has 1 (foreground)
        if np.count_nonzero(img) <= 40000:
            # print("thinning stopped at iteration %d" % (i))
            break
    return img

def pruning(thin,thick,prune):

    y=2
    kernel = np.ones((y,y),np.uint8)
    opening_thin = cv2.morphologyEx(thin, cv2.MORPH_OPEN, kernel)
    opening_thick = cv2.morphologyEx(thick, cv2.MORPH_OPEN, kernel)

    thin_prot = thin-opening_thin
    thick_prot = thick-opening_thick

    se = np.ones((13,13),np.uint8)
    dilation_thin_prot = cv2.dilate(thin_prot,se,iterations = 1)
    dilation_thick_prot = cv2.dilate(thick_prot,se,iterations = 1)

    union = cv2.bitwise_or(dilation_thick_prot,dilation_thin_prot)
    intersection = cv2.bitwise_and(dilation_thick_prot,dilation_thin_prot)

    # remove connected components
    xn=intersection
    xnp1=intersection
    se3 = np.ones((3,3),np.uint8)

    no_iter=0
    while True:
        no_iter+=1
        xn=xnp1
        xnp1 = cv2.dilate(xn,se3,iterations = 1)
        xnp1 = cv2.bitwise_and(xnp1,union)

        if np.array_equal(xn,xnp1) or no_iter>=prune:
            break

    union = union - xn

    thin_prot = cv2.bitwise_and(union,thin_prot)
    thick_prot = cv2.bitwise_and(union,thick_prot)

    thin = opening_thin + thin_prot
    thick = opening_thick + thick_prot

    return thin,thick

def find_extra(diffs,img,argmax_second,class_no,prob):
    not_sure = np.copy(diffs)
    perc=0
    num=0

    img2 = np.copy(argmax_second)
    img2[img2==class_no]=255
    img2[img2!=255]=0

    res = np.copy(img)

    for prob_diff in [prob]:
        if (np.count_nonzero(img))==0:
            break
        not_sure = np.copy(diffs)
        not_sure[not_sure<=prob_diff]=255
        not_sure[not_sure!=255]=0
        not_sure = not_sure.astype('uint8')

        extra_seeds = np.bitwise_and(img2,not_sure)
        res = np.bitwise_or(extra_seeds,img)

        perc = np.count_nonzero(extra_seeds)/(np.count_nonzero(res))
        break

    return (res-img)

def find_not_sure(diffs,img,argmax,class_no,prob):
    not_sure = np.copy(diffs)
    perc=0
    num=0

    for prob_diff in [prob,0]:
        if (np.count_nonzero(img))==0:
            break
        not_sure = np.copy(diffs)
        not_sure[not_sure<=prob_diff]=255
        not_sure[not_sure!=255]=0
        not_sure = not_sure.astype('uint8')

        not_sure = np.bitwise_and(not_sure,img)

        perc = np.count_nonzero(not_sure)/(np.count_nonzero(argmax==class_no))
        if perc<0.1:
            break

    return not_sure

def _get_seeds(img_prob, class_no, **param):
    """
    Given a particular class_no, return the seeded image
    Encoding:
    255 - Object
    0 - Background
    127 - Remaining/Unsure pixels
    """

    argmax = np.argmax(img_prob, axis=2)
    prob_max = np.max(img_prob, axis=2)
    img_prob[np.arange(img_prob.shape[0])[:,None],np.arange(img_prob.shape[1]),argmax] = 0

    argmax_second = np.argmax(img_prob,axis=2)
    prob_max_second = np.max(img_prob,axis=2)

    diffs = prob_max - prob_max_second

    seed_arr = np.copy(argmax)
    seed_arr[seed_arr==class_no]=-1
    seed_arr[seed_arr!=-1]=0
    seed_arr[seed_arr==-1]=255


    not_sure = find_not_sure(diffs,seed_arr,argmax,class_no,param['thresh_unsure'])
    extra = find_extra(diffs,seed_arr,argmax_second,class_no,param['thresh_unsure'])

    seed_arr[not_sure==255]=127
    seed_arr[extra==255]=127

    if param['dataset'] == 'pascal':
        cv2.imwrite('seed.png',seed_arr)
        return seed_arr
    #preparation for thinning

    thin = np.copy(seed_arr)
    thick = seed_arr

    if np.count_nonzero(seed_arr == 1) == 0: # if there are no foreground pixels after thresholding, make the not sure pixels as the foreground
        thin[thin==-1] = 1


    thin[thin<200] = 0
    thick[thick>100] = 255
    thick = 255-thick

    #thinning
    thin = thinning(thin,param['thin_iter'])
    thick = thinning(thick,param['thin_iter'])

    #pruning
    thin,thick = pruning(thin,thick,param['prune_iter'])

    # thin = np.clip(thin,0,255)
    thick = np.clip(thick,0,127)
    seed = thin + thick

    return seed


def get_seeds_UHD(img_fname, class_no, **param):
    """
    """
    img = cv2.imread(img_fname)
    output = np.load('./initial_segmentation_results/'+(img_fname.split('/')[-1]).split('.')[0]+'.npy')
    output = get_probs(output)
    img_prob = cv2.resize(output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    seed_arr = _get_seeds(img_prob, class_no, **param)
    return seed_arr

def get_seeds_PASCAL(img_fname, class_no, **param):
    img = cv2.imread(img_fname)
    output = np.load('./initial_segmentation_results/'+(img_fname.split('/')[-1]).split('.')[0]+'.npy')
    output = get_probs(output)

    seed_arr = _get_seeds(output, int(class_no), **param)
    return seed_arr

