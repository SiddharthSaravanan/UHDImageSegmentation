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

steps:

1. Download repository
2. MATLAB - Install matlab engine - https://www.mathworks.com/matlabcentral/answers/346068-how-do-i-properly-install-matlab-engine-using-the-anaconda-package-manager-for-python
          - download dollargradiend and add it to MATLAB's path - https://github.com/pdollar/toolbox | https://pdollar.github.io/toolbox/

3.Download the big test and validation datasets and put them in ./data/BIG
4.Download the pretrained models/ frozen checkpoints here: https://drive.google.com/drive/folders/1-6VULibtyUuDjasDbmvJCPndVZ5I2BM7?usp=sharing
  Put the two folders in .\models\research\deeplab\datasets\pascal_voc_seg\init_models

5. run the following 6 commands in order after downloading all dependencies (see conda_environment.yml)

-> python preprocess.py
-> set PYTHONPATH=.\models;.\models\research;.\models\research\slim;.\models\research\deeplab\datasets
-> set PATH=%PATH%;%PYTHONPATH%
-> python models/research/deeplab/datasets/build_voc2012_data.py --image_folder="./models/research/deeplab/datasets/im" --semantic_segmentation_folder="./models/research/deeplab/datasets/gt" --list_folder="./models/research/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/ImageSets/Segmentation" --image_format="png" --output_dir="./models/research/deeplab/datasets/pascal_voc_seg/tfrecord"

-> python models/research/deeplab/vis.py --logtostderr --vis_split="val" --model_variant="xception_65" --atrous_rates=12 --atrous_rates=24 --atrous_rates=36 --output_stride=8 --decoder_output_stride=4 --vis_crop_size="513,513" --dataset="pascal_voc_seg" --checkpoint_dir="./models/research/deeplab/datasets/pascal_voc_seg/init_models/deeplabv3_pascal_trainval" --vis_logdir="./models/research/deeplab/datasets/pascal_voc_seg/exp/train_on_trainval_set/vis" --dataset_dir="./models/research/deeplab/datasets/pascal_voc_seg/tfrecord" --max_number_of_iterations=1

-> python test.py

The results of the refinement can be found in ./results





