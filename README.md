# Mask R-CNN for skin cancer detection

This is a project for detecting and segmenting skin cancer lesions from the ISIC 2018 challenge (https://challenge.isic-archive.com/data/#2018). The dataset comprises RGB images of lesions and the corresponding ground truth masks.
![lesion_image](/assets/ISIC_0016714.jpg) ![lesion_segment_image](/assets/ISIC_0016714_segmentation.png) 

The project uses the Matterport (https://github.com/matterport) implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.


The repository includes (from the original Matterport implementation):
* Source code of Mask R-CNN built on FPN and ResNet101.
* Training code for MS COCO
* Pre-trained weights for MS COCO
* Jupyter notebooks to visualize the detection pipeline at every step
* ParallelModel class for multi-GPU training
* Evaluation on MS COCO metrics (AP)
* Example of training on your own dataset

  # Installation

1. Clone this repo
2. Install dependencies
   ```
   pip3 install -r requirements.txt
   ```
4. Run setup from the repository root directory
  ```
  python3 setup.py install
  ```
6. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases) and place it in the repo root directory.


