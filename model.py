"""
Python script to prepare FasterRCNN model.
"""
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import  FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import config

def model():
    # load the COCO pre-trained model
    # we will keep the image size to the original 800 for faster training,
    # you can increase the `min_size` in `config.py` for better ressults,
    # although it may increase the training time (a trade-off)

    # transform parameters

    # min_size = 800, max_size = 1333,
    # image_mean = None, image_std = None,
    # # RPN parameters
    # rpn_anchor_generator = None, rpn_head = None,
    # rpn_pre_nms_top_n_train = 2000, rpn_pre_nms_top_n_test = 1000,
    # rpn_post_nms_top_n_train = 2000, rpn_post_nms_top_n_test = 1000,
    # rpn_nms_thresh = 0.7,
    # rpn_fg_iou_thresh = 0.7, rpn_bg_iou_thresh = 0.3,
    # rpn_batch_size_per_image = 256, rpn_positive_fraction = 0.5,
    # rpn_score_thresh = 0.0,
    # # Box parameters
    # box_roi_pool = None, box_head = None, box_predictor = None,
    # box_score_thresh = 0.05, box_nms_thresh = 0.5, box_detections_per_img = 100,
    # box_fg_iou_thresh = 0.5, box_bg_iou_thresh = 0.5,
    # box_batch_size_per_image = 512, box_positive_fraction = 0.25,
    # bbox_reg_weights = None

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=True, min_size=config.MIN_SIZE, rpn_score_thresh=0.0)
    # one class is for pot holes, and the other is background
    num_classes = 2
    # get the input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace pre-trained head with our features head
    # the head layer will classify the images based on our data input features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model