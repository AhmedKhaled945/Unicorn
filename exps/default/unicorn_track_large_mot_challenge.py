#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2022 ByteDance. All Rights Reserved.
import os
from unicorn.exp import ExpTrack
"""
The main setting used in the Unicorn paper (ConvNext-Large Backbone)
We load weights pretrained on COCO with input resolution of 800x1280
"""
class Exp(ExpTrack):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.backbone_name = "convnext_large"
        self.in_channels = [384, 768, 1536]
        self.pretrain_name = "unicorn_det_convnext_large_800x1280"
        self.mot_test_name = "motchallenge"
        self.num_classes = 10
        self.mhs = False
        self.data_dir = "/content/drive/MyDrive/yolox_putney_training/home/ec2-user/SageMaker/Ahmed_Yolox_Trials/coco_formated_putney_extended"
        self.train_ann = "train.json"
        self.val_ann = "validation.json"
        self.mot_only=True
        self.mot_weight=1
        self.task = "uni" 
