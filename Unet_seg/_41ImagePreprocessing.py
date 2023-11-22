# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2023/3/16 18:02
"""
from torchvision.transforms import transforms, InterpolationMode
import cv2
import matplotlib.pyplot as plt
import numpy as np

ImgResize = (256,256) # image size

ValImgTransform = transforms.Compose([
	transforms.Resize(ImgResize),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.46], std=[0.10]),
])
ValLabelTransform = transforms.Compose([
	transforms.Resize(ImgResize, interpolation=InterpolationMode.NEAREST),
	transforms.ToTensor(),
])