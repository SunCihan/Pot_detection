# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2023/6/14 19:20
"""
import argparse, os, platform, sys, torch
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox
from _03CXQ_ImagePreprocessing import *


def detect_one_sml(img_path, ori_img, YOLO_model, save_dir):
	'''单独输入一副图像，不拆图直接检测'''
	os.makedirs(save_dir, exist_ok=True)
	imgsz = (640, 640)  # inference size (height, width)


	data = ROOT / 'data/Pot.yaml'  # dataset.yaml path
	conf_thres = 0.25  # confidence threshold
	iou_thres = 0.45  # NMS IOU threshold
	max_det = 1000  # maximum detections per image
	device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu

	classes = None  # filter by class: --class 0, or --class 0 2 3
	agnostic_nms = False  # class-agnostic NMS
	augment = False  # augmented inference
	visualize = False  # visualize features

	half = False  # use FP16 half-precision inference
	dnn = False  # use OpenCV DNN for ONNX inference

	# Load model
	device = select_device(device)
	# model = DetectMultiBackend(weight_path, device=device, dnn=dnn, data=data, fp16=half)
	model = YOLO_model
	stride, names, pt = model.stride, model.names, model.pt
	imgsz = check_img_size(imgsz, s=stride)  # check image size

	im = letterbox(ori_img, imgsz, stride=stride, auto=pt)[0]  # padded resize
	im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
	im = np.ascontiguousarray(im)  # contiguous

	# Dataloader
	bs = 1  # batch_size
	# Run inference
	model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
	seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
	# for path, im, ori_img, _, _ in dataset:        # im 的 shape 是 (3, 640, 640)； ori_img的 shape 是 (1024, 1024, 3)


	with dt[0]:
		im = torch.from_numpy(im).to(model.device)
		im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
		im /= 255  # 0 - 255 to 0.0 - 1.0
		if len(im.shape) == 3:
			im = im[None]  # expand for batch dim

	# Inference
	with dt[1]:
		visualize = increment_path(save_dir / Path(img_path).stem, mkdir=True) if visualize else False
		pred = model(im, augment=augment, visualize=visualize)

	# NMS
	with dt[2]:
		pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

	# 将640尺寸图像的box坐标恢复到原始图像尺寸
	for i, det in enumerate(pred):  # det的列依次为640图像中： 左上角xy, 右上角xy， 置信度， 类别
		if len(det):
			det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], ori_img.shape).round()
	return pred[0].cpu().numpy()

def detect_split_big(big_img, row_grid, row_overlap, col_grid, col_overlap, img_path, YOLO_model, save_dir):
	'''超大图像，拆分成小图，再逐一检测，合并'''
	split_dic = ImgSplit(big_img, row_grid, row_overlap, col_grid, col_overlap)     # 拆分小图
	preds = np.empty((0, 6), dtype=np.float32)      # 保存大图中检测到的所有目标
	for key in split_dic.keys():
		sml_img = np.ascontiguousarray(split_dic[key])
		pred = detect_one_sml(img_path, sml_img, YOLO_model, save_dir)      # 小图做目标检测
		if pred.shape[0] != 0:
			pred[:, 0:4] = pred[:, 0:4] + np.array([[key[1],key[0],key[1],key[0]]]) # 小图检测结果合并到大图上
			preds = np.vstack((preds, pred))
	return preds



