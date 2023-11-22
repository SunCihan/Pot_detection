# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2020/6/2 16:28 
"""
import cv2, copy
import numpy as np
np.set_printoptions(suppress=True, precision=4)
from collections import OrderedDict


def cal_tow_box_IOU(tar1, tar2):
	# https://blog.csdn.net/guyuealian/article/details/86488008
	# 计算包含大小图目标的ROI面积
	xmin1, ymin1, xmax1, ymax1, score1 = tar1
	xmin2, ymin2, xmax2, ymax2, score2 = tar2
	# 计算每个矩形的面积
	s1 = (xmax1 - xmin1) * (ymax1 - ymin1)  # C的面积
	s2 = (xmax2 - xmin2) * (ymax2 - ymin2)  # G的面积
	# 计算相交矩形
	xmin = max(xmin1, xmin2)
	ymin = max(ymin1, ymin2)
	xmax = min(xmax1, xmax2)
	ymax = min(ymax1, ymax2)
	w = max(0, xmax - xmin)
	h = max(0, ymax - ymin)
	area = w * h  # C∩G的面积
	# iou = area / (s1 + s2 - area)       # 交际/并集
	iou = area / np.minimum(s1, s2)      # 交际/并集
	return iou, np.array([min(xmin1, xmin2),
	                      min(ymin1, ymin2),
	                      max(xmax1, xmax2),
	                      max(ymax1, ymax2),
	                      max(score1, score2)])

def merge_result(resize_result):
	'''检测结果box合并'''
	merger_result = OrderedDict()
	for tar_name in resize_result.keys():
		if len(resize_result[tar_name]) in [0, 1]:  # box为0个或者1个的就不需要合并处理了
			merger_result[tar_name] = resize_result[tar_name]
		else:
			boxes = copy.deepcopy(resize_result[tar_name])  # 当前所有的box
			flags = np.zeros(len(boxes)).astype(bool)  # 标记当前box是否已经合并过，False没合并，True就是合并过
			merged_boxes = []  # 存储合并后的box
			for i in range(len(boxes) - 1):
				if not flags[i]:  # 这个box没有合并过
					big_box, flags[i] = boxes[i], True
					for j in range(i + 1, len(boxes)):
						if not flags[j]:
							iou, temp = cal_tow_box_IOU(big_box, boxes[j])
							if iou > 0.2:
								big_box, flags[j] = temp, True  # 得到新的大box
					merged_boxes.append(big_box)
			if not flags[-1]:  # 最后一个box如果没合并就需要加上
				merged_boxes.append(boxes[-1])
			merger_result[tar_name] = merged_boxes
	return merger_result