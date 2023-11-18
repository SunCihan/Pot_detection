#-*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2021/1/12 21:08
"""
import sys, cv2
import numpy as np
np.set_printoptions(suppress=True, precision=4)
import numpy.random as r
from tqdm import tqdm
import sklearn.metrics as m
import matplotlib.pyplot as plt

def PRC_mAP_MF(LabelFlatten, OutputFlatten, ShowPRC = False):
	Precision, Recall, th = m.precision_recall_curve(LabelFlatten, OutputFlatten)
	F1ScoreS = 2 * (Precision * Recall) / ((Precision + Recall) + sys.float_info.min)
	MF = F1ScoreS[np.argmax(F1ScoreS)]  # Maximum F-measure at optimal dataset scale
	mAP = m.average_precision_score(LabelFlatten, OutputFlatten)
	if ShowPRC:
		plt.figure('Precision Recall curve')
		plt.plot(Recall, Precision)
		plt.ylim([0.0, 1.0])
		plt.xlabel('Recall')
		plt.ylabel('Precision')
		# plt.show()
	return Recall, Precision, MF, mAP