#-*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2019/10/24 22:03
"""
import os,glob
import numpy as np
np.set_printoptions(suppress=True, precision=8)
import matplotlib.pyplot as plt

SaveFolder = 'IF32_Epoch700_512'

plt.ion()
BCETrainLosses = []
BCEValidLosses = []
with open(os.path.join(SaveFolder, 'log.txt'), 'r') as f:
	lines = f.readlines()
	for i, line in enumerate(lines):
		print(line)
		if 'Train' in line:
			BCELoss = float(line.strip().split(':')[-1])
			BCETrainLosses.append(np.array([i, BCELoss]))

		elif 'Valid' in line:
			BCELoss = float(line.strip().split(':')[-1])
			BCEValidLosses.append(np.array([i, BCELoss]))

BCETrainLosses = np.vstack(BCETrainLosses)
BCEValidLosses = np.vstack(BCEValidLosses)



fig = plt.figure(SaveFolder)

plt.plot(BCETrainLosses[..., 0], BCETrainLosses[..., 1], label='Train:BCE Loss')
plt.plot(BCEValidLosses[..., 0], BCEValidLosses[..., 1], label='Val:BCE Loss')

# plt.yscale('log')
plt.legend()
plt.ioff()
plt.show()





