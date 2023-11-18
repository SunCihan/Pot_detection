# -*- coding:utf-8 _*-
"""
Load Multi-exposure tube contour dataset (METCD)
@Author  : Xiaoqi Cheng
@Time    : 2020/10/24 19:40
"""
import glob

import torch, os, cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms, InterpolationMode
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random


ImgResize = (512,512) # image size

# %% Data augmentation
TrainImgTransform = transforms.Compose([
	transforms.Resize(ImgResize),
	transforms.RandomAffine(degrees=(-45, 45), translate=(0.2, 0.2), scale=(0.5, 1.5), shear=10),
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),
	# transforms.RandomResizedCrop(ImgResize, scale=(1., 1.), interpolation=Image.BILINEAR),
	# transforms.RandomCrop(ImgResize),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.46], std=[0.10]),
])
TrainLabelTransform = transforms.Compose([
	transforms.Resize(ImgResize, interpolation=InterpolationMode.NEAREST),
	transforms.RandomAffine(degrees=(-45, 45), translate=(0.2, 0.2), scale=(0.5, 1.5), shear=10),
	transforms.RandomHorizontalFlip(),
	transforms.RandomVerticalFlip(),
	# transforms.RandomResizedCrop(ImgResize, scale=(1., 1.), interpolation=Image.NEAREST),
	# transforms.RandomCrop(ImgResize),
	transforms.ToTensor(),
])

ValImgTransform = transforms.Compose([
	transforms.Resize(ImgResize),
	# transforms.RandomCrop(ImgResize),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.46], std=[0.10]),
])
ValLabelTransform = transforms.Compose([
	transforms.Resize(ImgResize, interpolation=InterpolationMode.NEAREST),
	# transforms.RandomCrop(ImgResize),
	transforms.ToTensor(),
])

class PotDataset(Dataset):
	def __init__(self, DatasetFolderPath, ImgTransform, LabelTransform, ShowSample=False, ):
		self.DatasetFolderPath = DatasetFolderPath
		self.ImgTransform = ImgTransform
		self.LabelTransform = LabelTransform
		self.ShowSample = ShowSample
		# self.SampleFolders = os.listdir(self.DatasetFolderPath)
		self.img_ori_paths = glob.glob(self.DatasetFolderPath+'/*.tif')
		# self.img_enh_paths = glob.glob(self.DatasetFolderPath+'/*.jpg')
		# self.img_lab_paths = glob.glob(self.DatasetFolderPath+'/*.png')
		self.img_enh_paths = [temp.replace('.tif', '.jpg') for temp in self.img_ori_paths]
		self.img_lab_paths = [temp.replace('.tif', '.png') for temp in self.img_ori_paths]
		# for a, b, c in zip(self.img_ori_paths, self.img_enh_paths, self.img_lab_paths):
		# 	print(a, b, c)

	def __len__(self):
		return len(self.img_ori_paths)

	def __getitem__(self, item):
		img_ori = Image.fromarray(cv2.imread(self.img_ori_paths[item], cv2.IMREAD_GRAYSCALE))
		SampleName = os.path.basename(self.img_ori_paths[item])
		img_enh = Image.fromarray(cv2.imread(self.img_enh_paths[item], cv2.IMREAD_GRAYSCALE))
		img_lab = cv2.imread(self.img_lab_paths[item], cv2.IMREAD_GRAYSCALE)
		MultiImgs = [img_ori, img_enh]      #The original image and the augmented image are merged into the input
		#MultiImgs = [img_enh, ]

		# img_lab = cv2.dilate(img_lab, np.ones((3, 3), np.uint8))
		LabelImg = Image.fromarray(img_lab)

		# %% Ensure that the input data and the label have the same transformation
		seed = np.random.randint(2147483647)
		TranMultiImgs = []
		for MultiImg in MultiImgs:
			random.seed(seed)
			torch.manual_seed(seed)
			TranMultiImgs.append(self.ImgTransform(MultiImg))
		random.seed(seed)
		torch.manual_seed(seed)
		LabelImg = self.LabelTransform(LabelImg)

		# %% Show Sample
		if self.ShowSample:
			plt.figure(self.img_ori_paths[item])
			Img = TranMultiImgs[0].numpy()[0]
			Label = LabelImg.numpy()[0]
			Img = (Normalization(Img) * 255).astype(np.uint8)
			Label = (Normalization(Label) * 255).astype(np.uint8)
			Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
			Img[..., 2] = Label
			plt.imshow(Img)
			# plt.savefig('1.jpg', dip = 600)
			# exit()
			plt.show()
		return TranMultiImgs, LabelImg, SampleName

def PotDatasetLoader(FolderPath, TrainBatchSize=3, ValBatchSize=1, TrainNumWorkers=0, ValNumWorkers=0, ShowSample=False):
	TrainFolderPath = os.path.join(FolderPath, 'Train')
	ValFolderPath = os.path.join(FolderPath, 'Val')
	TrainDataset = PotDataset(TrainFolderPath, TrainImgTransform, TrainLabelTransform, ShowSample, )
	ValDataset = PotDataset(ValFolderPath, ValImgTransform, ValLabelTransform, ShowSample, )
	TrainDataLoader = DataLoader(TrainDataset, batch_size=TrainBatchSize, shuffle=True, drop_last=False, num_workers=TrainNumWorkers, pin_memory=True)
	ValDataLoader = DataLoader(ValDataset, batch_size=ValBatchSize, shuffle=False, drop_last=False, num_workers=ValNumWorkers, pin_memory=True)
	return TrainDataset, TrainDataLoader, ValDataset, ValDataLoader

def Normalization(Array):
	min = np.min(Array)
	max = np.max(Array)
	if max - min == 0:
		return Array
	else:
		return (Array - min) / (max - min)


if __name__ == '__main__':
	FolderPath = 'data_seg'
	TrainDataset, TrainDataLoader, ValDataset, ValDataLoader = PotDatasetLoader(FolderPath, TrainBatchSize=1, ValBatchSize=1,
	                                                                             TrainNumWorkers=0, ValNumWorkers=0, ShowSample=True)
	for epoch in range(1):
		for i, (Imgs, Label, TMImg, SampleName) in enumerate(TrainDataLoader):
			Img = torch.cat(Imgs, dim=1)
			print(SampleName)
			print(Img.shape)
			print(Label.shape)


