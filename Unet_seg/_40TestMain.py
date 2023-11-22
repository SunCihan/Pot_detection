#-*- coding:utf-8 _*-
"""
运行本程序得到测试分割结果
@Author  : Xiaoqi Cheng
@Time    : 2021/1/15 9:52
"""
import os, torch, cv2, random, glob
import matplotlib.pyplot as plt
from PIL import Image
from _99Normalization import *
from _03FCN import *
from _41ImagePreprocessing import *



# %% Load model
ModelFolder = 'lr0.00001_IF32_Epoch700_512_cat1'
Model = Net(InputChannels=2, OutputChannels=1, InitFeatures=32, WithActivateLast=True, ActivateFunLast=torch.sigmoid).cuda()
Model.load_state_dict(torch.load(ModelFolder+'/0200.pt', map_location = 'cuda'))
Model.eval()
torch.set_grad_enabled(False)

# %% Testing
SaveFolder = 'TestResult'
img_ori_paths = glob.glob('../data_seg/test/*.tif')
img_enh_paths = [temp.replace('.tif', '.jpg') for temp in img_ori_paths]
img_lab_paths = [temp.replace('.tif', '.png') for temp in img_ori_paths]

for img_ori_path, img_enh_path, img_lab_path in zip(img_ori_paths, img_enh_paths, img_lab_paths):
	# %% 载入需要检测图像
	print(img_ori_path, img_enh_path, img_lab_path)
	img_ori = cv2.imread(img_ori_path, cv2.IMREAD_GRAYSCALE)
	img_enh = cv2.imread(img_enh_path, cv2.IMREAD_GRAYSCALE)
	img_lab = cv2.imread(img_lab_path, cv2.IMREAD_GRAYSCALE)
	ORI_ImgResize = img_ori.shape

	img_ori_PIL, img_enh_PIL = Image.fromarray(img_ori), Image.fromarray(img_enh)
	img_ori_tensor = torch.unsqueeze(ValImgTransform(img_ori_PIL), dim=0)
	img_enh_tensor = torch.unsqueeze(ValImgTransform(img_enh_PIL), dim=0)

	# %% 输入神经网络检测
	Input = torch.cat([img_ori_tensor, img_enh_tensor], dim=1)
	#Input = torch.cat([img_ori_tensor, ], dim=1)
	InputImg = Input.float().to('cuda')
	OutputImg = Model(InputImg)
	# Generate result image
	OutputImg = OutputImg.cpu().numpy()[0, 0]
	OutputImg = (OutputImg*255).astype(np.uint8)
	OutputImg = cv2.resize(OutputImg, ORI_ImgResize[::-1])
	ResultImg = cv2.cvtColor(img_enh, cv2.COLOR_GRAY2RGB)
	ROI_points = np.argwhere(OutputImg > 20)
	ResultImg[ROI_points[:, 0], ROI_points[:, 1], 2] = 255

	os.makedirs(os.path.join(ModelFolder, SaveFolder), exist_ok=True)
	cv2.imwrite(os.path.join(ModelFolder, SaveFolder, os.path.basename(img_enh_path)), ResultImg)

