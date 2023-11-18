#-*- coding:utf-8 _*-
"""
运行本程序得到测试分割结果
@Author  : Xiaoqi Cheng
@Time    : 2021/1/15 9:52
"""
from seg_related._03FCN import *
from torchvision.transforms import transforms, InterpolationMode
from PIL import Image
import cv2

ImgResize = (512,512) # image size

ValImgTransform = transforms.Compose([
	transforms.Resize(ImgResize),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.46], std=[0.10]),
])
ValLabelTransform = transforms.Compose([
	transforms.Resize(ImgResize, interpolation=InterpolationMode.NEAREST),
	transforms.ToTensor(),
])

def load_unet_model(path, device):
	UNET_model = Net(InputChannels=1, OutputChannels=1, InitFeatures=32, WithActivateLast=True, ActivateFunLast=torch.sigmoid).to(device)
	UNET_model.load_state_dict(torch.load(path, map_location = 'cuda'))
	UNET_model.eval()
	torch.set_grad_enabled(False)
	return UNET_model

def unet_segment(ROI, UNET_model, device, color, threshold):
	# %% 裁剪一个box，进行图像分割
	img_ori = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
	img_enh = img_ori
	ORI_ImgResize = img_ori.shape
	img_ori_PIL, img_enh_PIL = Image.fromarray(img_ori), Image.fromarray(img_enh)
	img_ori_tensor = torch.unsqueeze(ValImgTransform(img_ori_PIL), dim=0)
	img_enh_tensor = torch.unsqueeze(ValImgTransform(img_enh_PIL), dim=0)
	#  输入神经网络检测
	Input = torch.cat([img_enh_tensor, ], dim=1)
	InputImg = Input.float().to(device)
	OutputImg = UNET_model(InputImg)
	OutputImg = OutputImg.cpu().numpy()[0, 0]
	OutputImg = (OutputImg*255).astype(np.uint8)
	OutputImg = cv2.resize(OutputImg, ORI_ImgResize[::-1])
	ResultImg = cv2.cvtColor(img_enh, cv2.COLOR_GRAY2RGB)
	ROI_points = np.argwhere(OutputImg > threshold)
	for i in range(3):
		ResultImg[ROI_points[:, 0], ROI_points[:, 1], i] = ResultImg[ROI_points[:, 0], ROI_points[:, 1], i]//2 + color[i]//2
	return ResultImg