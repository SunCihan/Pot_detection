# -*- coding:utf-8 _*-
"""
0 Scratch；1 Surface Scratch；2 Pit；3 Fingerprint；4 Greasy Dirt
@Author  : Xiaoqi Cheng
@Time    : 2023/6/14 20:56
"""
import glob, cv2
import numpy as np
from detect_one import *
from seg_related._01Unet_fun import *
img_paths = glob.glob('../test_original/*.png')
img_paths = [Path(img_path) for img_path in img_paths]
yolo_path = 'runs/train/5x/weights/best.pt'
unet_path = '../2023.06.16Unet_seg/IF32_Epoch700_512/0500.pt'
save_dir = 'runs\\detect'
names = ['Scratch', 'Surface Scratch', 'Pit', 'Fingerprint',  'Greasy Dirt']
# 拆图参数
row_grid = 4096  # 裁剪小图行高
row_overlap = 0  # 裁剪小图行重叠
col_grid = 4096  # 裁剪小图列宽
col_overlap = 1024  # 裁剪小图列重叠
# 其他参数
device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
half = False  # use FP16 half-precision inference
dnn = False  # use OpenCV DNN for ONNX inference
data = 'data/Pot.yaml'  # dataset.yaml path
my_colors = [(0, 0, 255),
             (255, 0, 255),
             (125, 125, 0),
             (75, 200, 0),
             (25, 150, 150)]
threshold = 50  # 图像分割结果二值化的阈值
if __name__ == '__main__':
	# %% 载入yolot， unet模型
	device = select_device(device)
	YOLO_model = DetectMultiBackend(yolo_path, device=device, dnn=dnn, data=data, fp16=half)
	UNET_model = load_unet_model(unet_path, device)
	# %% 开始遍历图像检测
	for img_path in img_paths:
		print('Dealing img_path:', img_path)
		big_img = cv2.imread(img_path)  # BGR, 原始尺寸图像4096*10000****
		box_img = big_img.copy()        # 绘制box显示、保存结果的图像
		annotator = Annotator(box_img, line_width=5, example=str(names), font_size=5)
		# %% 检测，并处理预测结果
		preds = detect_split_big(big_img, row_grid, row_overlap, col_grid, col_overlap, img_path, YOLO_model, save_dir)
		preds = merge_predicts(preds)
		for i, det in enumerate(preds):  # det的列依次为640图像中： 左上角xy, 右上角xy， 置信度， 类别
			xyxy, conf, cls = det[0:4], det[4], det[5]

			# %% 对检测到的目标进行图像分割
			ROI = big_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :]
			seg_ROI = unet_segment(ROI, UNET_model, device, my_colors[int(cls)], threshold)
			box_img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2]), :] = seg_ROI

			# %% 在图片上绘制box，另存box
			label = f'{names[int(cls)]} {conf:.2f}'
			annotator.box_label(xyxy, label, color=my_colors[int(cls)])     # 绘制box
			# save_one_box(xyxy, big_img, file=Path(save_dir + '\crops\\'+names[int(cls)]+f'\{img_path.stem}.jpg'), BGR=True)       # 将目标检测结果ROI另存

		# box_img = cv2.resize(box_img, None, fx = 0.1, fy = 0.1)
		# cv2.imshow('ss', box_img)
		# cv2.waitKey()
		save_path = os.path.join(save_dir, img_path.name).replace('.png', '.jpg')
		cv2.imwrite(save_path, box_img)
