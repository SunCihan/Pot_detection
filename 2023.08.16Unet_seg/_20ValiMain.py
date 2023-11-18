#-*- coding:utf-8 _*-
"""
Validatation
@Author  : Xiaoqi Cheng
@Time    : 2021/1/11 9:52
"""
import torch,glob
from _99Normalization import *
from _99SaveLoad import *
from _02MultiPotDatasetLoader import *
from _03FCN import *
from _21CalEvaluationMetrics import *

SaveFolder = "lr0.00001_IF32_Epoch700_512_cat1"
Width = 2

# %% Load Pot and model
FolderPath = '../data_seg'
TrainDataset, TrainDataLoader, ValDataset, ValDataLoader = PotDatasetLoader(FolderPath, TrainBatchSize=1, ValBatchSize=1,
                                                                             TrainNumWorkers=0, ValNumWorkers=0, Width=Width)
ModelNames = ['0700']

for ModelName in ModelNames:
	SaveFilePath = os.path.join(SaveFolder, 'result' + ModelName + '.txt')
	if os.path.exists(SaveFilePath):
		print(SaveFilePath+' already exist!')
		continue

	Model = Net(InputChannels=2, OutputChannels=1, InitFeatures=32, WithActivateLast=True, ActivateFunLast=torch.sigmoid).cuda()
	Model.load_state_dict(torch.load(os.path.join(SaveFolder, ModelName+'.pt'), map_location = 'cuda'))

	# %% Evaluation
	Model.eval()
	torch.set_grad_enabled(False)
	OutputS = []
	LabelS = []
	for Iter, (Input, Label, SampleName) in enumerate(ValDataLoader):
		# print(SampleName)
		Input = torch.cat(Input, dim=1)
		InputImg = Input.float().to('cuda')
		OutputImg = Model(InputImg)
		# Record
		Output = OutputImg.detach().cpu().numpy()[0]
		Label = Label.detach().cpu().numpy()[0]
		OutputS.append(Output)
		LabelS.append(Label)

		OutputImg = OutputImg.cpu().numpy()[0, 0]
		OutputImg = (OutputImg*255).astype(np.uint8)
		# TMImg = TMImg.numpy()[0][0]
		# TMImg = (Normalization(TMImg) * 255).astype(np.uint8)
		# ResultImg = cv2.cvtColor(TMImg, cv2.COLOR_GRAY2RGB)
		ResultImg = cv2.cvtColor(OutputImg, cv2.COLOR_GRAY2RGB)
		LabelImg = (Normalization(Label[0]) * 255).astype(np.uint8)
		ResultImg[..., 2] = cv2.add(ResultImg[..., 2], OutputImg)

	# %% Calculate evaluation metrics
	OutputFlatten = np.vstack(OutputS).ravel()
	LabelFlatten = np.vstack(LabelS).ravel()

	_, _, MF, mAP = PRC_mAP_MF(LabelFlatten, OutputFlatten, ShowPRC = False)
	print('MF:', MF)
	print('mAP:', mAP)


	with open(SaveFilePath, 'w') as f:
		# f.write('AUC: '+str(AUC)+'\n')
		f.write('MF: '+str(MF)+'\n')
		f.write('mAP: '+str(mAP)+'\n')



