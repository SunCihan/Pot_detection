# -*- coding:utf-8 _*-
"""
运行_01TrainMain.py是开始训练
运行_10ShowTrainingProcess_BCE.py是显示训练loss下降过程
运行_40TestMain.py是测试数据
@Author  : Xiaoqi Cheng
@Time    : 2020/10/24 9:33
"""
import logging, os, torch
from _99Timer import *
from _02MultiPotDatasetLoader import *
from _03FCN import *
import warnings
warnings.filterwarnings('ignore')

def Train(SaveFolder):
	# %% InitParameters
	BatchSize = 5
	Epochs = 1000
	Lr = 0.00001
	LrDecay = 0.1
	LrDecayPerEpoch = 200

	ValidPerEpoch = 20
	SaveEpoch = [100,200,300,400,450,500,550,600,700,800,900,1000]        # epochs need to save temporarily
	torch.cuda.set_device(0)
	Device = torch.device('cuda:0')
	# BCELossWeightCoefficient = 2

	# %% Load Multi Pot dataset
	print('**************StartTraining*****************')
	os.makedirs(SaveFolder, exist_ok=SaveFolder)
	logging.basicConfig(filename=os.path.join(SaveFolder, 'log.txt'), filemode='w', level=logging.WARNING, format='%(asctime)s %(message)s', datefmt='%Y-%m-%d-%H:%M:%S')

	FolderPath = '../data_seg'
	TrainDataset, TrainDataLoader, ValDataset, ValDataLoader = PotDatasetLoader(FolderPath, TrainBatchSize=BatchSize, ValBatchSize=BatchSize,
	                                                                             TrainNumWorkers=5, ValNumWorkers=1, ShowSample = False)
	Model = Net(InputChannels=2, OutputChannels=1, InitFeatures=32, WithActivateLast=True, ActivateFunLast=torch.sigmoid).to(Device)
	# Model.load_state_dict(torch.load('init.pt', map_location=Device))
	# torchsummary.summary(Model, input_size=(1, 512, 512))
	# %% Init optimizer and learning rate
	CriterionBCELoss = nn.BCELoss().to(Device)
	for Epoch in range(1, Epochs + 1):
		End = timer(8)
		if Epoch == 1:
			Optimizer = torch.optim.Adam(Model.parameters(), lr=Lr)
			LrScheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=LrDecayPerEpoch, gamma=LrDecay)

		# %% Validation
		if Epoch % ValidPerEpoch == 0 or Epoch == 1:
			Model.eval()
			torch.cuda.empty_cache()
			BCELoss = 0
			print('\tValidate:', end='>>', flush=True)
			for Iter, (InputImgs, Label, SampleName) in enumerate(ValDataLoader):
				print(Iter, end=' ', flush=True)
				InputImg = torch.cat(InputImgs, dim=1)
				InputImg = InputImg.float().to(Device)
				Label = Label.float().to(Device)
				# Weight = Label * (BCELossWeightCoefficient - 1) + 1
				# CriterionBCELoss.weight = Weight
				with torch.set_grad_enabled(False):
					OutputImg = Model(InputImg)
					BatchBCELoss = CriterionBCELoss(OutputImg, Label)
					BCELoss += BatchBCELoss.item() * len(SampleName)
			AveBCELoss = BCELoss / ValDataset.__len__()
			print('\t\t\t\tValidat_AveBCELoss:{0:04f}'.format(AveBCELoss))
			logging.warning('\t\tValidate_AveBCELoss:{0:04f}'.format(AveBCELoss))

		# %% Training
		Model.train()
		# torch.cuda.empty_cache()
		BCELoss = 0
		print('Epoch:%d, LR:%.8f ' % (Epoch, LrScheduler.get_lr()[0]), end='>> ', flush=True)
		for Iter, (InputImgs, Label, SampleName) in enumerate(TrainDataLoader):
			print(Iter, end=' ', flush=True)
			InputImg = torch.cat(InputImgs, dim=1)
			InputImg = InputImg.float().to(Device)
			Label = Label.float().to(Device)
			# Weight = Label * (BCELossWeightCoefficient - 1) + 1
			# CriterionBCELoss.weight = Weight
			Optimizer.zero_grad()
			with torch.set_grad_enabled(True):
				OutputImg = Model(InputImg)
				BatchBCELoss = CriterionBCELoss(OutputImg, Label)
				BatchBCELoss.backward()
				Optimizer.step()
				BCELoss += BatchBCELoss.item() * len(SampleName)
		AveBCELoss = BCELoss / TrainDataset.__len__()
		print('\tTrain_AveBCELoss:{0:04f}'.format(float(AveBCELoss)))
		logging.warning('\tTrain_AveBCELoss:{0:04f}'.format(float(AveBCELoss)))
		End('Epoch')


		# %% Saving
		if Epoch in SaveEpoch:
			torch.save(Model.state_dict(), os.path.join(SaveFolder, '{0:04d}.pt'.format(Epoch)))
			print("Save path:", os.path.join(SaveFolder, '{0:04d}.pt'.format(Epoch)))
		LrScheduler.step()

	log = logging.getLogger()
	for hdlr in log.handlers[:]:
		if isinstance(hdlr, logging.FileHandler):
			log.removeHandler(hdlr)

if __name__ == '__main__':
	torch.backends.cudnn.benchmark = True
	SaveFolder = 'lr0.00001_IF32_Epoch700_512_cat1'
	Train(SaveFolder)


