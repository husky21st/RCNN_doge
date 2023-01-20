import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from torchvision.models import alexnet, AlexNet_Weights
from pprint import pprint


class MyModel(object):
	def __init__(self):
		self.weights = AlexNet_Weights.DEFAULT
		self.net = alexnet(weights=self.weights)
		self.net.classifier[6] = nn.Linear(in_features=4096, out_features=21, bias=True)
		self.DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
		self.net.to(self.DEVICE)
		print('Device is :' + str(self.DEVICE))

	def freeze(self):
		for param in self.net.parameters():
			param.requires_grad = False

	def unfreeze(self):
		for param in self.net.parameters():
			param.requires_grad = True

	def unfreeze_last_layer(self):
		last_layer = list(self.net.children())[-1][-1]
		for param in last_layer.parameters():
			param.requires_grad = True

	# def train(self, loader):
	# 	if os.path.exists('cnn_model_20000.pth'):
	# 		return None
	# 	# Hyper Parameters
	# 	# related by https://github.com/BVLC/caffe/blob/master/models/bvlc_reference_caffenet/solver.prototxt
	# 	fine_tuning_lr = 0.001
	# 	max_iter = 450000
	# 	gamma = 0.1
	# 	step_size = 100000
	# 	display = 20
	# 	momentum = 0.9
	# 	weight_decay = 0.0005
	# 	snapshot = 10000
	# 	batch_size = 32
	#
	# 	criterion = nn.CrossEntropyLoss()
	# 	optimizer = optim.SGD(self.net.parameters(), lr=fine_tuning_lr, momentum=momentum, weight_decay=weight_decay)
	# 	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
	#
	# 	itr = 0
	# 	laps = (max_iter + len(loader) * batch_size - 1) // len(loader) * batch_size
	# 	self.net.train()
	# 	for epoch in tqdm(range(laps)):
	# 		for _, (batch, label) in tqdm(enumerate(loader)):
	# 			if itr >= max_iter:
	# 				break
	# 			for param in self.net.parameters():
	# 				param.grad = None
	#
	# 			batch = batch.to(self.DEVICE)
	# 			label = label.to(self.DEVICE).float()
	#
	# 			outputs = self.net(batch)
	#
	# 			loss = criterion(outputs, label)
	#
	# 			loss.backward()
	# 			optimizer.step()
	# 			scheduler.step()
	#
	# 			itr += batch_size
	# 			if itr and itr % snapshot == 0:
	# 				checkpoint_path = 'cnn_model_' + str(itr) + '.pth'
	# 				torch.save(self.net.state_dict(), checkpoint_path)
	#
	# 	torch.save(self.net.state_dict(), 'cnn_model.pth')

	def eval(self):
		self.net.eval()
		if os.path.exists('cnn_model.pth'):
			try:
				self.net.load_state_dict(torch.load('cnn_model.pth'))
			except Exception as e:
				with open('error.out', mode='a+') as f:
					f.write(f'load model error\n{e}\n')
		self.net.classifier[6] = nn.Identity()

	# def predict(self, img):
	# 	self.net.eval()
	# 	model = deepcopy(self.net)
	# 	if os.path.exists('cnn_model_20000.pth'):
	# 		model.load_state_dict(torch.load('cnn_model_20000.pth'))
	# 	model.classifier[6] = nn.Identity()
	# 	img = img.to(self.DEVICE)
	# 	output = model.forward(img).squeeze(dim=0)
	# 	return output.cpu().detach().numpy()


