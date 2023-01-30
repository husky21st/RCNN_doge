import torch
import torch.nn as nn
import os
from datetime import datetime
from torchvision.models import alexnet, AlexNet_Weights


class MyModel(object):
	def __init__(self, model_dir):
		self.model_dir = model_dir
		self.net = alexnet(weights=AlexNet_Weights.DEFAULT)
		self.net.classifier[6] = nn.Linear(in_features=4096, out_features=21, bias=True)
		self.DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
		self.net.to(self.DEVICE)
		print('Device is :' + str(self.DEVICE))

	def eval(self):
		self.net.eval()
		model_path = os.path.join(self.model_dir, 'cnn_model.pth')
		try:
			self.net.load_state_dict(torch.load(model_path, map_location=self.DEVICE))
			print('load model : cnn_model.pth')
			self.net.classifier[6] = nn.Identity()
		except Exception as e:
			print('load model error')
			dt_now = datetime.now()
			with open('error.out', mode='a+') as f:
				f.write(f'{dt_now}\nload model error\n{e}\n')

	def save_model(self, model_path):
		self.net.classifier[6] = nn.Identity()
		torch.save(self.net.net.state_dict(), model_path)
