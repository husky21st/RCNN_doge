import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from tqdm import tqdm
from torchvision.models import vgg16_bn, VGG16_BN_Weights
from pprint import pprint
from copy import copy
from sklearn.svm import SVC, LinearSVC
import argparse

from model import MyModel
from regionProposal import make_ROIs
from dataset import MyDataset
from utils import *
from API_mAP_detect_txt import *



def get_VOCData(path, year):
	download = False
	train = datasets.VOCDetection(root=path, year=year, image_set='train', download=download)
	val = datasets.VOCDetection(root=path, year=year, image_set='val', download=download)
	trainval = datasets.VOCDetection(root=path, year=year, image_set='trainval', download=download)
	test = datasets.VOCDetection(root=path, year=year, image_set='test', download=download)

	return {'train': train, 'val': val, 'trainval': trainval, 'test': test}


def train_net(model, train_data, roi_dir, load_model=False):
	if os.path.exists('cnn_model.pth') and load_model:
		return None
	# Hyper Parameters
	# related by https://github.com/BVLC/caffe/blob/master/models/bvlc_reference_caffenet/solver.prototxt
	fine_tuning_lr = 0.001
	max_iter = 450000
	gamma = 0.1
	step_size = 100000
	momentum = 0.9
	weight_decay = 0.0005
	snapshot = 10000

	dataset = MyDataset(train_data, roi_dir, train=True, train_mode='net')
	loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.net.parameters(), lr=fine_tuning_lr, momentum=momentum, weight_decay=weight_decay)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

	itr = 0
	laps = max_iter // len(loader)
	model.net.train()
	model.freeze()
	model.unfreeze_last_layer()
	for epoch in tqdm(range(laps), desc='training net'):
		if epoch == 1:
			model.unfreeze()
		for batch, label in tqdm(enumerate(loader), total=len(loader), desc='train batch'):
			if itr >= max_iter:
				break
			for param in model.net.parameters():
				param.grad = None

			batch = batch.to(model.DEVICE)
			label = label.to(model.DEVICE).float()

			outputs = model.net(batch)

			loss = criterion(outputs, label)

			loss.backward()
			optimizer.step()
			scheduler.step()

			itr += 1
			if itr % snapshot == 0 and itr:
				checkpoint_path = 'cnn_model_' + str(itr) + '.pth'
				torch.save(model.net.state_dict(), checkpoint_path)

	torch.save(model.net.state_dict(), 'cnn_model.pth')


def train_svm(SVMlist, model, train_data, roi_dir):
	net_output = list()
	net_label = list()
	features_dir = './features'

	dataset = MyDataset(train_data, roi_dir, train=True, train_mode='svm')
	loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

	model.eval()
	with torch.inference_mode():
		for itr, (batch, label) in tqdm(enumerate(loader), total=len(loader), desc='predict feature vec for training svm'):
			batch = batch.to(model.DEVICE).float()
			outputs = model.net(batch)
			outputs = outputs.cpu().detach().numpy()
			net_output.append(outputs)
			label = label.numpy()
			net_label.append(label)
			if itr % 1000 == 0 and itr:
				net_output = np.concatenate(net_output)
				net_label = np.concatenate(net_label)
				x_train_list = [list() for _ in range(20)]
				y_train_list = [list() for _ in range(20)]
				for feature_vec, label in zip(net_output, net_label):
					if label >= 0:
						x_train_list[label].append(feature_vec)
						y_train_list[label].append(1)
					else:
						x_train_list[label + 20].append(feature_vec)
						y_train_list[label + 20].append(0)
				if not os.path.exists(features_dir):
					os.mkdir(features_dir)
				for i in range(20):
					x_train = np.array(x_train_list[i])
					y_train = np.array(y_train_list[i])
					if not os.path.exists(os.path.join(features_dir, str(i))):
						os.mkdir(os.path.join(features_dir, str(i)))
					np.save(os.path.join(features_dir, str(i), str(itr)+'_x'), np.array(x_train, dtype=np.float32))
					np.save(os.path.join(features_dir, str(i), str(itr) + '_y'), np.array(y_train, dtype=np.float32))
				net_output = list()
				net_label = list()

	pprint(net_output)
	pprint(net_label)

	for i in range(20):
		# x_train = np.array(x_train_list[i])
		# y_train = np.array(y_train_list[i])
		SVMlist[i].fit(x_train, y_train)


def predict(model, SVMlist, test_data, roi_dir):
	net_output = list()
	net_label = list()

	dataset = MyDataset(test_data, roi_dir, train=False)
	loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

	with torch.inference_mode():
		for itr, (batch, label) in tqdm(enumerate(loader), desc='predict feature vec for test'):
			batch = batch.to(model.DEVICE).float()
			outputs = model.net(batch)
			outputs = outputs.cpu().detach().numpy()
			net_output.append(outputs)
			label = label.numpy()
			net_label.append(label)
	net_output = np.concatenate(net_output)
	net_label = np.concatenate(net_label)

	svm_output = list()
	svm_score = list()
	for i in range(20):
		svm_output[i] = SVMlist[i].predict(net_output)
		svm_score[i] = SVMlist[i].decision_function(net_output)
	pprint(svm_output)
	pprint(svm_score)


def main(args):
	SEED = 39
	seed_everything(SEED)
	generator = torch.Generator()
	generator.manual_seed(SEED)

	data_dir = args.datadir
	roi_dir = args.roidir
	load_model = args.load_model
	data = get_VOCData(data_dir, args.voc)
	train_data = data['trainval']
	# make_ROIs(train_data , roi_dir)  # 5011

	model = MyModel()
	train_net(model, train_data, roi_dir, load_model)

	SVMlist = [LinearSVC() for _ in range(20)]
	train_svm(SVMlist, model, train_data, roi_dir)

	# make_ROIs(data['test'], roi_dir)  # 4952
	predict(model, SVMlist, data['test'], roi_dir)


if __name__ == '__main__':
	parser = argparse.ArgumentParser("RCNN: VOC2007 data only")
	#  Test data is valid only for 2007
	parser.add_argument(
		'--voc',
		type=str,
		default='2007',
		choices=[
			'2007',
		],
	)
	parser.add_argument(
		'--datadir',
		type=str,
		default='./data',
	)
	parser.add_argument(
		'--roidir',
		type=str,
		default='./ROIs',
	)
	parser.add_argument(
		'--load_model',
		action='store_true',
	)
	args = parser.parse_args()
	main(args)
	# train_loader = make_CNN_loader(data['trainval'], roi_dir)
	# cnn_model = MyModel()
	# cnn_model.train(train_loader)
	# x_train, y_train = make_SVM_loader(data['train'], cnn_model)

	# SVMlist = [SVC() for _ in range(20)]
	# for i in range(20):
	# 	if len(np.unique(y_train[i])) == 1:
	# 		continue
	# 	SVMlist[i].fit(x_train[i], y_train[i])
	#
	# predicts = list()
	# for i in range(20):
	# 	predicts.append(SVMlist[i].predict(x_train[i]))
	# predicts = np.array(predicts)
	# iou = predicts >= 0.03
	# print(iou)
	# transform = transforms.Compose(
	# 	[transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
	# for img, a in tqdm(data['test']):
	# 	annotation = a['annotation']
	# 	target_num = annotation['filename'].split('.')[0]
	# 	api = API_mAP_detect_txt(annotation['filename'], data_dir)
	# 	proposal_bboxes = np.load(os.path.join(roi_dir, target_num + '.npy'))
	# 	target_label_info = get_target_labels(annotation['object'])
	# 	tensor_img = transforms.functional.to_tensor(img)
	# 	for label in target_label_info:
	# 		proposal_img = tensor_img[:, label['bbox'][1]:label['bbox'][3], label['bbox'][0]:label['bbox'][2]]
	#
	# 		input_img = proposal_img.unsqueeze(dim=0)
	# 		x_feature = cnn_model.predict(transform(input_img))
	# 		for i in range(20):
	# 			iou = SVMlist[i].predict(x_feature)
	# 			if iou == 1:
	# 				api.iter(label['target_idx'], iou, label['bbox'][0], label['bbox'][1], label['bbox'][2] - label['bbox'][0], label['bbox'][3] - label['bbox'][1])
	# 	api.make_txt()
