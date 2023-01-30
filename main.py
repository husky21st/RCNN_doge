import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import cv2
import argparse

from model import MyModel
from regionProposal import search
from dataset import TrainDataset, PredictDataset
from utils import *


def get_VOCData(path, year):
	download = False
	train = datasets.VOCDetection(root=path, year=year, image_set='train', download=download)
	val = datasets.VOCDetection(root=path, year=year, image_set='val', download=download)
	trainval = datasets.VOCDetection(root=path, year=year, image_set='trainval', download=download)
	test = datasets.VOCDetection(root=path, year=year, image_set='test', download=download)

	return {'train': train, 'val': val, 'trainval': trainval, 'test': test}


def train_net(model, train_data, roi_dir, model_dir, load_model=False):
	model_path = os.path.join(model_dir, 'cnn_model.pth')
	if os.path.exists(model_path) and load_model:
		print('load model : cnn_model.pth')
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

	dataset = TrainDataset(train_data, roi_dir, train_mode='net')
	loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.net.parameters(), lr=fine_tuning_lr, momentum=momentum, weight_decay=weight_decay)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

	itr = 0
	laps = max_iter // len(loader)
	model.net.train()
	for _ in tqdm(range(laps), total=laps, desc='training net'):
		for _, (batch, label) in tqdm(enumerate(loader), total=len(loader), leave=False, desc='train batch'):
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

	model.save_model(model_path)


def train_svm(model, train_data, roi_dir, model_dir, load_model=False):
	if os.path.exists(os.path.join(model_dir, 'SVMs')):
		if os.path.exists(os.path.join(model_dir, 'SVMs', 'linear_19.xml')) and load_model:
			print('load model : SVM_*.xml')
			return None
	else:
		os.mkdir(os.path.join(model_dir, 'SVMs'))

	kernels = ['linear', 'histogram']
	model.eval()
	for label_idx in tqdm(range(20), desc='training svm per class'):
		dataset = TrainDataset(train_data, roi_dir, train_mode='svm', label_idx=label_idx)
		loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
		net_output = np.zeros((len(dataset), 4096), np.float32)
		net_label = np.zeros(len(dataset), np.int32)
		with torch.inference_mode():
			for i, (batch, label) in tqdm(enumerate(loader), total=len(loader), leave=False, desc='predict feature vec for training svm'):
				batch = batch.to(model.DEVICE).float()
				outputs = model.net(batch)
				outputs = outputs.cpu().detach().numpy()
				net_output[i*BATCH_SIZE: i*BATCH_SIZE + len(outputs)] = outputs  # for memory efficiency
				label = label.numpy()
				net_label[i*BATCH_SIZE: i*BATCH_SIZE + len(label)] = label

		for kernel in tqdm(kernels, total=len(kernels), leave=False, desc='training SVMs'):
			clf = cv2.ml.SVM_create()
			clf.setType(cv2.ml.SVM_C_SVC)
			if kernel == 'linear':
				clf.setKernel(cv2.ml.SVM_LINEAR)
			elif kernel == 'histogram':
				clf.setKernel(cv2.ml.SVM_INTER)
			clf.setGamma(1)
			clf.setC(1)
			clf.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TermCriteria_EPS, 100, 1e-6))
			clf.train(net_output, cv2.ml.ROW_SAMPLE, net_label)

			svm_path = os.path.join(model_dir, 'SVMs', kernel + '_' + str(label_idx) + '.xml')
			clf.save(svm_path)
			del clf


def predict(model, test_data, roi_dir, model_dir, detection_dir):
	clf_list = {'linear': list(), 'histogram': list()}
	for i in range(20):
		for kernel in clf_list.keys():
			svm_path = os.path.join(model_dir, 'SVMs', kernel + '_' + str(i) + '.xml')
			clf_list[kernel].append(cv2.ml.SVM_load(svm_path))
	model.eval()
	for i, imgdata in tqdm(enumerate(test_data), total=len(test_data), desc='predict test per image'):
		net_output = list()
		net_label = list()
		dataset = PredictDataset(imgdata, roi_dir)
		loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
		with torch.inference_mode():
			for _, (batch, label) in tqdm(enumerate(loader), total=len(loader), leave=False, desc='predict feature vec'):
				batch = batch.to(model.DEVICE).float()
				outputs = model.net(batch)
				outputs = outputs.cpu().detach().numpy()
				net_output.append(outputs)
				net_label.append(label)

		net_output = np.concatenate(net_output)  # (N, 4096)
		net_label = np.concatenate(net_label)  # (N, 4)

		for kernel in clf_list.keys():
			predict_score = np.zeros((20, len(net_output)), dtype=np.float32)
			check_predict_score = np.zeros((20, len(net_output)), dtype=bool)
			for label_idx in tqdm(range(20), leave=False, desc='predict class'):
				clf = clf_list[kernel][label_idx]
				svm_output = clf.predict(net_output)[1].ravel()
				check_svm_output = svm_output == 1
				check_predict_score[label_idx] = check_svm_output
				svm_predict_idx = np.where(check_svm_output)[0]
				if not svm_predict_idx.size:
					continue
				ret, alpha, svidx = clf.getDecisionFunction(0)
				support_vectors = clf.getSupportVectors()
				w = support_vectors[svidx[0]]  # (i(num_of_SV), 4096)
				b = ret
				if kernel == 'linear':
					predict_score[label_idx, svm_predict_idx] = np.einsum('ik,dk->d', w, net_output[svm_predict_idx]) - b
				elif kernel == 'histogram':
					K = np.array([np.minimum(w, net_output[p_idx]) for p_idx in svm_predict_idx], dtype=np.float32)  # Histogram intersection kernel
					predict_score[label_idx, svm_predict_idx] = np.einsum('ij,djk->d', alpha, K) - b
				if (svm_output[svm_predict_idx[0]] == 1 and predict_score[label_idx][svm_predict_idx[0]] < 0) or (svm_output[svm_predict_idx[0]] == 0 and predict_score[label_idx][svm_predict_idx[0]] > 0):
					predict_score[label_idx] *= -1

			check_detect_object = check_predict_score.any(axis=0)
			check_score = predict_score[:, check_detect_object]
			suppression = nms(net_label[check_detect_object], check_score)
			detection_list = list()
			for target_id, label, score in suppression:
				target = TARGET_LIST[target_id]
				predict_data_info = target + ' ' + str(score) + ' ' + ' '.join(list(map(str, label)))
				detection_list.append(predict_data_info)

			make_dir(os.path.join(detection_dir, kernel))
			img_number = imgdata[1]['annotation']['filename'].split('.')[0]
			out_txt_file_path = os.path.join(detection_dir, kernel, img_number + '.txt')
			with open(out_txt_file_path, mode="w") as f:
				f.write("\n".join(detection_list))


def main(args):
	SEED = 39
	seed_everything(SEED)
	generator = torch.Generator()
	generator.manual_seed(SEED)

	data_dir = args.datadir
	roi_dir = args.roidir
	detection_dir = args.detectiondir
	model_dir = args.modeldir
	for dir_path in [data_dir, roi_dir, detection_dir, model_dir]:
		make_dir(dir_path)
	load_model = args.load_model
	global NUM_WORKERS
	NUM_WORKERS = args.num_workers
	load_roi = args.load_roi

	data = get_VOCData(data_dir, args.voc)
	train_data = data['trainval']
	test_data = data['test']
	if not load_roi:
		search(train_data, roi_dir)  # 5011
		search(test_data, roi_dir)  # 4952

	model = MyModel(model_dir)
	train_net(model, train_data, roi_dir, model_dir, load_model)
	train_svm(model, train_data, roi_dir, model_dir, load_model)

	predict(model, test_data, roi_dir, model_dir, detection_dir)


if __name__ == '__main__':
	parser = argparse.ArgumentParser("RCNN: VOC2007 data only")
	parser.add_argument(
		'--num_workers',
		type=int,
		default=0,
	)
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
		default='./ROI',
	)
	parser.add_argument(
		'--detectiondir',
		type=str,
		default='./detection',
	)
	parser.add_argument(
		'--modeldir',
		type=str,
		default='./model_param',
	)
	parser.add_argument(
		'--load_model',
		action='store_true',
	)
	parser.add_argument(
		'--load_roi',
		action='store_true',
	)

	args = parser.parse_args()
	main(args)
