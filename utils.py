import torch
import numpy as np
import os
import random
from tqdm import tqdm

BATCH_SIZE = 128
NUM_WORKERS = 14

TARGET_LIST = {'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6, 'aeroplane': 7, 'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13, 'bottle': 14, 'chair': 15, 'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19}


def seed_everything(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms = True
	torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2 ** 32
	np.random.seed(worker_seed)
	random.seed(worker_seed)


def get_target_labels(bbox_data):
	target_info = []
	for obj in bbox_data:
		target = obj['name']
		target_idx = TARGET_LIST[target]
		roi = np.array([int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']), int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax'])], dtype=np.int32)

		target_info.append({'target': target, 'target_idx': target_idx, 'roi': roi})
	return target_info


def get_iou(target_label_data, b):
	iou_list = list()
	for target_label_info in target_label_data:
		a = target_label_info['roi']
		a_area = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
		b_area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

		abx_mn = np.maximum(a[0], b[:, 0])  # xmin
		aby_mn = np.maximum(a[1], b[:, 1])  # ymin
		abx_mx = np.minimum(a[2], b[:, 2])  # xmax
		aby_mx = np.minimum(a[3], b[:, 3])  # ymax
		w = np.maximum(0, abx_mx - abx_mn + 1)
		h = np.maximum(0, aby_mx - aby_mn + 1)
		intersect = w * h

		iou = intersect / (a_area + b_area - intersect)
		iou_list.append(iou)
	return np.array(iou_list, dtype=np.float32)
