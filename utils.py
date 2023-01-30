import torch
import numpy as np
import os
import random
from collections import OrderedDict

BATCH_SIZE = 128
NUM_WORKERS = 0

TARGET_SPACE = OrderedDict({'person': 0, 'bird': 1, 'cat': 2, 'cow': 3, 'dog': 4, 'horse': 5, 'sheep': 6, 'aeroplane': 7,
			   'bicycle': 8, 'boat': 9, 'bus': 10, 'car': 11, 'motorbike': 12, 'train': 13, 'bottle': 14, 'chair': 15,
			   'diningtable': 16, 'pottedplant': 17, 'sofa': 18, 'tvmonitor': 19})
TARGET_LIST = list(TARGET_SPACE.keys())


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


def make_dir(dir_path):
	if not os.path.exists(dir_path):
		os.mkdir(dir_path)


def get_target_labels(bbox_data, target_num=None):
	target_info = []
	if target_num is None:
		for obj in bbox_data:
			target = obj['name']
			target_idx = TARGET_SPACE[target]
			roi = np.array([int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']), int(obj['bndbox']['xmax']),
							int(obj['bndbox']['ymax'])], dtype=np.int32)
			target_info.append({'target': target, 'target_idx': target_idx, 'roi': roi})
	else:
		for obj in bbox_data:
			target = obj['name']
			target_idx = TARGET_SPACE[target]
			if target_idx == target_num:
				roi = np.array([int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']), int(obj['bndbox']['xmax']),
								int(obj['bndbox']['ymax'])], dtype=np.int32)
				target_info.append({'target': target, 'target_idx': target_idx, 'roi': roi})
	return target_info


def calculate_score(a, b, iou=True):
	a_area = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
	b_area = (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)

	abx_mn = np.maximum(a[0], b[:, 0])  # xmin
	aby_mn = np.maximum(a[1], b[:, 1])  # ymin
	abx_mx = np.minimum(a[2], b[:, 2])  # xmax
	aby_mx = np.minimum(a[3], b[:, 3])  # ymax
	w = np.maximum(0, abx_mx - abx_mn + 1)
	h = np.maximum(0, aby_mx - aby_mn + 1)
	intersect = w * h

	if iou:
		iou = intersect / (a_area + b_area - intersect)
	else:
		iou = intersect / a_area
	return iou


def get_iou(target_label_data, b):
	iou_list = list()
	for target_label_info in target_label_data:
		a = target_label_info['roi']
		iou = calculate_score(a, b)
		iou_list.append(iou)
	return np.array(iou_list, dtype=np.float32)


def nms(bboxes, scores, iou_threshold=0.2):
	suppression_list = list()
	while len(bboxes) > 0:

		max_idx = np.unravel_index(np.argmax(scores), scores.shape)
		target_id = max_idx[0]
		argmax = max_idx[1]

		bbox = bboxes[argmax]
		score = scores[max_idx]
		bboxes = np.delete(bboxes, argmax, 0)
		scores = np.delete(scores, argmax, 1)

		suppression_list.append((target_id, bbox, score))

		delete_idx = np.where(calculate_score(bbox, bboxes) >= iou_threshold)[0]

		bboxes = np.delete(bboxes, delete_idx, 0)
		scores = np.delete(scores, delete_idx, 1)

	return suppression_list
