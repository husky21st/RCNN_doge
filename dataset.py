import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as F
from typing import List, Tuple
import os
import numpy as np
from tqdm import tqdm
from utils import get_target_labels, get_iou


class MyDataset(Dataset):
    def __init__(self, alldata, roi_dir, train=True, train_mode='net') -> None:
        super().__init__()
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.alldata = alldata
        if train:
            if train_mode == 'net':
                self.data = self.make_train_net_data(roi_dir)
            elif train_mode == 'svm':
                self.data = self.make_train_svm_data(roi_dir)
            else:
                raise RuntimeError('train_mode option is only allowed "net" and "svm"')
        else:
            self.data = self.make_test_data(roi_dir)

    def __getitem__(
        self,
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_idx = self.data[index]['img_idx']
        img = self.alldata[img_idx][0]
        roi = self.data[index]['roi']
        label = self.data[index]['label']
        numpy_img = np.array(img, dtype=np.float32)
        proposal_img = numpy_img[roi[1]:roi[3], roi[0]:roi[2], :]
        proposal_img = F.to_tensor(proposal_img)
        return self.transform(proposal_img), torch.from_numpy(label)

    def __len__(self) -> int:
        return len(self.data)

    def make_train_net_data(self, roi_dir):
        positive_data = list()
        negative_data = list()
        for i, (_, a) in tqdm(enumerate(self.alldata), desc='make train net data'):
            annotation = a['annotation']
            target_num = annotation['filename'].split('.')[0]
            proposal_rois = np.load(os.path.join(roi_dir, target_num + '.npy')).astype(np.int32)
            target_label_data = get_target_labels(annotation['object'])
            iou = get_iou(target_label_data, proposal_rois)
            positive_bboxes = iou >= 0.5
            for j, positive in enumerate(positive_bboxes.T):
                target_label = np.zeros(21, dtype=np.float32)
                if np.any(positive):
                    for k in np.nonzero(positive)[0]:
                        target_label[target_label_data[k]['target_idx']] = 1
                    positive_data.append({'img_idx': i, 'roi': proposal_rois[j], 'label': target_label})
                else:
                    target_label[20] = 1
                    negative_data.append({'img_idx': i, 'roi': proposal_rois[j], 'label': target_label})

        np.random.shuffle(positive_data)
        np.random.shuffle(negative_data)
        p_size = len(positive_data)
        p_remain = 32 - p_size % 32
        positive_data.extend(positive_data[:p_remain])
        n_size = len(negative_data)
        n_remain = 96 - n_size % 96
        negative_data.extend(negative_data[:n_remain])
        data = list()
        for i in range(n_size // 96):
            last_p = ((i + 1) * 32) % p_size
            if last_p == 0:
                last_p = p_size
            data.extend(positive_data[(i * 32) % p_size: last_p])
            data.extend(negative_data[i * 96:(i + 1) * 96])
        return data

    def make_train_svm_data(self, roi_dir):
        data = list()
        for i, (_, a) in tqdm(enumerate(self.alldata), desc='make train svm data'):
            annotation = a['annotation']
            target_num = annotation['filename'].split('.')[0]
            proposal_rois = np.load(os.path.join(roi_dir, target_num + '.npy')).astype(np.int32)
            target_label_data = get_target_labels(annotation['object'])
            iou = get_iou(target_label_data, proposal_rois)

            for label in target_label_data:
                target_roi = label['roi']
                data.append({'img_idx': i, 'roi': target_roi, 'label': np.array(label['target_idx'], dtype=np.int32)})
            negative_bboxes = iou < 0.3
            for j, proposal_roi in enumerate(proposal_rois):
                negative_bbox = negative_bboxes[:, j].ravel()
                is_negative_dict = dict()
                for k, negative in enumerate(negative_bbox):
                    target_idx = target_label_data[k]['target_idx']
                    if target_idx not in is_negative_dict:
                        is_negative_dict[target_idx] = negative
                    else:
                        is_negative_dict[target_idx] = is_negative_dict[target_idx] and negative

                for target_idx, negative in is_negative_dict.items():
                    if negative:
                        data.append({'img_idx': i, 'roi': proposal_roi, 'label': np.array(target_idx - 20, dtype=np.int32)})
        return data

    def make_test_data(self, roi_dir):
        data = list()
        for i, (_, a) in tqdm(enumerate(self.alldata), desc='make test data'):
            annotation = a['annotation']
            target_num = annotation['filename'].split('.')[0]
            proposal_rois = np.load(os.path.join(roi_dir, target_num + '.npy')).astype(np.int32)
            for proposal_roi in proposal_rois:
                data.append({'img_idx': i, 'roi': proposal_roi, 'label': np.array(i, dtype=np.int32)})
        return data
