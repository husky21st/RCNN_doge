import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from typing import Tuple
import os
import numpy as np
from utils import get_target_labels, get_iou


class TrainDataset(Dataset):
    def __init__(self, alldata, roi_dir, train_mode='net', label_idx=None) -> None:
        super().__init__()
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.alldata = alldata
        if train_mode == 'net':
            self.data = self.make_train_net_data(roi_dir)
        elif train_mode == 'svm':
            if label_idx is not None:
                self.data = self.make_train_svm_data(label_idx, roi_dir)
            else:
                raise RuntimeError('svm mode need label_idx option')
        else:
            raise RuntimeError('train_mode option is only allowed "net" and "svm"')

    def __getitem__(
        self,
        index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_idx = self.data[index]['img_idx']
        img = self.alldata[img_idx][0]
        roi = self.data[index]['roi']
        label = self.data[index]['label']

        return self.transform(F.to_tensor(img.crop(roi))), torch.from_numpy(label)

    def __len__(self) -> int:
        return len(self.data)

    def make_train_net_data(self, roi_dir):
        positive_data = list()
        negative_data = list()
        for i, (_, a) in enumerate(self.alldata):
            annotation = a['annotation']
            target_num = annotation['filename'].split('.')[0]
            proposal_rois = np.load(os.path.join(roi_dir, target_num + '.npy')).astype(np.int32)
            target_label_data = get_target_labels(annotation['object'])
            iou = get_iou(target_label_data, proposal_rois)
            positive_bboxes = iou >= 0.5
            for j, positive in enumerate(positive_bboxes.T):
                target_label = np.zeros(21, dtype=np.float32)
                if positive.any():
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

    def make_train_svm_data(self, label_idx, roi_dir):
        positive_data = list()
        negative_data = list()
        target_max = 0
        for i, (_, a) in enumerate(self.alldata):
            annotation = a['annotation']
            target_label_data = get_target_labels(annotation['object'], label_idx)
            if not target_label_data:
                continue
            target_max = max(target_max, len(target_label_data))
            for label in target_label_data:
                positive_data.append({'img_idx': i, 'roi': label['roi'], 'label': np.array(1, dtype=np.int32)})

        hard_negative_num = target_max * 5  # Hard negative mining
        for i, (_, a) in enumerate(self.alldata):
            annotation = a['annotation']
            target_label_data = get_target_labels(annotation['object'], label_idx)
            target_num = annotation['filename'].split('.')[0]
            proposal_rois = np.load(os.path.join(roi_dir, target_num + '.npy')).astype(np.int32)
            hard_negative_num = min(hard_negative_num, len(proposal_rois))
            if target_label_data:
                iou = get_iou(target_label_data, proposal_rois)
                negative_bboxes = iou < 0.3
                for j in range(hard_negative_num):
                    idx = (len(proposal_rois) * j // hard_negative_num)
                    if idx > len(proposal_rois):
                        break
                    proposal_roi = proposal_rois[idx]
                    negative_bbox = negative_bboxes[:, idx]
                    if negative_bbox.all():
                        negative_data.append({'img_idx': i, 'roi': proposal_roi, 'label':  np.array(0, dtype=np.int32)})
            else:
                for j in range(hard_negative_num):
                    idx = (len(proposal_rois) * j // hard_negative_num)
                    if idx > len(proposal_rois):
                        break
                    proposal_roi = proposal_rois[idx]
                    negative_data.append({'img_idx': i, 'roi': proposal_roi, 'label': np.array(0, dtype=np.int32)})
        return positive_data + negative_data


class PredictDataset(Dataset):
    def __init__(self, imgdata, roi_dir) -> None:
        super().__init__()
        self.transform = transforms.Compose([transforms.Resize((224, 224)),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])
        self.imgdata = imgdata
        self.img = self.imgdata[0]
        self.data = self.make_data(roi_dir)

    def __getitem__(
            self,
            index: int
    ) -> Tuple[torch.Tensor, np.ndarray]:

        roi = self.data[index]

        return self.transform(F.to_tensor(self.img.crop(roi))), roi

    def __len__(self) -> int:
        return len(self.data)

    def make_data(self, roi_dir):
        annotation = self.imgdata[1]['annotation']
        target_num = annotation['filename'].split('.')[0]
        return np.load(os.path.join(roi_dir, target_num + '.npy')).astype(np.int32)
