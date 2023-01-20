import cv2
import os
import numpy as np
from tqdm import tqdm
from selective_search import selective_search


def make_ROIs(dataset, save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    search(dataset, save_dir)


def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:
        pass
    elif new_image.shape[2] == 3:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def search(img_dataset, save_dir):
    mode = 'fast'
    minH = 8
    minW = 8
    for img, d in tqdm(img_dataset):
        save_file = d['annotation']['filename'].split('.')[0]
        if os.path.exists(os.path.join(save_dir, save_file + '.npy')):
            continue
        try:
            img = pil2cv(img)
        except Exception as e:
            with open('error.out', mode='a+') as f:
                f.write(f'img error:{save_file}\n{e}\n')

        boxes = []
        _boxes = selective_search(img, mode=mode, random_sort=False)
        for box in _boxes:
            if abs(box[2] - box[0]) < minW or abs(box[3] - box[1]) < minH:
                continue
            boxes.append(box)
            if len(boxes) >= 2000:
                break
        np.save(os.path.join(save_dir, save_file), np.array(boxes, dtype=np.int32))
