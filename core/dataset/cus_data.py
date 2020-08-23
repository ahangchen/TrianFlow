import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'evaluation'))
import numpy as np
import cv2
import copy

import torch
import pdb

class CusDataLoader:
    def __init__(self, data_dir, img_hw=(256, 832), init=True):
        self.data_dir = data_dir
        self.img_hw = img_hw

        self.data_list = self.get_data_list()
        self.num_total = len(self.data_list)

    def get_data_list(self):
        img_names = sorted(os.listdir(self.data_dir))
        data_list = []
        for i in range(self.num_total - 1):
            data = {}
            data['img1_dir'] = os.path.join(self.data_dir, img_names[i])
            data['img2_dir'] = os.path.join(self.data_dir, img_names[i + 1])
            data_list.append(data)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        '''
        Returns:
        - img		torch.Tensor (N * H, W, 3)
        - K	torch.Tensor (num_scales, 3, 3)
        - K_inv	torch.Tensor (num_scales, 3, 3)
        '''
        data = self.data_list[idx]
        # load img
        img1 = cv2.imread(data['img1_dir'])
        img2 = cv2.imread(data['img2_dir'])
        h, w = (img1.shape[0], img1.shape[1])
        w_crop = h * 4 / 3
        pad = (w - w_crop)
        img1 = cv2.resize(img1[:, pad: pad + w_crop], (640, 480))
        img = np.concatenate([img1, img2], 0)
        img = img / 255.
        img  = img.transpose(2,0,1)

        # load intrinsic
        cam_intrinsic = np.array([[640, 0, 320], [0, 640, 240], [0, 0, 1]])
        K, K_inv = cam_intrinsic, np.linalg.inv(cam_intrinsic)
        return torch.from_numpy(img).float(), torch.from_numpy(K).float(), torch.from_numpy(K_inv).float()

if __name__ == '__main__':
    pass

