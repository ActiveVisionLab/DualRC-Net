import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class StrongLossDataset(Dataset):
    def __init__(self, file, image_path, transforms, shrink = 1):
        self.image_path = image_path
        self.file = file
        self.shrink = shrink
        self.transforms = transforms
        with open(file, 'r') as f:
            self.pairs = f.readlines()

    def __getitem__(self, item):
        pair = self.pairs[item].strip().split(' ')
        image1 = pair[0]
        image2 = pair[1]
        image1 = plt.imread(os.path.join(self.image_path, image1))
        image2 = plt.imread(os.path.join(self.image_path, image2))


        try:
            h_1, w_1, C = image1.shape
            if C == 4:      # RGBA image, only keep first three dimensions
                image1 = image1[:, :, :3]
        except ValueError:      # gray scale image, convert them into three channels image
            image1 = image1[:, :, np.newaxis]
            image1 = np.concatenate((image1, image1, image1), axis=2)
            h_1, w_1, _ = image1.shape

        try:
            h_2, w_2, C = image2.shape
            if C == 4:
                image2 = image2[:, :, :3]
        except ValueError:
            image2 = image2[:, :, np.newaxis]
            image2 = np.concatenate((image2, image2, image2), axis=2)
            h_2, w_2, _ = image2.shape

        if self.shrink != 1:
            image1 = self.shrink_image(image1)
            image2 = self.shrink_image(image2)

        image1 = self.transforms(image1)
        image2 = self.transforms(image2)
        _, h_1_, w_1_ = image1.shape
        _, h_2_, w_2_ = image2.shape

        temp = np.array(pair[2:], dtype=float)
        correspondence = torch.tensor(temp, dtype=torch.float)
        correspondence = correspondence.reshape(-1, 4)
        correspondence[:, 0] = correspondence[:, 0] * w_1_ / w_1
        correspondence[:, 1] = correspondence[:, 1] * h_1_ / h_1
        correspondence[:, 2] = correspondence[:, 2] * w_2_ / w_2
        correspondence[:, 3] = correspondence[:, 3] * h_2_ / h_2
        source_points = torch.cat((correspondence[:, 0: 2], torch.ones((correspondence.shape[0], 1))), dim=1)
        target_points = torch.cat((correspondence[:, 2: 4], torch.ones((correspondence.shape[0], 1))), dim=1)
        n = correspondence.shape[0]
        assignment = torch.eye(n)
        dict = {
            'index':item,
            'source_image': image1,
            'target_image': image2,
            'source_points': source_points,
            'target_points': target_points,
            'assignment': assignment
        }
        return dict

    def __len__(self):
        return len(self.pairs)

    def get_matches(self, indice):
        matches = []
        for index in indice:
            pair = self.pairs[index].strip().split(' ')
            temp = np.array(pair[2:], dtype=float)
            correspondence = torch.tensor(temp, dtype=torch.float)
            correspondence = correspondence.reshape(-1, 4)
            matches.append(correspondence)

        return matches

    def shrink_image(self, image):
        H, W, _ = image.shape
        H_ = int(np.round(H*self.shrink))
        W_ = int(np.round(W*self.shrink))
        image = TF.to_pil_image(image)
        image = TF.resize(image, (H_, W_))
        return image