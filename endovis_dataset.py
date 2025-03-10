import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

class EndovisDataset(Dataset):
    def __init__(self, folderpath):
        self.folder_path = folderpath
        self.dataset_length = np.load(os.path.join(folderpath, 'metadata.npz'))["length"]

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        """
        Returns 
            data: (3, h, w)  numpy array
            truth: 
                if segmentation: a (4, h, w) numpy array, where each channel represents:
                    0: left shaft
                    1: left tool
                    2: right shaft
                    3: right tool
                if pose, a (2, 7) numpy array, where each channel represents:
                    0: left pose
                    1: right pose

            left truth: (2, h, w) numpy array if this is segmentation, or (7,) pose
            right truth: (2, h, w) numpy array if this is segmentation, or (7,) pose
        """
        loaded_data = np.load(os.path.join(self.folder_path, "%d.npz" % idx), allow_pickle=True)
        data = loaded_data["data"].astype(np.float32) / 255
        truth = np.concatenate((loaded_data['left_mask'], loaded_data['right_mask']), axis=0).astype(np.float32)
        return data, truth