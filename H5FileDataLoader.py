import torch.utils.data as data
import torch
import h5py
import numpy as np

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')

    def __getitem__(self, index):
        return torch.from_numpy(np.expand_dims(self.data[index,0,:,:], axis=0)).float(), torch.from_numpy(np.expand_dims(self.target[index,0,:,:], axis=0)).float()

    def __len__(self):
        return self.data.shape[0]


class DatasetFromHdf5_clone(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        hf = h5py.File(file_path)
        self.data = hf.get('data')
        self.target = hf.get('label')

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index,:,:,:]).float(), torch.from_numpy(self.target[index,:,:,:]).float()

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    dataset = DatasetFromHdf5('/home/ty/code/pytorch-edsr/data/edsr_x4.h5')

    dataLoader = torch.utils.data.DataLoader(dataset,
                                             batch_size=32,
                                             shuffle=True,
                                             num_workers=int(1))

    for i, (data, gt) in enumerate(dataLoader):
        print(i)