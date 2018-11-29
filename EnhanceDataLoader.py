from torch.utils.data import Dataset
from PIL import Image
import os
import transforms
import torch
from matplotlib import pyplot as plt

class EnhanceDataset(Dataset):

    def __init__(self, left_high_dir, right_low_dir, gt_dir, image_names, transform=None):
        self.left_high_dir = left_high_dir
        self.right_low_dir = right_low_dir
        self.gt_dir = gt_dir
        self.image_names = image_names

        if transform is not None:
            self.transform = transform

    def __getitem__(self, index):
        low_img = Image.open(os.path.join(self.right_low_dir, self.image_names[index] + '.jpg'))
        high_img = Image.open(os.path.join(self.left_high_dir, self.image_names[index] + '.png'))
        gt_img = Image.open(os.path.join(self.gt_dir, self.image_names[index] + '.png'))

        low, high, gt = self.transform(low_img, high_img, gt_img)

        return low, high, gt


    def __len__(self):
        return len(self.image_names)


if __name__ == '__main__':
    train_file = 'Dataset05/train_file.txt'
    gt_root = 'Dataset05/training_aug/groundtruth'
    left_high_root = 'Dataset05/training_aug/left_high'
    right_low_root = 'Dataset05/training_aug/right_low'

    list_file = open(train_file)
    image_names = [line.strip() for line in list_file]


    dataset = EnhanceDataset(left_high_root, right_low_root, gt_root, image_names,
                   transform=transforms.Compose([
                       transforms.RandomCrop(280),
                       transforms.RandomHorizontalFlip(),
                       transforms.RandomVerticalFlip(),
                       transforms.RandomRotation(),
                    transforms.ToTensor()]))

    dataLoader = torch.utils.data.DataLoader(dataset,
                                batch_size=32,
                                shuffle=True,
                                num_workers=int(1))

    for i, (low, higt, gt) in enumerate(dataLoader):
        print(i)
        # print(low)