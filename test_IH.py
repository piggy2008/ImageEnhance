import torch
import torch.nn as nn
import torch.nn.functional as F
from models import SRNet
from edsr import Net
import os
import time
from H5FileDataLoader import DatasetFromHdf5
from EnhanceDataLoader import EnhanceDataset
import transforms
from utils import load_part_of_model
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def toTensor(picA, picB, picC):
    pics = [picA, picB, picC]
    output = []
    for pic in pics:
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
            if pic.mode == 'YCbCr':
                nchannel = 3
            else:
                nchannel = len(pic.mode)
            img = img.view(pic.size[1], pic.size[0], nchannel)
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
            img = img.float().div(255.)
        output.append(img)
    return output[0], output[1], output[2]

def test():
    device = torch.device('cuda')
    test_file = 'Dataset05/test_file.txt'
    gt_root = 'Dataset05/validation/groundtruth'
    left_high_root = 'Dataset05/validation/left_high'
    right_low_root = 'Dataset05/validation/right_low'
    list_file = open(test_file)
    image_names = [line.strip() for line in list_file]


    model = SRNet().to(device)
    model.load_state_dict(torch.load('model/checkpoint_2018-11-28 11:17:30/model_epoch_15.pth'))
    model.eval()
    # model = load_part_of_model(model, 'checkpoint/model_epoch_5.pth')
    size = 250
    for name in image_names:
        left_high = Image.open(os.path.join(left_high_root, name + '0.jpg'))
        right_low = Image.open(os.path.join(right_low_root, name + '1.jpg'))
        gt = Image.open(os.path.join(gt_root, name + '1.png'))
        left_high = left_high.crop((0, 0, size, size))
        right_low = right_low.crop((0, 0, size, size))
        gt = gt.crop((0, 0, size, size))

        input1, input2, target = toTensor(right_low, left_high, gt)
        input1 = input1.unsqueeze(0)
        input1 = input1.type(torch.cuda.FloatTensor)
        input2 = input2.unsqueeze(0)
        input2 = input2.type(torch.cuda.FloatTensor)
        result = model(input1, input2)
        result = result.data.cpu().numpy()
        # result = result * 255
        # result = result.astype(np.uint8)
        target = target.data.cpu().numpy()
        input1 = input1.data.cpu().numpy()
        plt.subplot(1, 3, 1)
        plt.imshow(result[0, 0, :, :])
        plt.subplot(1, 3, 2)
        plt.imshow(target[0, :, :])
        plt.subplot(1, 3, 3)
        plt.imshow(input1[0, 0, :, :])
        plt.show()


def save_checkpoint(model, epoch, time):
    model_folder = os.path.join('model', 'checkpoint_' + time)
    model_out_path = model_folder + "/model_epoch_{}.pth".format(epoch)
    # state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(model.state_dict(), model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == '__main__':

    test()