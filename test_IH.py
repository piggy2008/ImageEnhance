import torch
import torch.nn as nn
import torch.nn.functional as F
from models import SRNet, DINetwok
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
import math
from skimage.measure import compare_psnr, compare_ssim

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
            img = img.float().sub(0.48102492).div(255.)
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


    # model = SRNet().to(device)
    model = DINetwok().to(device)

    model.load_state_dict(torch.load('model/checkpoint_2018-12-11 22:35:17/model_epoch_400.pth'))


    model.eval()
    # model = load_part_of_model(model, 'checkpoint/model_epoch_5.pth')
    size = 200
    psnr_total = []
    ssim_total = []
    pix_total = []
    for name in image_names:
        left_high = Image.open(os.path.join(left_high_root, name + '0.jpg'))
        right_low = Image.open(os.path.join(right_low_root, name + '1.jpg'))
        gt = Image.open(os.path.join(gt_root, name + '1.png'))
        # left_high = left_high.crop((0, 0, size, size))
        # right_low = right_low.crop((0, 0, size, size))
        # gt = gt.crop((0, 0, size, size))
        w, h = gt.size
        iter_w = math.ceil(w / size)
        iter_h = math.ceil(h / size)
        crop_w = size
        crop_h = size
        count = 0
        psnr_img = []
        ssim_img = []
        pix_mean = []
        for i in range(iter_w):
            if i == iter_w - 1:
                crop_w = w - i * size
            for j in range(iter_h):
                if j == iter_h - 1:
                    crop_h = h - j * size

                crop_high = left_high.crop((i * 200, j * 200, i * 200 + crop_w,  j * 200 + crop_h))
                crop_low = right_low.crop((i * 200, j * 200, i * 200 + crop_w, j * 200 + crop_h))
                crop_gt = gt.crop((i * 200, j * 200, i * 200 + crop_w, j * 200 + crop_h))
                # crop_high.save('Dataset05/' + str(count) + '.jpg')
                count += 1
                print('(' + str(i*200) + ',' + str(j*200) + ')')

                input1, input2, target = toTensor(crop_low, crop_high, crop_gt)
                input1 = input1.unsqueeze(0)
                input1 = input1.type(torch.cuda.FloatTensor)
                input2 = input2.unsqueeze(0)
                input2 = input2.type(torch.cuda.FloatTensor)
                result = model(input1, input2)
                result = result.data.cpu().numpy()
                #result = result * 255
                #result = result.astype(np.uint8)
                target = target.data.cpu().numpy()
                # input1 = input1.data.cpu().numpy()
                # plt.subplot(1, 3, 1)
                # plt.imshow(result[0, 0, :, :])
                # plt.subplot(1, 3, 2)
                # plt.imshow(target[0, :, :])
                # plt.subplot(1, 3, 3)
                # plt.imshow(input1[0, 0, :, :])
                # plt.show()
                pix_mean.append(np.mean(target))
                psnr = compare_psnr(target[0, :, :], result[0, 0, :, :])
                ssim = compare_ssim(result[0, 0, :, :], target[0, :, :])
                psnr_img.append(psnr)
                ssim_img.append(ssim)

        psnr_img_mean = np.mean(psnr_img)
        ssim_img_mean = np.mean(ssim_img)
        pix_img_mean = np.mean(pix_mean)

        psnr_total.append(psnr_img_mean)
        ssim_total.append(ssim_img_mean)
        pix_total.append(pix_img_mean)

    final_psnr = np.mean(psnr_total)
    final_ssim = np.mean(ssim_total)
    final_pix_mean = np.mean(pix_total)

    print('psnr:', final_psnr)
    print('ssim:', final_ssim)
    print('pixel mean:', final_pix_mean)



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