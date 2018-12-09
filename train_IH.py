import torch
import torch.nn as nn
import torch.nn.functional as F
from models import SRNet, DINetwok
import os
import time
from H5FileDataLoader import DatasetFromHdf5
from EnhanceDataLoader import EnhanceDataset
import transforms
from utils import load_part_of_model

def adjust_learning_rate(optimizer, epoch, param):
    """Sets the learning rate to the initial LR decayed by 10"""
    if param['max_epoch'] >= epoch:
        scale_running_lr = 1.0
    else:
        scale_running_lr = ((1. - float(epoch) / param['total_epoch']) ** param['lr_pow'])
    param['running_lr'] = param['lr'] * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = param['running_lr']


def train(epochs):
    device = torch.device('cuda')
    param = {}
    param['lr'] = 0.001
    param['max_epoch'] = 60
    param['total_epoch'] = epochs
    param['lr_pow'] = 0.95
    param['running_lr'] = param['lr']

    train_file = 'Dataset05/train_file.txt'
    gt_root = 'Dataset05/training_aug/groundtruth'
    left_high_root = 'Dataset05/training_aug/left_high'
    right_low_root = 'Dataset05/training_aug/right_low'
    list_file = open(train_file)
    image_names = [line.strip() for line in list_file]

    crit = nn.L1Loss()
    #crit = nn.BCELoss()

    # model = SRNet().to(device)
    model = DINetwok().to(device)
    # model.load_state_dict(torch.load('model/2018-10-26 22:11:34/50000/snap_model.pth'))
    # model = load_part_of_model_PSP_LSTM(model, param['pretrained_model'])
    # model.load_state_dict(torch.load(param['pretrained_model']))
    # optimizers = create_optimizers(nets, param)
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'])
    model.train()
    # model = load_part_of_model(model, 'checkpoint/model_epoch_5.pth')

    dataset = EnhanceDataset(left_high_root, right_low_root, gt_root, image_names,
                             transform=transforms.Compose([

                                 transforms.RandomCrop(180),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.RandomVerticalFlip(),
                                 transforms.RandomRotation(),
                                 transforms.ToTensor()]))

    training_data_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=10,
                                             shuffle=True,
                                             num_workers=int(2))
    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    for epoch in range(1, epochs + 1):
        for iteration, (low, high, target) in enumerate(training_data_loader):
            low = low.type(torch.cuda.FloatTensor)
            high = high.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.FloatTensor)

            final, lstm_branck = model(low, high)

            loss = crit(final, target)
            #loss_lstm = crit(lstm_branck, target)

            #loss = 0.9 * loss + 0.1 * loss_lstm


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if iteration % 2 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f}; lr:{:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0], param['running_lr']))
            adjust_learning_rate(optimizer, epoch, param)

        print("Epochs={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

        if epoch % 20 == 0:
            save_checkpoint(model, epoch, time_str)



def save_checkpoint(model, epoch, time):
    model_folder = os.path.join('model', 'checkpoint_' + time)
    model_out_path = model_folder + "/model_epoch_{}.pth".format(epoch)
    # state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(model.state_dict(), model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == '__main__':
    total_epochs = 300
    # data_path = '/home/ty/code/pytorch-edsr/data/edsr_x4.h5'
    train(total_epochs)