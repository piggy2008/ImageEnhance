import torch
import torch.nn as nn
import torch.nn.functional as F
from models import SRNet
from edsr import Net
import os
from H5FileDataLoader import DatasetFromHdf5

def adjust_learning_rate(optimizer, epoch, param):
    """Sets the learning rate to the initial LR decayed by 10"""
    lr = param['lr'] * (0.1 ** (epoch // 200))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(data_path, epochs):
    device = torch.device('cuda')
    param = {}
    param['lr'] = 0.0001
    crit = nn.L1Loss()
    model = SRNet().to(device)
    # model.load_state_dict(torch.load('model/2018-10-26 22:11:34/50000/snap_model.pth'))
    # model = load_part_of_model_PSP_LSTM(model, param['pretrained_model'])
    # model.load_state_dict(torch.load(param['pretrained_model']))
    # optimizers = create_optimizers(nets, param)
    optimizer = torch.optim.Adam(model.parameters(), lr=param['lr'])
    model.train()
    dataset = DatasetFromHdf5(data_path)

    training_data_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=4,
                                             shuffle=True,
                                             num_workers=int(1))

    for epoch in range(epochs):
        for iteration, (input, target) in enumerate(training_data_loader):
            input = input.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.FloatTensor)

            loss = crit(model(input), target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if iteration % 2 == 0:
                print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
        # adjust_learning_rate(optimizer, epoch - 1, param)

        print("Epochs={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

        if epoch % 5 == 0:
            save_checkpoint(model, epoch)



def save_checkpoint(model, epoch):
    model_folder = "checkpoint/"
    model_out_path = model_folder + "model_epoch_{}.pth".format(epoch)
    # state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    torch.save(model.state_dict(), model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == '__main__':
    total_epochs = 20
    data_path = '/home/ty/code/pytorch-edsr/data/edsr_x4.h5'
    train(data_path, total_epochs)