import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_mean, sign):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = float(sign) * torch.Tensor(rgb_mean)

        # Freeze the MeanShift layer
        for params in self.parameters():
            params.requires_grad = False


class _Residual_Block(nn.Module):
    def __init__(self):
        super(_Residual_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        identity_data = x
        output = self.relu(self.conv1(x))
        output = self.conv2(output)
        output *= 0.1
        output = torch.add(output, identity_data)
        return output


class SRNet(nn.Module):
    def __init__(self):
        super(SRNet, self).__init__()

        self.conv_input = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self.residual = self.make_layer(_Residual_Block, 16)

        self.conv_mid = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)

        self.conv_output = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False)

        # self.fuse = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)

        # conv-lstm
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels=32 + 32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.conv_f = nn.Sequential(
            nn.Conv2d(in_channels=32 + 32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.conv_g = nn.Sequential(
            nn.Conv2d(in_channels=32 + 32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.conv_o = nn.Sequential(
            nn.Conv2d(in_channels=32 + 32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.conv_lstm_output = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x, x2):
        # out = self.sub_mean(x)
        # x = F.upsample_bilinear(x, (192, 192))
        out = self.conv_input(x)
        residual = out
        out = self.conv_mid(self.residual(out))
        out = torch.add(out, residual)
        # out = self.upscale4x(out)
        out = self.conv_output(out)

        out2 = self.conv_input(x2)
        residual2 = out2
        out2 = self.conv_mid(self.residual(out2))
        out2 = torch.add(out2, residual2)
        # out = self.upscale4x(out)
        out2 = self.conv_output(out2)

        lstm_input = torch.cat([out, out2], 1)

        h = torch.zeros(x.size(0), 32, x.size(2), x.size(3)).type(torch.cuda.FloatTensor)
        c = torch.zeros(x.size(0), 32, x.size(2), x.size(3)).type(torch.cuda.FloatTensor)
        lstm_seq = []
        for i in range(4):
            z = torch.cat([lstm_input, h], 1)
            i = self.conv_i(z)
            f = self.conv_f(z)
            g = self.conv_g(z)
            o = self.conv_o(z)
            c = f * c + i * g
            h = o * F.tanh(c)
            output_lstm = self.conv_lstm_output(h)
            lstm_seq.append(output_lstm)


        return lstm_seq[len(lstm_seq) - 1]

class _DIN_block(nn.Module):
    def __init__(self):
        super(_DIN_block, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, stride=1, padding=1, groups=4)
        self.conv1_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=64, out_channels=48, kernel_size=3, stride=1, padding=1, groups=4)
        self.conv2_3 = nn.Conv2d(in_channels=48, out_channels=80, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        identity_data = x

        x = F.leaky_relu(self.conv1_1(x), negative_slope=0.05)
        x = F.leaky_relu(self.conv1_2(x), negative_slope=0.05)
        x = F.leaky_relu(self.conv1_3(x), negative_slope=0.05)
        slice1 = x.narrow(1, 0, 16)
        slice2 = x.narrow(1, 16, 48)

        x = F.leaky_relu(self.conv2_1(slice2), negative_slope=0.05)
        x = F.leaky_relu(self.conv2_2(x), negative_slope=0.05)
        x = F.leaky_relu(self.conv2_3(x), negative_slope=0.05)

        output = torch.add(torch.cat([identity_data, slice1], dim=1), x)

        return output

class ChannelWiseBlock(nn.Module):
    def __init__(self, in_channel, reduction=64):
        super(ChannelWiseBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c)
        y = self.fc(y)
        y = y.view(b, c, 1, 1)
        return x * y

class SpatialWiseBlock(nn.Module):
    def __init__(self, in_channel):
        super(SpatialWiseBlock, self).__init__()
        c = in_channel // 16
        self.conv_in = nn.Conv2d(in_channel, c, kernel_size=1)
        self.conv_out = nn.Conv2d(c, 1, kernel_size=1)

        #encoder
        self.conv1 = nn.Conv2d(c, 2 * c, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(2 * c)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.conv2 = nn.Conv2d(2 * c, 4 * c, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(4 * c)

        #dencoder
        self.deconv1 = nn.ConvTranspose2d(4 * c, 2 * c, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(2 * c)
        self.unpool1 = nn.MaxUnpool2d(kernel_size=2)
        self.deconv2 = nn.ConvTranspose2d(2* c, c, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(c)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.conv_in(x)
        y = self.conv1(y)
        y = F.relu(self.bn1(y), inplace=True)
        size = y.size()
        y, indices = self.pool1(y)
        y = self.conv2(y)
        y = F.relu(self.bn2(y), inplace=True)

        y = self.deconv1(y)
        y = F.relu(self.bn3(y), inplace=True)
        y = self.unpool1(y, indices, size)
        y = self.deconv2(y)
        y = F.relu(self.bn4(y), inplace=True)

        y = self.conv_out(y)
        return x * y

class DINetwok(nn.Module):
    def __init__(self):
        super(DINetwok, self).__init__()


        # low part
        self.low_conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.low_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.low_block1 = nn.Sequential(_DIN_block())
        self.low_down1 = nn.Conv2d(in_channels=80, out_channels=64, kernel_size=1, stride=1)

        self.low_channel_wise = ChannelWiseBlock(64, 16)
        # self.low_spatial_wise = SpatialWiseBlock(64)

        self.low_block2 = nn.Sequential(_DIN_block())
        self.low_down2 = nn.Conv2d(in_channels=80, out_channels=16, kernel_size=1, stride=1)

        self.low_channel_wise2 = ChannelWiseBlock(16, 4)

        # high part
        self.high_conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.high_conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.high_block1 = nn.Sequential(_DIN_block())
        self.high_down1 = nn.Conv2d(in_channels=80, out_channels=64, kernel_size=1, stride=1)

        self.high_channel_wise = ChannelWiseBlock(64, 16)
        # self.high_spatial_wise = SpatialWiseBlock(64)

        self.high_block2 = nn.Sequential(_DIN_block())
        self.high_down2 = nn.Conv2d(in_channels=80, out_channels=16, kernel_size=1, stride=1)

        self.high_channel_wise2 = ChannelWiseBlock(16, 4)

        self.fuse = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1)

        # conv-lstm
        self.conv_i = nn.Sequential(
            nn.Conv2d(in_channels=32 + 32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.conv_f = nn.Sequential(
            nn.Conv2d(in_channels=32 + 32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.conv_g = nn.Sequential(
            nn.Conv2d(in_channels=32 + 32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        self.conv_o = nn.Sequential(
            nn.Conv2d(in_channels=32 + 32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.conv_lstm_output = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, low, high):
        low = F.leaky_relu(self.low_conv1(low), negative_slope=0.05)
        low = F.leaky_relu(self.low_conv2(low), negative_slope=0.05)
<<<<<<< HEAD

        #add bn
=======
        #add batch normalization
>>>>>>> origin/master
        norm = nn.BatchNorm2d(64).cuda()
        low = norm(low)

        low = self.low_block1(low)
        low = F.leaky_relu(self.low_down1(low), negative_slope=0.05)

        low = self.low_channel_wise(low)
        # low = self.low_spatial_wise(low)

<<<<<<< HEAD
        # add bn
        norm = nn.BatchNorm2d(64).cuda()
=======
        # add batch normalization
>>>>>>> origin/master
        low = norm(low)

        low = self.low_block2(low)
        low = F.leaky_relu(self.low_down2(low), negative_slope=0.05)

        low = self.low_channel_wise2(low)

        high = F.leaky_relu(self.high_conv1(high), negative_slope=0.05)
        high = F.leaky_relu(self.high_conv2(high), negative_slope=0.05)
        high = self.high_block1(high)
        high = F.leaky_relu(self.high_down1(high), negative_slope=0.05)

        high = self.high_channel_wise(high)
        # high = self.high_spatial_wise(high)

        high = self.high_block2(high)
        high = F.leaky_relu(self.high_down2(high), negative_slope=0.05)

        high = self.high_channel_wise2(high)

        lstm_input = torch.cat([low, high], 1)

        fuse = self.fuse(lstm_input)

        h = torch.zeros(low.size(0), 32, low.size(2), low.size(3)).type(torch.cuda.FloatTensor)
        c = torch.zeros(low.size(0), 32, low.size(2), low.size(3)).type(torch.cuda.FloatTensor)
        lstm_seq = []
        for i in range(10):
            z = torch.cat([lstm_input, h], 1)
            i = self.conv_i(z)
            f = self.conv_f(z)
            g = self.conv_g(z)
            o = self.conv_o(z)
            c = f * c + i * g
            h = o * F.tanh(c)
            output_lstm = self.conv_lstm_output(h)
            lstm_seq.append(output_lstm)

        final = fuse + lstm_seq[len(lstm_seq) - 1]

        return final, lstm_seq[len(lstm_seq) - 1]

if __name__ == '__main__':
    device = torch.device('cuda')
    model = DINetwok().to(device)
    input = torch.zeros([8, 1, 100, 100]).type(torch.cuda.FloatTensor)
    input2 = torch.zeros([8, 1, 100, 100]).type(torch.cuda.FloatTensor)
    output = model(input, input2)