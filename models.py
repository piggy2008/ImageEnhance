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
