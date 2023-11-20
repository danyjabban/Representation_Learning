class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        """
        :param in_channels: number of channels for the input
        :param out_channels: number of channels for the output
        :param stride: stride of first conv2d block
        :param downsample: downsampling function (option A)
        """
        # conv1: downsamples feature map when stride != 1 (padding = 1 to ensure correct shape),
        # batch norm then ReLU
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
                                   nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1),
                                   nn.ReLU())
        # conv2: doesn't downsample feature map since already performed in self.conv1, batch norm,
        # padding=1 to ensure correct output shape (conv2 outputs same shape as conv1's output)
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1))
        self.relu2 = nn.ReLU()  # applied after adding x in the forward function

        # in case we need these
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # if number of channels for input/output mismatch, must downsample
        # if x's num_channels doesn't match out's desired n_channels:
        if x.size(dim=1) != self.out_channels:
            x = x[:, :, ::2, ::2]  # downsample feature map by a factor of 2
            # following option (A), concatenate a zeros to increase n_channels
            # (bottom-left page776 of IEEE version).
            zzzzz = torch.zeros(x.size(), device=x.device)  # get zeros of same size as x
            x_new = torch.cat((x, zzzzz), 1)  # concatenate channels
        else:
            x_new = x

        assert out.size() == x_new.size(), "out.size=={:s} != x.size=={:s}".format(str(list(out.size())),
                                                                                   str(list(x.size())))
        return self.relu2(out + x_new)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # implementing ResNet-20, so n=3
        # first conv to get 16-chan output, then (6x conv(in_channel=32), 6x conv(in_channel=16),
        # 6x conv(in_channel=8)), then average pooling, then FC, finally softmax
        # maintain feature map size, increase n_channels to 16 after 1st convolution
        self.conv_initial = nn.Sequential(nn.Conv2d(3, 16, 3, stride=1, padding=1),
                                          nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1),
                                          nn.ReLU())  # output 32*32 feature map

        # first six conv layers (ReLU, batchNorm are applied within Residual)
        self.conv32_0 = Residual(in_channels=16, out_channels=16, stride=1)  # output 32*32 feature map
        self.conv32_2 = Residual(in_channels=16, out_channels=16, stride=1)  # output 32*32 feature map
        self.conv32_4 = Residual(in_channels=16, out_channels=16, stride=1)  # output 32*32 feature map

        # feature map is 16*16 after conv16_0
        # next six conv layers, (ReLU, batchNorm are applied within Residual)
        self.conv16_0 = Residual(in_channels=16, out_channels=32, stride=2)  # output 16*16 feature map
        self.conv16_2 = Residual(in_channels=32, out_channels=32, stride=1)  # output 16*16 feature map
        self.conv16_4 = Residual(in_channels=32, out_channels=32, stride=1)  # output 16*16 feature map

        # feature map decreases to 8*8 after conv8_0
        # last six conv layers, (ReLU, batchNorm are applied within Residual)
        self.conv8_0 = Residual(in_channels=32, out_channels=64, stride=2)  # output 8*8 feature map
        self.conv8_2 = Residual(in_channels=64, out_channels=64, stride=1)  # output 8*8 feature map
        self.conv8_4 = Residual(in_channels=64, out_channels=64, stride=1)  # output 8*8 feature map
        # output after self.conv8_4 should be 64*8*8

        # global average pooling (64*(8*8) -> 64*(1))
        self.glbal_avg_pool = nn.AvgPool2d(kernel_size=8)

        # fully-conneted layer, in_channels=64, out_channels=10
        self.fc = nn.Sequential(nn.Linear(64, 10))

    def forward(self, x):
        out = self.conv_initial(x)
        out = self.conv32_0(out)
        out = self.conv32_2(out)
        out = self.conv32_4(out)

        out = self.conv16_0(out)
        out = self.conv16_2(out)
        out = self.conv16_4(out)

        out = self.conv8_0(out)
        out = self.conv8_2(out)
        out = self.conv8_4(out)

        out = self.glbal_avg_pool(out)
        out = out.view(out.size(0), -1)  # flatten before FC
        out = self.fc(out)  # includes SoftMax
        return out