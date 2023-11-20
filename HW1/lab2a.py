class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # Layer definition
        self.conv1 = CONV(in_channels=3, out_channels=16, kernel_size=5, stride=1,
                             padding=2, dilation=1, groups=1,
                             bias=False, padding_mode='zeros')
        self.conv2 = CONV(in_channels=16, out_channels=16, kernel_size=3, stride=1,
                             padding=2, dilation=1, groups=1,
                             bias=False, padding_mode='zeros')
        self.conv3 = CONV(in_channels=16, out_channels=32, kernel_size=7, stride=1,
                             padding=2, dilation=1, groups=1,
                             bias=False, padding_mode='zeros')
        self.fc1   = FC(in_features=288, out_features=32, bias=True)
        self.fc2   = FC(in_features=32, out_features=10, bias=True)

    def forward(self, x):
        #print(x.size())
        # Forward pass computation
        # Conv 1
        out = F.relu(self.conv1(x))#; print(out.size())
        # MaxPool
        out = F.max_pool2d(out, 4, stride=2)#; print(out.size())
        # Conv 2
        out = F.relu(self.conv2(out)); print(out.size())
        # MaxPool
        out = F.max_pool2d(out, 3, stride=2)#; print(out.size())
        # Conv 3
        out = F.relu(self.conv3(out)); print(out.size())
        # MaxPool
        out = F.max_pool2d(out, 2, stride=2)#; print(out.size())
        # Flatten
        out = out.view(out.size(0), -1)#; print(out.size())
        # FC 1
        out = F.relu(self.fc1(out))#; print(out.size())
        # FC 2
        out = F.relu(self.fc2(out))#; print(out.size())
        return out