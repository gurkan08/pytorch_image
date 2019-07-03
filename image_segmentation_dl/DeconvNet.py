import torch
import torch.nn as nn
import torch.nn.functional as F

"""
reference paper: Noh, H., Hong, S., & Han, B. (2015). Learning deconvolution network for semantic segmentation. In Proceedings of the IEEE international conference on computer vision (pp. 1520-1528).

author: gürkan şahin, rev.: 26/09/2018

#5 down-5 up layer
"""

class DeconvNet(nn.Module):
    def __init__(self, input_nbr, label_nbr, kernel_size):
        super(DeconvNet, self).__init__()

        batchNorm_momentum = 0.1

        # down_1
        self.down1_conv1 = nn.Conv2d(input_nbr, 64, kernel_size=kernel_size, padding=1)
        self.down1_batch1 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.down1_conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=1)  # (256,256)
        self.down1_batch2 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.down1_maxpool1 = nn.MaxPool2d(2, stride=2, return_indices=True)  # (128,128)

        # down_2
        self.down2_conv1 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=1)
        self.down2_batch1 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.down2_conv2 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=1)  # (128,128)
        self.down2_batch2 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.down2_maxpool1 = nn.MaxPool2d(2, stride=2, return_indices=True)  # (64,64)

        # down_3
        self.down3_conv1 = nn.Conv2d(128, 256, kernel_size=kernel_size, padding=1)
        self.down3_batch1 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.down3_conv2 = nn.Conv2d(256, 256, kernel_size=kernel_size, padding=1)
        self.down3_batch2 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.down3_conv3 = nn.Conv2d(256, 256, kernel_size=kernel_size, padding=1)  # (64,64)
        self.down3_batch3 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.down3_maxpool1 = nn.MaxPool2d(2, stride=2, return_indices=True)  # (32,32)

        # down_4
        self.down4_conv1 = nn.Conv2d(256, 512, kernel_size=kernel_size, padding=1)
        self.down4_batch1 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.down4_conv2 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=1)
        self.down4_batch2 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.down4_conv3 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=1)  # (32,32)
        self.down4_batch3 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.down4_maxpool1 = nn.MaxPool2d(2, stride=2, return_indices=True)  # (16,16)

        # down_5
        self.down5_conv1 = nn.Conv2d(512, 1024, kernel_size=kernel_size, padding=1)
        self.down5_batch1 = nn.BatchNorm2d(1024, momentum=batchNorm_momentum)
        self.down5_conv2 = nn.Conv2d(1024, 1024, kernel_size=kernel_size, padding=1)
        self.down5_batch2 = nn.BatchNorm2d(1024, momentum=batchNorm_momentum)
        self.down5_conv3 = nn.Conv2d(1024, 1024, kernel_size=kernel_size, padding=1)  # (16,16)
        self.down5_batch3 = nn.BatchNorm2d(1024, momentum=batchNorm_momentum)
        self.down5_maxpool1 = nn.MaxPool2d(2, stride=2, return_indices=True)  # (8,8)

        # middle
        self.mid_conv1 = nn.Conv2d(1024, 1024, kernel_size=7)  # (2,2) boyutuna düşürmek için 7*7 filtre uygula
        self.mid_batch1 = nn.BatchNorm2d(1024, momentum=batchNorm_momentum)
        self.mid_conv2 = nn.Conv2d(1024, 1024, kernel_size=1)  # (2,2)
        self.mid_batch2 = nn.BatchNorm2d(1024, momentum=batchNorm_momentum)

        # mid (2,2)'yi upsampling ile (8,8) formatına dönüştür
        self.up1_upsampling1 = torch.nn.Upsample(scale_factor=4, mode='bilinear')

        # up_1
        self.up1_maxunpool1 = nn.MaxUnpool2d(2, stride=2)
        self.up1_conv1 = nn.Conv2d(1024, 512, kernel_size=kernel_size, padding=1)
        self.up1_batch1 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.up1_conv2 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=1)
        self.up1_batch2 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.up1_conv3 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=1)  # (16,16)
        self.up1_batch3 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        # up_2
        self.up2_maxunpool1 = nn.MaxUnpool2d(2, stride=2)
        self.up2_conv1 = nn.Conv2d(512, 256, kernel_size=kernel_size, padding=1)
        self.up2_batch1 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.up2_conv2 = nn.Conv2d(256, 256, kernel_size=kernel_size, padding=1)
        self.up2_batch2 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.up2_conv3 = nn.Conv2d(256, 256, kernel_size=kernel_size, padding=1)  # (16,16)
        self.up2_batch3 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        # up_3
        self.up3_maxunpool1 = nn.MaxUnpool2d(2, stride=2)
        self.up3_conv1 = nn.Conv2d(256, 128, kernel_size=kernel_size, padding=1)
        self.up3_batch1 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.up3_conv2 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=1)
        self.up3_batch2 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.up3_conv3 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=1)  # (16,16)
        self.up3_batch3 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        # up_4
        self.up4_maxunpool1 = nn.MaxUnpool2d(2, stride=2)
        self.up4_conv1 = nn.Conv2d(128, 64, kernel_size=kernel_size, padding=1)
        self.up4_batch1 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.up4_conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=1)
        self.up4_batch2 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        # up_5
        self.up5_maxunpool1 = nn.MaxUnpool2d(2, stride=2)
        self.up5_conv1 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=1)
        self.up5_batch1 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.up5_conv2 = nn.Conv2d(64, 1, kernel_size=1, padding=0)  # en son conv. padding=0 olacak !

    def forward(self, x):
        # down_1
        x_down1 = F.relu(self.down1_batch1(self.down1_conv1(x)))
        x_down1 = F.relu(self.down1_batch2(self.down1_conv2(x_down1)))
        x_down1_maxpool1, x_down1_maxpool1_indexs = self.down1_maxpool1(x_down1)

        # down_2
        x_down2 = F.relu(self.down2_batch1(self.down2_conv1(x_down1_maxpool1)))
        x_down2 = F.relu(self.down2_batch2(self.down2_conv2(x_down2)))
        x_down2_maxpool1, x_down2_maxpool1_indexs = self.down2_maxpool1(x_down2)

        # down_3
        x_down3 = F.relu(self.down3_batch1(self.down3_conv1(x_down2_maxpool1)))
        x_down3 = F.relu(self.down3_batch2(self.down3_conv2(x_down3)))
        x_down3 = F.relu(self.down3_batch3(self.down3_conv3(x_down3)))
        x_down3_maxpool1, x_down3_maxpool1_indexs = self.down3_maxpool1(x_down3)

        # down_4
        x_down4 = F.relu(self.down4_batch1(self.down4_conv1(x_down3_maxpool1)))
        x_down4 = F.relu(self.down4_batch2(self.down4_conv2(x_down4)))
        x_down4 = F.relu(self.down4_batch3(self.down4_conv3(x_down4)))
        x_down4_maxpool1, x_down4_maxpool1_indexs = self.down4_maxpool1(x_down4)

        # down_5
        x_down5 = F.relu(self.down5_batch1(self.down5_conv1(x_down4_maxpool1)))
        x_down5 = F.relu(self.down5_batch2(self.down5_conv2(x_down5)))
        x_down5 = F.relu(self.down5_batch3(self.down5_conv3(x_down5)))
        x_down5_maxpool1, x_down5_maxpool1_indexs = self.down5_maxpool1(x_down5)

        # mid
        x_mid = F.relu(self.mid_batch1(self.mid_conv1(x_down5_maxpool1)))
        x_mid = F.relu(self.mid_batch2(self.mid_conv2(x_mid)))

        # mid upsampling
        x_mid = self.up1_upsampling1(x_mid)

        # up_1
        x_up1 = self.up1_maxunpool1(x_mid, x_down5_maxpool1_indexs)  # x, maxpool_indexs
        x_up1 = F.relu(self.up1_batch1(self.up1_conv1(x_up1)))
        x_up1 = F.relu(self.up1_batch2(self.up1_conv2(x_up1)))
        x_up1 = F.relu(self.up1_batch3(self.up1_conv3(x_up1)))

        # up_2
        x_up2 = self.up2_maxunpool1(x_up1, x_down4_maxpool1_indexs)
        x_up2 = F.relu(self.up2_batch1(self.up2_conv1(x_up2)))
        x_up2 = F.relu(self.up2_batch2(self.up2_conv2(x_up2)))
        x_up2 = F.relu(self.up2_batch3(self.up2_conv3(x_up2)))

        # up_3
        x_up3 = self.up3_maxunpool1(x_up2, x_down3_maxpool1_indexs)
        x_up3 = F.relu(self.up3_batch1(self.up3_conv1(x_up3)))
        x_up3 = F.relu(self.up3_batch2(self.up3_conv2(x_up3)))
        x_up3 = F.relu(self.up3_batch3(self.up3_conv3(x_up3)))

        # up_4
        x_up4 = self.up4_maxunpool1(x_up3, x_down2_maxpool1_indexs)
        x_up4 = F.relu(self.up4_batch1(self.up4_conv1(x_up4)))
        x_up4 = F.relu(self.up4_batch2(self.up4_conv2(x_up4)))

        # up_5
        x_up5 = self.up5_maxunpool1(x_up4, x_down1_maxpool1_indexs)
        x_up5 = F.relu(self.up5_batch1(self.up5_conv1(x_up5)))
        x_up5 = self.up5_conv2(x_up5)  # sigmoidden geçir!

        return x_up5