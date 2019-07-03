import torch
import torch.nn as nn
import torch.nn.functional as F

"""
reference paper: Arsalan, M., Naqvi, R. A., Kim, D. S., Nguyen, P. H., Owais, M., & Park, K. R. (2018). IrisDenseNet: Robust Iris Segmentation Using Densely Connected Fully Convolutional Networks in the Images by Visible Light and Near-Infrared Light Camera Sensors. Sensors, 18(5), 1501.

author: gürkan şahin, rev.: 26/09/2018

#5 down-5 up layer
"""


class IrisDenseNet(nn.Module):
    def __init__(self, input_nbr, label_nbr, kernel_size):
        super(IrisDenseNet, self).__init__()

        batchNorm_momentum = 0.1

        # down_1
        # dense block-1
        self.down1_conv1 = nn.Conv2d(input_nbr, 64, kernel_size=kernel_size, padding=1)
        self.down1_batch1 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.down1_conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=1)
        self.down1_batch2 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        # transition layer-1
        self.down1_conv3 = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.down1_maxpool1 = nn.MaxPool2d(2, stride=2, return_indices=True)

        # down_2
        # dense block-2
        self.down2_conv1 = nn.Conv2d(64, 128, kernel_size=kernel_size, padding=1)
        self.down2_batch1 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.down2_conv2 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=1)
        self.down2_batch2 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        # transition layer-2
        self.down2_conv3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.down2_maxpool1 = nn.MaxPool2d(2, stride=2, return_indices=True)

        # down_3
        # dense blcok-3
        self.down3_conv1 = nn.Conv2d(128, 256, kernel_size=kernel_size, padding=1)
        self.down3_batch1 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.down3_conv2 = nn.Conv2d(256, 256, kernel_size=kernel_size, padding=1)
        self.down3_batch2 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.down3_conv3 = nn.Conv2d(256, 256, kernel_size=kernel_size, padding=1)
        self.down3_batch3 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        # transition lyer-3
        self.down3_conv4 = nn.Conv2d(256, 256, kernel_size=1, padding=0)
        self.down3_maxpool1 = nn.MaxPool2d(2, stride=2, return_indices=True)

        # down_4
        # dense block-4
        self.down4_conv1 = nn.Conv2d(256, 512, kernel_size=kernel_size, padding=1)
        self.down4_batch1 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.down4_conv2 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=1)
        self.down4_batch2 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.down4_conv3 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=1)
        self.down4_batch3 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        # transition layer-4
        self.down4_conv4 = nn.Conv2d(512, 512, kernel_size=1, padding=0)
        self.down4_maxpool1 = nn.MaxPool2d(2, stride=2, return_indices=True)

        # down_5
        # dense block-5
        self.down5_conv1 = nn.Conv2d(512, 1024, kernel_size=kernel_size, padding=1)
        self.down5_batch1 = nn.BatchNorm2d(1024, momentum=batchNorm_momentum)
        self.down5_conv2 = nn.Conv2d(1024, 1024, kernel_size=kernel_size, padding=1)
        self.down5_batch2 = nn.BatchNorm2d(1024, momentum=batchNorm_momentum)
        self.down5_conv3 = nn.Conv2d(1024, 1024, kernel_size=kernel_size, padding=1)
        self.down5_batch3 = nn.BatchNorm2d(1024, momentum=batchNorm_momentum)
        # transition layer-5
        self.down5_conv4 = nn.Conv2d(1024, 1024, kernel_size=1, padding=0)
        self.down5_maxpool1 = nn.MaxPool2d(2, stride=2, return_indices=True)

        # up_1
        self.up1_upsampling1 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.up1_maxunpool1 = nn.MaxUnpool2d(2, stride=2)
        self.up1_conv1 = nn.Conv2d(1024 * 2, 512, kernel_size=kernel_size, padding=1)  # cat(upsample, maxunpool) yaptığımız için filtre*2
        self.up1_conv2 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=1)
        self.up1_conv3 = nn.Conv2d(512, 512, kernel_size=kernel_size, padding=1)

        # up_2
        self.up2_upsampling1 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.up2_maxunpool1 = nn.MaxUnpool2d(2, stride=2)
        self.up2_conv1 = nn.Conv2d(512 * 2, 256, kernel_size=kernel_size, padding=1)
        self.up2_conv2 = nn.Conv2d(256, 256, kernel_size=kernel_size, padding=1)
        self.up2_conv3 = nn.Conv2d(256, 256, kernel_size=kernel_size, padding=1)

        # up_3
        self.up3_upsampling1 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.up3_maxunpool1 = nn.MaxUnpool2d(2, stride=2)
        self.up3_conv1 = nn.Conv2d(256 * 2, 128, kernel_size=kernel_size, padding=1)
        self.up3_conv2 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=1)
        self.up3_conv3 = nn.Conv2d(128, 128, kernel_size=kernel_size, padding=1)

        # up_4
        self.up4_upsampling1 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.up4_maxunpool1 = nn.MaxUnpool2d(2, stride=2)
        self.up4_conv1 = nn.Conv2d(128 * 2, 64, kernel_size=kernel_size, padding=1)
        self.up4_conv2 = nn.Conv2d(64, 64, kernel_size=kernel_size, padding=1)

        # up_5
        self.up5_upsampling1 = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.up5_maxunpool1 = nn.MaxUnpool2d(2, stride=2)
        self.up5_conv1 = nn.Conv2d(64 * 2, 64, kernel_size=kernel_size, padding=1)
        self.up5_conv2 = nn.Conv2d(64, 1, kernel_size=1, padding=0)  # en son 1*1 conv. padding=0 olacak !


    def forward(self, x):
        # down_1
        x_down1 = F.relu(self.down1_batch1(self.down1_conv1(x)))
        x_down1 = F.relu(self.down1_batch2(self.down1_conv2(x_down1)))
        x_down1 = self.down1_conv3(x_down1)  # (256,256)
        # print("x_down1:",x_down1.shape)
        x_down1_maxpool1, x_down1_maxpool1_indexs = self.down1_maxpool1(x_down1)  # (128,128)
        # print("x_down1_maxpool1:",x_down1_maxpool1.shape)

        # down_2
        x_down2 = F.relu(self.down2_batch1(self.down2_conv1(x_down1_maxpool1)))
        x_down2 = F.relu(self.down2_batch2(self.down2_conv2(x_down2)))
        x_down2 = self.down2_conv3(x_down2)  # (128,128)
        # print("x_down_2:",x_down2.shape)
        x_down2_maxpool1, x_down2_maxpool1_indexs = self.down2_maxpool1(x_down2)  # (64,64)
        # print("x_down2_maxpool1:",x_down2_maxpool1.shape)

        # down_3
        x_down3 = F.relu(self.down3_batch1(self.down3_conv1(x_down2_maxpool1)))
        x_down3 = F.relu(self.down3_batch2(self.down3_conv2(x_down3)))
        x_down3 = F.relu(self.down3_batch3(self.down3_conv3(x_down3)))
        x_down3 = self.down3_conv4(x_down3)  # (64,64)
        # print("x_down3:",x_down3.shape)
        x_down3_maxpool1, x_down3_maxpool1_indexs = self.down3_maxpool1(x_down3)  # (32,32)
        # print("x_down3_maxpool1:",x_down3_maxpool1.shape)

        # down_4
        x_down4 = F.relu(self.down4_batch1(self.down4_conv1(x_down3_maxpool1)))
        x_down4 = F.relu(self.down4_batch2(self.down4_conv2(x_down4)))
        x_down4 = F.relu(self.down4_batch3(self.down4_conv3(x_down4)))
        x_down4 = self.down4_conv4(x_down4)  # (32,32)
        # print("x_down4:",x_down4.shape)
        x_down4_maxpool1, x_down4_maxpool1_indexs = self.down4_maxpool1(x_down4)  # (16,16)
        # print("x_down4_maxpool1:",x_down4_maxpool1.shape)

        # down_5
        x_down5 = F.relu(self.down5_batch1(self.down5_conv1(x_down4_maxpool1)))
        x_down5 = F.relu(self.down5_batch2(self.down5_conv2(x_down5)))
        x_down5 = F.relu(self.down5_batch3(self.down5_conv3(x_down5)))
        x_down5 = self.down5_conv4(x_down5)  # (16,16)
        # print("x_down5:",x_down5.shape)
        x_down5_maxpool1, x_down5_maxpool1_indexs = self.down5_maxpool1(x_down5)  # (8,8)
        # print("x_down5_maxpool1",x_down5_maxpool1.shape)

        # up_1
        x_upsampling = self.up1_upsampling1(x_down5_maxpool1)  # (16,16)
        # print("x_up1:",x_up1.shape)
        x_unpool = self.up1_maxunpool1(x_down5_maxpool1, x_down5_maxpool1_indexs)  # (16,16)
        # print("x_up1:",x_up1.shape)
        x_up1 = torch.cat((x_upsampling, x_unpool), dim=1)  # channel sayılarından birleştir, (batch,2048,16,16)
        # print("x_up1:",x_up1.shape)
        x_up1 = self.up1_conv1(x_up1)
        x_up1 = self.up1_conv2(x_up1)
        x_up1 = self.up1_conv3(x_up1)

        # up_2
        x_upsampling = self.up2_upsampling1(x_up1)
        x_unpool = self.up2_maxunpool1(x_up1, x_down4_maxpool1_indexs)
        x_up2 = torch.cat((x_upsampling, x_unpool), dim=1)
        x_up2 = self.up2_conv1(x_up2)
        x_up2 = self.up2_conv2(x_up2)
        x_up2 = self.up2_conv3(x_up2)

        # up_3
        x_upsampling = self.up3_upsampling1(x_up2)
        x_unpool = self.up3_maxunpool1(x_up2, x_down3_maxpool1_indexs)
        x_up3 = torch.cat((x_upsampling, x_unpool), dim=1)
        x_up3 = self.up3_conv1(x_up3)
        x_up3 = self.up3_conv2(x_up3)
        x_up3 = self.up3_conv3(x_up3)

        # up_4
        x_upsampling = self.up4_upsampling1(x_up3)
        x_unpool = self.up4_maxunpool1(x_up3, x_down2_maxpool1_indexs)
        x_up4 = torch.cat((x_upsampling, x_unpool), dim=1)
        x_up4 = self.up4_conv1(x_up4)
        x_up4 = self.up4_conv2(x_up4)

        # up_5
        x_upsampling = self.up5_upsampling1(x_up4)
        x_unpool = self.up5_maxunpool1(x_up4, x_down1_maxpool1_indexs)
        x_up5 = torch.cat((x_upsampling, x_unpool), dim=1)
        x_up5 = self.up5_conv1(x_up5)
        x_up5 = self.up5_conv2(x_up5)  # sigmoidden geçir

        return x_up5


