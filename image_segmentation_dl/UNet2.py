import torch

"""
reference paper: Ronneberger, O., Fischer, P., & Brox, T. (2015, October). U-net: Convolutional networks for biomedical image segmentation. In International Conference on Medical
image computing and computer-assisted intervention (pp. 234-241). Springer, Cham.

author: gürkan şahin, rev.: 26/09/2018
"""

#6 down-6 up layer

class UNet1(torch.nn.Module):
    def __init__(self):
        super(UNet1, self).__init__()

        #################### DOWN #########################################
        # down_block1
        self.down_block1_conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)  # padding yapılınca n*m image boyutu azalmadan aynen kalıyor !
        self.down_block1_bn1 = torch.nn.BatchNorm2d(16)
        self.down_block1_conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.down_block1_bn2 = torch.nn.BatchNorm2d(16)
        self.down_block1_conv3 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.down_block1_bn3 = torch.nn.BatchNorm2d(16)
        self.down_block1_max_pool = torch.nn.MaxPool2d(2, 2)
        self.down_block1_relu = torch.nn.ReLU()

        # down_block2
        self.down_block2_conv1 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.down_block2_bn1 = torch.nn.BatchNorm2d(32)
        self.down_block2_conv2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.down_block2_bn2 = torch.nn.BatchNorm2d(32)
        self.down_block2_conv3 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.down_block2_bn3 = torch.nn.BatchNorm2d(32)
        self.down_block2_max_pool = torch.nn.MaxPool2d(2, 2)
        self.down_block2_relu = torch.nn.ReLU()

        # down_block3
        self.down_block3_conv1 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.down_block3_bn1 = torch.nn.BatchNorm2d(64)
        self.down_block3_conv2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.down_block3_bn2 = torch.nn.BatchNorm2d(64)
        self.down_block3_conv3 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.down_block3_bn3 = torch.nn.BatchNorm2d(64)
        self.down_block3_max_pool = torch.nn.MaxPool2d(2, 2)
        self.down_block3_relu = torch.nn.ReLU()

        # down_block4
        self.down_block4_conv1 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.down_block4_bn1 = torch.nn.BatchNorm2d(128)
        self.down_block4_conv2 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.down_block4_bn2 = torch.nn.BatchNorm2d(128)
        self.down_block4_conv3 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.down_block4_bn3 = torch.nn.BatchNorm2d(128)
        self.down_block4_max_pool = torch.nn.MaxPool2d(2, 2)
        self.down_block4_relu = torch.nn.ReLU()

        # down_block5
        self.down_block5_conv1 = torch.nn.Conv2d(128, 256, 3, padding=1)
        self.down_block5_bn1 = torch.nn.BatchNorm2d(256)
        self.down_block5_conv2 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.down_block5_bn2 = torch.nn.BatchNorm2d(256)
        self.down_block5_conv3 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.down_block5_bn3 = torch.nn.BatchNorm2d(256)
        self.down_block5_max_pool = torch.nn.MaxPool2d(2, 2)
        self.down_block5_relu = torch.nn.ReLU()

        # down_block6
        self.down_block6_conv1 = torch.nn.Conv2d(256, 512, 3, padding=1)
        self.down_block6_bn1 = torch.nn.BatchNorm2d(512)
        self.down_block6_conv2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.down_block6_bn2 = torch.nn.BatchNorm2d(512)
        self.down_block6_conv3 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.down_block6_bn3 = torch.nn.BatchNorm2d(512)
        self.down_block6_max_pool = torch.nn.MaxPool2d(2, 2)
        self.down_block6_relu = torch.nn.ReLU()
        #################### DOWN #########################################


        #################### MIDDLE BLOCK #########################################
        self.middle_block_conv1 = torch.nn.Conv2d(512, 1024, 3, padding=1)
        self.middle_block_bn1 = torch.nn.BatchNorm2d(1024)
        self.middle_block_conv2 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.middle_block_bn2 = torch.nn.BatchNorm2d(1024)
        self.middle_block_conv3 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.middle_block_bn3 = torch.nn.BatchNorm2d(1024)
        self.middle_block_relu = torch.nn.ReLU()
        #################### MIDDLE BLOCK #########################################


        #################### UP #########################################
        # up_block1
        self.up_block1_up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_block1_conv1 = torch.nn.Conv2d(512 + 1024, 512, 3, padding=1)  # (prev_channel + input_channel, output_channel)
        self.up_block1_bn1 = torch.nn.BatchNorm2d(512)
        self.up_block1_conv2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.up_block1_bn2 = torch.nn.BatchNorm2d(512)
        self.up_block1_conv3 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.up_block1_bn3 = torch.nn.BatchNorm2d(512)
        self.up_block1_relu = torch.nn.ReLU()

        # up_block2
        self.up_block2_up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_block2_conv1 = torch.nn.Conv2d(256 + 512, 256, 3, padding=1)
        self.up_block2_bn1 = torch.nn.BatchNorm2d(256)
        self.up_block2_conv2 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.up_block2_bn2 = torch.nn.BatchNorm2d(256)
        self.up_block2_conv3 = torch.nn.Conv2d(256, 256, 3, padding=1)
        self.up_block2_bn3 = torch.nn.BatchNorm2d(256)
        self.up_block2_relu = torch.nn.ReLU()

        # up_block3
        self.up_block3_up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_block3_conv1 = torch.nn.Conv2d(128 + 256, 128, 3, padding=1)
        self.up_block3_bn1 = torch.nn.BatchNorm2d(128)
        self.up_block3_conv2 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.up_block3_bn2 = torch.nn.BatchNorm2d(128)
        self.up_block3_conv3 = torch.nn.Conv2d(128, 128, 3, padding=1)
        self.up_block3_bn3 = torch.nn.BatchNorm2d(128)
        self.up_block3_relu = torch.nn.ReLU()

        # up_block4
        self.up_block4_up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_block4_conv1 = torch.nn.Conv2d(64 + 128, 64, 3, padding=1)
        self.up_block4_bn1 = torch.nn.BatchNorm2d(64)
        self.up_block4_conv2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.up_block4_bn2 = torch.nn.BatchNorm2d(64)
        self.up_block4_conv3 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.up_block4_bn3 = torch.nn.BatchNorm2d(64)
        self.up_block4_relu = torch.nn.ReLU()

        # up_block5
        self.up_block5_up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_block5_conv1 = torch.nn.Conv2d(32 + 64, 32, 3, padding=1)
        self.up_block5_bn1 = torch.nn.BatchNorm2d(32)
        self.up_block5_conv2 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.up_block5_bn2 = torch.nn.BatchNorm2d(32)
        self.up_block5_conv3 = torch.nn.Conv2d(32, 32, 3, padding=1)
        self.up_block5_bn3 = torch.nn.BatchNorm2d(32)
        self.up_block5_relu = torch.nn.ReLU()

        # up_block6
        self.up_block6_up_sampling = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_block6_conv1 = torch.nn.Conv2d(16 + 32, 16, 3, padding=1)
        self.up_block6_bn1 = torch.nn.BatchNorm2d(16)
        self.up_block6_conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.up_block6_bn2 = torch.nn.BatchNorm2d(16)
        self.up_block6_conv3 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.up_block6_bn3 = torch.nn.BatchNorm2d(16)
        self.up_block6_relu = torch.nn.ReLU()

        #################### UP #########################################


        #################### LAST CONV. #########################################
        self.last_conv1 = torch.nn.Conv2d(16, 16, 3, padding=1)  # 16,16
        self.last_bn1 = torch.nn.BatchNorm2d(16)  # 16
        self.last_conv2 = torch.nn.Conv2d(16, 1, 1, padding=0)  # en son conv, 16 channel (filter)'ı 1 channel a indiriyor
        self.last_relu = torch.nn.ReLU()

    #################### LAST CONV. #########################################


    def forward(self, x):
        # down_block1
        x1 = self.down_block1_relu(self.down_block1_bn1(self.down_block1_conv1(x)))
        # print("x1:", x1.shape)
        x1 = self.down_block1_relu(self.down_block1_bn2(self.down_block1_conv2(x1)))
        # print("x1:", x1.shape)
        x1 = self.down_block1_relu(self.down_block1_bn3(self.down_block1_conv3(x1)))
        # print("x1:", x1.shape)
        self.x1 = x1  # up kısmında copy-and-crop işlemi için kullanılacak!
        x1 = self.down_block1_max_pool(x1)
        # print("x1:", x1.shape, self.x1.shape)

        # down_block2
        x2 = self.down_block2_relu(self.down_block2_bn1(self.down_block2_conv1(x1)))
        x2 = self.down_block2_relu(self.down_block2_bn2(self.down_block2_conv2(x2)))
        x2 = self.down_block2_relu(self.down_block2_bn3(self.down_block2_conv3(x2)))
        self.x2 = x2
        x2 = self.down_block2_max_pool(x2)
        # print("x2:", x2.shape, self.x2.shape)

        # down_block3
        x3 = self.down_block3_relu(self.down_block3_bn1(self.down_block3_conv1(x2)))
        x3 = self.down_block3_relu(self.down_block3_bn2(self.down_block3_conv2(x3)))
        x3 = self.down_block3_relu(self.down_block3_bn3(self.down_block3_conv3(x3)))
        self.x3 = x3
        x3 = self.down_block3_max_pool(x3)
        # print("x3:", x3.shape, self.x3.shape)

        # down_block4
        x4 = self.down_block4_relu(self.down_block4_bn1(self.down_block4_conv1(x3)))
        x4 = self.down_block4_relu(self.down_block4_bn2(self.down_block4_conv2(x4)))
        x4 = self.down_block4_relu(self.down_block4_bn3(self.down_block4_conv3(x4)))
        self.x4 = x4
        x4 = self.down_block4_max_pool(x4)
        # print("x4:", x4.shape, self.x4.shape)

        # down_block5
        x5 = self.down_block5_relu(self.down_block5_bn1(self.down_block5_conv1(x4)))
        x5 = self.down_block5_relu(self.down_block5_bn2(self.down_block5_conv2(x5)))
        x5 = self.down_block5_relu(self.down_block5_bn3(self.down_block5_conv3(x5)))
        self.x5 = x5
        x5 = self.down_block5_max_pool(x5)
        # print("x5:", x5.shape, self.x5.shape)

        # down_block6
        x6 = self.down_block6_relu(self.down_block6_bn1(self.down_block6_conv1(x5)))
        x6 = self.down_block6_relu(self.down_block6_bn2(self.down_block6_conv2(x6)))
        x6 = self.down_block6_relu(self.down_block6_bn3(self.down_block6_conv3(x6)))
        self.x6 = x6
        x6 = self.down_block6_max_pool(x6)
        # print("x6:", x6.shape, self.x6.shape)

        # middle_block
        x7 = self.middle_block_relu(self.middle_block_bn1(self.middle_block_conv1(x6)))
        x7 = self.middle_block_relu(self.middle_block_bn2(self.middle_block_conv2(x7)))
        x7 = self.middle_block_relu(self.middle_block_bn3(self.middle_block_conv3(x7)))
        # print("middle:", x7.shape)

        # up_block1
        x = self.up_block1_up_sampling(x7)
        # print("up1:", x.shape)
        x = torch.cat((x, self.x6), dim=1)  # .cat(x,prev_feature_map=x6)
        # print("cat:",x.shape)
        x = self.up_block1_relu(self.up_block1_bn1(self.up_block1_conv1(x)))
        x = self.up_block1_relu(self.up_block1_bn2(self.up_block1_conv2(x)))
        x = self.up_block1_relu(self.up_block1_bn3(self.up_block1_conv3(x)))
        # print("up1:", x.shape)

        # up_block2
        x = self.up_block2_up_sampling(x)
        x = torch.cat((x, self.x5), dim=1)
        x = self.up_block2_relu(self.up_block2_bn1(self.up_block2_conv1(x)))
        x = self.up_block2_relu(self.up_block2_bn2(self.up_block2_conv2(x)))
        x = self.up_block2_relu(self.up_block2_bn3(self.up_block2_conv3(x)))
        # print("up2:", x.shape)

        # up_block3
        x = self.up_block3_up_sampling(x)
        x = torch.cat((x, self.x4), dim=1)
        x = self.up_block3_relu(self.up_block3_bn1(self.up_block3_conv1(x)))
        x = self.up_block3_relu(self.up_block3_bn2(self.up_block3_conv2(x)))
        x = self.up_block3_relu(self.up_block3_bn3(self.up_block3_conv3(x)))
        # print("up3:", x.shape)

        # up_block4
        x = self.up_block4_up_sampling(x)
        x = torch.cat((x, self.x3), dim=1)
        x = self.up_block4_relu(self.up_block4_bn1(self.up_block4_conv1(x)))
        x = self.up_block4_relu(self.up_block4_bn2(self.up_block4_conv2(x)))
        x = self.up_block4_relu(self.up_block4_bn3(self.up_block4_conv3(x)))
        # print("up4:", x.shape)

        # up_block5
        x = self.up_block5_up_sampling(x)
        x = torch.cat((x, self.x2), dim=1)
        x = self.up_block5_relu(self.up_block5_bn1(self.up_block5_conv1(x)))
        x = self.up_block5_relu(self.up_block5_bn2(self.up_block5_conv2(x)))
        x = self.up_block5_relu(self.up_block5_bn3(self.up_block5_conv3(x)))
        # print("up5:", x.shape)

        # up_block6
        x = self.up_block6_up_sampling(x)
        x = torch.cat((x, self.x1), dim=1)
        x = self.up_block6_relu(self.up_block6_bn1(self.up_block6_conv1(x)))
        x = self.up_block6_relu(self.up_block6_bn2(self.up_block6_conv2(x)))
        x = self.up_block6_relu(self.up_block6_bn3(self.up_block6_conv3(x)))
        # print("up6:", x.shape)

        # last conv.
        x = self.last_relu(self.last_bn1(self.last_conv1(x)))
        x = self.last_conv2(x)  # en son  conv çıkışı (x) üzerinde sigmoid kullan ! (main.py eklendi)
        # print("last:", x.shape)
        return x

