# coding=utf-8
import math
import torch
import torch.nn as nn
from network.util import init_weights
import torch.nn.utils.weight_norm as weightNorm


class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        # self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        # if type in ['conv', 'gap'] and len(bottleneck_dim) > 3:
            # bottleneck_dim = bottleneck_dim[-3:]
        ic(bottleneck_dim)
        if type == 'wn':
            self.layer = weightNorm(
                nn.Linear(bottleneck_dim[1:], class_num), name="weight")
            # self.fc.apply(init_weights)
        elif type == 'gap':
            if len(bottleneck_dim) > 3:
                bottleneck_dim = bottleneck_dim[-3:]
            self.layer = nn.AvgPool3d(bottleneck_dim, stride=(1,1,1))
        elif type == 'conv':
            if len(bottleneck_dim) > 3:
                bottleneck_dim = bottleneck_dim[-4:]
            ic(bottleneck_dim)
            self.layer = nn.Conv3d(bottleneck_dim[0], class_num, kernel_size=bottleneck_dim[1:])
            ic(self.layer)
        else:
            print('bottleneck dim: ', bottleneck_dim)
            self.layer = nn.Sequential(
                            torch.nn.Flatten(start_dim=1, end_dim=-1),
                            nn.Linear(math.prod(bottleneck_dim), class_num)
            )
        self.layer.apply(init_weights)

    def forward(self, x):
        print('=> feat_classifier forward')
        ic(x.size())
        x = self.layer(x)
        ic(x.size())
        if self.type in ['gap','conv']:
            x = torch.squeeze(x)
            if len(x.shape) < 2:
                x = torch.unsqueeze(x,0)
        print('returning x: ', x.size())
        return x
        

class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        # self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        # self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x

class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv3d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 64, 5, padding=padding),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)
    

class AugNet3D(nn.Module):
    def __init__(self, noise_lv):
        super(AugNet3D, self).__init__()
        ############# Trainable Parameters
        self.noise_lv = nn.Parameter(torch.zeros(1))
        self.shift_var = nn.Parameter(torch.empty(1,174,210,174))
        nn.init.normal_(self.shift_var, 1, 0.1)
        self.shift_mean = nn.Parameter(torch.zeros(1,174,210,174))
        nn.init.normal_(self.shift_mean, 0, 0.1)

        self.shift_var2 = nn.Parameter(torch.empty(1,170,206,170))
        nn.init.normal_(self.shift_var2, 1, 0.1)
        self.shift_mean2 = nn.Parameter(torch.zeros(1,170,206,170))
        nn.init.normal_(self.shift_mean2, 0, 0.1)

        self.shift_var3 = nn.Parameter(torch.empty(1,166, 202, 166))
        nn.init.normal_(self.shift_var3, 1, 0.1)
        self.shift_mean3 = nn.Parameter(torch.zeros(1,166, 202, 166))
        nn.init.normal_(self.shift_mean3, 0, 0.1)

        self.shift_var4 = nn.Parameter(torch.empty(1,178, 214, 178))
        nn.init.normal_(self.shift_var4, 1, 0.1)
        self.shift_mean4 = nn.Parameter(torch.zeros(1,178, 214, 178))
        nn.init.normal_(self.shift_mean4, 0, 0.1)

        # self.shift_var5 = nn.Parameter(torch.empty(3, 206, 206))
        # nn.init.normal_(self.shift_var5, 1, 0.1)
        # self.shift_mean5 = nn.Parameter(torch.zeros(3, 206, 206))
        # nn.init.normal_(self.shift_mean5, 0, 0.1)
        #
        # self.shift_var6 = nn.Parameter(torch.empty(3, 204, 204))
        # nn.init.normal_(self.shift_var6, 1, 0.5)
        # self.shift_mean6 = nn.Parameter(torch.zeros(3, 204, 204))
        # nn.init.normal_(self.shift_mean6, 0, 0.1)

        # self.shift_var7 = nn.Parameter(torch.empty(3, 202, 202))
        # nn.init.normal_(self.shift_var7, 1, 0.5)
        # self.shift_mean7 = nn.Parameter(torch.zeros(3, 202, 202))
        # nn.init.normal_(self.shift_mean7, 0, 0.1)

        self.norm = nn.InstanceNorm3d(1)

        ############## Fixed Parameters (For MI estimation
        self.spatial = nn.Conv3d(1, 1, 9).cuda()
        self.spatial_up = nn.ConvTranspose3d(1, 1, 9).cuda()

        self.spatial2 = nn.Conv3d(1, 1, 13).cuda()
        self.spatial_up2 = nn.ConvTranspose3d(1, 1, 13).cuda()

        self.spatial3 = nn.Conv3d(1, 1, 17).cuda()
        self.spatial_up3 = nn.ConvTranspose3d(1, 1, 17).cuda()

        self.spatial4 = nn.Conv3d(1, 1, 5).cuda()
        self.spatial_up4 = nn.ConvTranspose3d(1, 1, 5).cuda()


        self.color = nn.Conv3d(1, 1, 1).cuda()

        for param in list(list(self.color.parameters()) +
                          list(self.spatial.parameters()) + list(self.spatial_up.parameters()) +
                          list(self.spatial2.parameters()) + list(self.spatial_up2.parameters()) +
                          list(self.spatial3.parameters()) + list(self.spatial_up3.parameters()) +
                          list(self.spatial4.parameters()) + list(self.spatial_up4.parameters())
                          ):
            param.requires_grad=False

    def forward(self, x, estimation=False):
        print('----------------Augnet3D forward----------------')
        print('Estimation: ', estimation)
        if not estimation:
            spatial = nn.Conv3d(1, 1, 9).cuda()
            spatial_up = nn.ConvTranspose3d(1, 1, 9).cuda()

            print('spatial: ', spatial.weight.size())
            print('spatial_up: ', spatial_up.weight.size())

            spatial2 = nn.Conv3d(1, 1, 13).cuda()
            spatial_up2 = nn.ConvTranspose3d(1, 1, 13).cuda()

            spatial3 = nn.Conv3d(1, 1, 17).cuda()
            spatial_up3 = nn.ConvTranspose3d(1, 1, 17).cuda()

            spatial4 = nn.Conv3d(1, 1, 5).cuda()
            spatial_up4 = nn.ConvTranspose3d(1, 1, 5).cuda()


            color = nn.Conv3d(1, 1, 1).cuda()
            weight = torch.randn(5)

            x = x + torch.randn_like(x) * self.noise_lv * 0.01
            print('x: ', x.size())
            x_c = torch.sigmoid(F.dropout(color(x), p=.2))
            print('x_c: ', x_c.size())

            x_sdown = spatial(x)
            print('x_sdown: ', x_sdown.size())
            print('x_sdown norm: ', self.norm(x_sdown).size())
            print('shift var: ', self.shift_var.size(), ', shift mean: ', self.shift_mean.size())
            x_sdown = self.shift_var * self.norm(x_sdown) + self.shift_mean
            x_s = torch.sigmoid(spatial_up(x_sdown))
            print('x_s: ', x_s.size())
            #
            x_s2down = spatial2(x)
            print('x_s2down: ', x_s2down.size())
            x_s2down = self.shift_var2 * self.norm(x_s2down) + self.shift_mean2
            x_s2 = torch.sigmoid(spatial_up2(x_s2down))
            print('x_s2: ', x_s2.size())

            #
            #
            x_s3down = spatial3(x)
            print('x_s3down: ', x_s3down.size())
            x_s3down = self.shift_var3 * self.norm(x_s3down) + self.shift_mean3
            x_s3 = torch.sigmoid(spatial_up3(x_s3down))
            print('x_s3: ', x_s3.size())

            #
            x_s4down = spatial4(x)
            print('x_s4down: ', x_s4down.size())
            x_s4down = self.shift_var4 * self.norm(x_s4down) + self.shift_mean4
            x_s4 = torch.sigmoid(spatial_up4(x_s4down))
            print('x_s4: ', x_s4.size())


            output = (weight[0] * x_c + weight[1] * x_s + weight[2] * x_s2+ weight[3] * x_s3 + weight[4]*x_s4) / weight.sum()
            print('output: ', output.size())
        else:
            x = x + torch.randn_like(x) * self.noise_lv * 0.01
            x_c = torch.sigmoid(self.color(x))
            #
            x_sdown = self.spatial(x)
            x_sdown = self.shift_var * self.norm(x_sdown) + self.shift_mean
            x_s = torch.sigmoid(self.spatial_up(x_sdown))
            #
            x_s2down = self.spatial2(x)
            x_s2down = self.shift_var2 * self.norm(x_s2down) + self.shift_mean2
            x_s2 = torch.sigmoid(self.spatial_up2(x_s2down))

            x_s3down = self.spatial3(x)
            x_s3down = self.shift_var3 * self.norm(x_s3down) + self.shift_mean3
            x_s3 = torch.sigmoid(self.spatial_up3(x_s3down))

            x_s4down = self.spatial4(x)
            x_s4down = self.shift_var4 * self.norm(x_s4down) + self.shift_mean4
            x_s4 = torch.sigmoid(self.spatial_up4(x_s4down))

            output = (x_c + x_s + x_s2 + x_s3 + x_s4) / 5
        return output
