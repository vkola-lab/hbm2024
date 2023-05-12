# coding=utf-8
import sys
sys.path.append('..')
from feature_extractor.for_image_data.backbone import CNN_GAP, ResNet3D, UNet3D
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

vgg_dict = {"vgg11": models.vgg11, "vgg13": models.vgg13, "vgg16": models.vgg16, "vgg19": models.vgg19,
            "vgg11bn": models.vgg11_bn, "vgg13bn": models.vgg13_bn, "vgg16bn": models.vgg16_bn, "vgg19bn": models.vgg19_bn}

# class AttentionModule(nn.Module):
#     def __init__(self, in_channels, out_channels, drop_rate=0.1):
#         super(AttentionModule, self).__init__()
#         self.conv = nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=False)
#         self.attention = ConvLayer(in_channels, out_channels, drop_rate, (1, 1, 0), (1, 1, 0))

#     def forward(self, x, return_attention=True):
#         feats = self.conv(x)
#         att = F.softmax(self.attention(x))

#         out = feats * att

#         if return_attention:
#             return att, out
        
#         return out


class VGGBase(nn.Module):
    def __init__(self, vgg_name):
        super(VGGBase, self).__init__()
        model_vgg = vgg_dict[vgg_name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_vgg.classifier[i])
        self.in_features = model_vgg.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class UNet3DBase(nn.Module):
    def __init__(self, n_class=1, act='relu', attention=False, pretrained=True, drop_rate=0.1, blocks=4):
        super(UNet3DBase, self).__init__()
        model = UNet3D(n_class=n_class, attention=attention, pretrained=pretrained, blocks=blocks)

        self.blocks = blocks

        self.down_tr64 = model.down_tr64
        self.down_tr128 = model.down_tr128
        self.down_tr256 = model.down_tr256
        self.down_tr512 = model.down_tr512
        if self.blocks == 5:
            self.down_tr1024 = model.down_tr1024
        # self.block_modules = nn.ModuleList([self.down_tr64, self.down_tr128, self.down_tr256, self.down_tr512])

        self.in_features = model.in_features
        ic(attention)
        if attention:
            self.attention_module = model.attention_module
        #     self.attention_module = AttentionModule(512, n_class, drop_rate=drop_rate)
        # self.avgpool = nn.AvgPool3d((6,7,6), stride=(6,6,6))

    def forward(self, x, stage='normal', attention=False):
        # ic('UNet3DBase forward')
        self.out64, self.skip_out64 = self.down_tr64(x)
        # ic(self.out64.shape, self.skip_out64.shape)
        self.out128,self.skip_out128 = self.down_tr128(self.out64)
        # ic(self.out128.shape, self.skip_out128.shape)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        # ic(self.out256.shape, self.skip_out256.shape)
        self.out512,self.skip_out512 = self.down_tr512(self.out256)
        # ic(self.out512.shape, self.skip_out512.shape)
        if self.blocks == 5:
            self.out1024,self.skip_out1024 = self.down_tr1024(self.out512)
        # ic(self.out1024.shape, self.skip_out1024.shape)
        # ic(hasattr(self, 'attention_module'))
        if hasattr(self, 'attention_module'):
            att, feats = self.attention_module(self.out1024 if self.blocks == 5 else self.out512)
        else:
            feats = self.out1024 if self.blocks == 5 else self.out512
        if attention:
            return att, feats
        return feats

        # self.out_up_256 = self.up_tr256(self.out512,self.skip_out256)
        # self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        # self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
        # self.out = self.out_tr(self.out_up_64)

        # return self.out

res_dict = {"resnet18": models.resnet18, "resnet34": models.resnet34, "resnet50": models.resnet50,
            "resnet101": models.resnet101, "resnet152": models.resnet152, "resnext50": models.resnext50_32x4d, "resnext101": models.resnext101_32x8d}


class ResBase(nn.Module):
    def __init__(self, res_name, num_classes=2, attention=False):
        super(ResBase, self).__init__()
        self.res_name, self.depth = res_name.split('_')
        if self.res_name == 'resnet3d':
            model_resnet = ResNet3D(int(self.depth), num_classes=num_classes, attention=attention)
            self.conv1 = model_resnet.conv1
            self.conv2 = model_resnet.conv2
            self.conv3 = model_resnet.conv3
            self.maxpool = model_resnet.maxpool
            self.block_modules = model_resnet.block_modules
            if attention:
                self.attention_module = model_resnet.attention_module
        else:
            model_resnet = res_dict[res_name](pretrained=True)
            self.conv1 = model_resnet.conv1
            self.bn1 = model_resnet.bn1
            self.relu = model_resnet.relu
            self.maxpool = model_resnet.maxpool
            self.layer1 = model_resnet.layer1
            self.layer2 = model_resnet.layer2
            self.layer3 = model_resnet.layer3
            self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        
        dummy_inp = torch.rand((1,1,182,218,182))
        dummy_feats = self.forward(dummy_inp, stage='get_features')
        if attention:
            dummy_feats = dummy_feats[0]
        self.in_features = list(dummy_feats.shape)

        # if attention:
        #     self.in_features = tuple(model_resnet.last_conv.weight.shape[2:])
        # else:
        #     self.in_features = model_resnet.fc.in_features
        print('in_features: ', self.in_features)

    def forward(self, x, stage='normal'):
        if self.res_name == 'resnet3d':
            x = self.conv1(x)
            print('conv1 out: ', x.size())
            x = self.conv2(x)
            print('conv2 out: ', x.size())
            x = self.conv3(x)
            print('conv3 out: ', x.size())
            x = self.maxpool(x)
            print('maxpool out: ', x.size())
            for i in range(len(self.block_modules)):
                x = self.block_modules[i](x)
                ic(i, x.size())
            print('Block modules out: ', x.size())
            if hasattr(self, 'attention_module'):
                att, feats = self.attention_module(x)
            else:
                feats = x
            
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
        
        if stage == 'get_features':
            return feats if not hasattr(self,'attention_module') else (att, feats)
        return feats
        

class CNNBase(nn.Module):
    def __init__(self, cnn_name, num_classes=2, attention=False):
        super(CNNBase, self).__init__()
        self.cnn_name = cnn_name
        self.instance_norm = False
        model = CNN_GAP(fil_num=20, drop_rate=0.137, num_classes=num_classes, instance_norm=self.instance_norm, attention=attention)
        if self.instance_norm:
            self.norm = model.norm
        self.block_modules = model.block_modules
        # self.block1 = model.block1
        # self.block2 = model.block2
        # self.block3 = model.block3
        # self.block4 = model.block4
        self.attention = attention
        if attention:
            self.attention_module = model.attention_module

        self.in_features = model.in_features
        print('in features: ', self.in_features)

    def forward(self, x, stage='normal'):
        ic('-----CNNBase forward------')
        if self.instance_norm:
            x = self.norm(x)
            ic('norm out: ', x.size())
        # x = self.conv1(x)
        # # print('conv1 out: ', x.size())
        # x = self.conv2(x)
        # # print('conv2 out: ', x.size())
        # x = self.conv3(x)
        # # print('conv3 out: ', x.size())
        # x = self.maxpool(x)
        # print('maxpool out: ', x.size())
        x = self.block_modules(x)
        print('Block modules out: ', x.size())
        if self.attention:
            att, feats = self.attention_module(x)
            ic('feats: ', feats.size())
        else:
            feats = x
        
        if stage == 'get_features':
            return feats if not hasattr(self,'attention_module') else (att, feats)
        return feats

class DTNBase(nn.Module):
    def __init__(self):
        super(DTNBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )
        self.in_features = 256*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.in_features = 50*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x
