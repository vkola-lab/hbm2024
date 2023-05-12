import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.nn import init
import torch.nn.functional as F
# convnet without the last layer


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act)
        layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act)
    else:
        layer1 = LUConv(in_channel, 32*(2**depth),act)
        layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act)

    return nn.Sequential(layer1,layer2)


# class InputTransition(nn.Module):
#     def __init__(self, outChans, elu):
#         super(InputTransition, self).__init__()
#         self.conv1 = nn.Conv3d(1, 16, kernel_size=5, padding=2)
#         self.bn1 = ContBatchNorm3d(16)
#         self.relu1 = ELUCons(elu, 16)
#
#     def forward(self, x):
#         # do we want a PRELU here as well?
#         out = self.bn1(self.conv1(x))
#         # split input in to 16 channels
#         x16 = torch.cat((x, x, x, x, x, x, x, x,
#                          x, x, x, x, x, x, x, x), 1)
#         out = self.relu1(torch.add(out, x16))
#         return out

class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out

class UNet3D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=1, act='relu', pretrained=False, input_size=(1,1,182,218,182), attention=False, drop_rate=0.1, blocks=4):
        super(UNet3D, self).__init__()

        self.blocks = blocks
        self.down_tr64 = DownTransition(1,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.down_tr256 = DownTransition(128,2,act)
        self.down_tr512 = DownTransition(256,3,act)

        self.up_tr256 = UpTransition(512, 512,2,act)
        self.up_tr128 = UpTransition(256,256, 1,act)
        self.up_tr64 = UpTransition(128,128,0,act)
        self.out_tr = OutputTransition(64, 1)

        self.pretrained = pretrained
        self.attention = attention
        if pretrained:
            weight_dir = 'pretrained_ckpts/Genesis_Chest_CT.pt'
            checkpoint = torch.load(weight_dir)
            state_dict = checkpoint['state_dict']
            unParalled_state_dict = {}
            for key in state_dict.keys():
                unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            self.load_state_dict(unParalled_state_dict)
            del self.up_tr256
            del self.up_tr128
            del self.up_tr64
            del self.out_tr
        
        if self.blocks == 5:
            self.down_tr1024 = DownTransition(512,4,act)
            

        # self.conv1 = nn.Conv3d(512, 256, 1, 1, 0, bias=False)
        # self.conv2 = nn.Conv3d(256, 128, 1, 1, 0, bias=False)
        # self.conv3 = nn.Conv3d(128, 64, 1, 1, 0, bias=False)

        if attention:
            self.attention_module = AttentionModule(1024 if self.blocks==5 else 512, n_class, drop_rate=drop_rate)
        # Output.
        self.avgpool = nn.AvgPool3d((6,7,6), stride=(6,6,6))

        dummy_inp = torch.rand(input_size)
        dummy_feats = self.forward(dummy_inp, stage='get_features')
        dummy_feats = dummy_feats[0]
        self.in_features = list(dummy_feats.shape)
        ic(self.in_features)

        self._init_weights()

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    init.kaiming_normal_(m.weight)
                elif isinstance(m, ContBatchNorm3d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight)
                    init.constant_(m.bias, 0)
        elif self.attention:
            for m in self.attention_module.modules():
                if isinstance(m, nn.Conv3d):
                    init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm3d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
        else:
            pass
        # Zero initialize the last batchnorm in each residual branch.
        # for m in self.modules():
        #     if isinstance(m, BottleneckBlock):
        #         init.constant_(m.out_conv.bn.weight, 0)
    
    def forward(self, x, stage='normal', attention=False):
        ic('backbone forward')
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128,self.skip_out128 = self.down_tr128(self.out64)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        self.out512,self.skip_out512 = self.down_tr512(self.out256)
        if self.blocks == 5:
            self.out1024,self.skip_out1024 = self.down_tr1024(self.out512)
            ic(self.out1024.shape)
        # self.out = self.conv1(self.out512)
        # self.out = self.conv2(self.out)
        # self.out = self.conv3(self.out)
        # self.out = self.conv(self.out)
        ic(hasattr(self, 'attention_module'))
        if hasattr(self, 'attention_module'):
            att, feats = self.attention_module(self.out1024 if self.blocks==5 else self.out512)
        else:
            feats = self.out1024 if self.blocks==5 else self.out512
        ic(feats.shape)
        if attention:
            return att, feats
        return feats

class AlexNetFc(nn.Module):
    def __init__(self):
        super(AlexNetFc, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features


class ResNet18Fc(nn.Module):
    def __init__(self):
        super(ResNet18Fc, self).__init__()
        model_resnet18 = models.resnet18(pretrained=True)
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool
        self.__in_features = model_resnet18.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class ResNet34Fc(nn.Module):
    def __init__(self):
        super(ResNet34Fc, self).__init__()
        model_resnet34 = models.resnet34(pretrained=True)
        self.conv1 = model_resnet34.conv1
        self.bn1 = model_resnet34.bn1
        self.relu = model_resnet34.relu
        self.maxpool = model_resnet34.maxpool
        self.layer1 = model_resnet34.layer1
        self.layer2 = model_resnet34.layer2
        self.layer3 = model_resnet34.layer3
        self.layer4 = model_resnet34.layer4
        self.avgpool = model_resnet34.avgpool
        self.__in_features = model_resnet34.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class ResNet50Fc(nn.Module):
    def __init__(self):
        super(ResNet50Fc, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class ResNet101Fc(nn.Module):
    def __init__(self):
        super(ResNet101Fc, self).__init__()
        model_resnet101 = models.resnet101(pretrained=True)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool
        self.__in_features = model_resnet101.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class ResNet152Fc(nn.Module):
    def __init__(self):
        super(ResNet152Fc, self).__init__()
        model_resnet152 = models.resnet152(pretrained=True)
        self.conv1 = model_resnet152.conv1
        self.bn1 = model_resnet152.bn1
        self.relu = model_resnet152.relu
        self.maxpool = model_resnet152.maxpool
        self.layer1 = model_resnet152.layer1
        self.layer2 = model_resnet152.layer2
        self.layer3 = model_resnet152.layer3
        self.layer4 = model_resnet152.layer4
        self.avgpool = model_resnet152.avgpool
        self.__in_features = model_resnet152.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


############################################################################

class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, relu=True, relu_type='leaky'):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=False) if (relu and relu_type=='leaky') else nn.ReLU(inplace=False) if relu else None
    
    def forward(self, x):
        # print('--------ConvBNRelu forward--------')
        # print('input: ', x.size())
        x = self.conv(x)
        # print('conv out: ', x.size())
        out = self.bn(x)
        # print('BN out: ', out.size())
        if self.relu is not None:
            out = self.relu(out)
            # print('Relu out: ', out.size())
        # print('-----------------------------------')
        return out

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, downsample, width=1,
                 pool_residual=False):
        super().__init__()
        self.out_channels = 4 * mid_channels
        # Width factor applies only to inner 3x3 convolution.
        mid_channels = int(mid_channels * width)
    
        # Skip connection.
        if downsample:
            if pool_residual:
                pool = nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
                conv = ConvBNRelu(
                    in_channels, self.out_channels, kernel_size=1,
                    stride=1, padding=0, relu=False)
                self.skip_connection = nn.Sequential(pool, conv)
            else:
                self.skip_connection = ConvBNRelu(
                    in_channels, self.out_channels, kernel_size=1,
                    stride=2, padding=0, relu=False)
        elif in_channels != self.out_channels:
            self.skip_connection = ConvBNRelu(
                in_channels, self.out_channels, kernel_size=1,
                stride=1, padding=0, relu=False)
        else:
            self.skip_connection = None

        # Main branch.
        self.in_conv = ConvBNRelu(
            in_channels, mid_channels, kernel_size=1,
            stride=1, padding=0)
        self.mid_conv = ConvBNRelu(
            mid_channels, mid_channels, kernel_size=3,
            stride=(2 if downsample else 1), padding=1)
        self.out_conv = ConvBNRelu(
            mid_channels, self.out_channels, kernel_size=1,
            stride=1, padding=0, relu=False)
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.skip_connection is not None:
            residual = self.skip_connection(x)
        else:
            residual = x

        out = self.out_conv(self.mid_conv(self.in_conv(x)))
        out += residual
        return self.out_relu(out)


class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.1):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.attention = ConvLayer(in_channels, out_channels, drop_rate, (1, 1, 0), (1, 1, 0))

    def forward(self, x, return_attention=True):
        feats = self.conv(x)
        att = F.softmax(self.attention(x))

        out = feats * att

        if return_attention:
            return att, out
        
        return out



class ResNet(nn.Module):
    def __init__(self, block, module_sizes, module_channels, num_classes,
                 width=1, pool_residual=False, attention=False, drop_rate=0.1):
        super().__init__()

        # Input trunk, Inception-style.
        self.conv1 = ConvBNRelu(1, module_channels[0] // 2, kernel_size=3,
                                stride=2, padding=1)
        self.conv2 = ConvBNRelu(module_channels[0] // 2, module_channels[0] // 2,
                                kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBNRelu(module_channels[0] // 2, module_channels[0],
                                kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Build the main network.
        modules = []
        out_channels = module_channels[0]
        for module_idx, (num_layers, mid_channels) in enumerate(zip(module_sizes, module_channels)):
            blocks = []
            for i in range(num_layers):
                in_channels = out_channels
                downsample = i == 0 and module_idx > 0
                b = block(in_channels, mid_channels, downsample,
                            width=width, pool_residual=pool_residual)
                out_channels = b.out_channels
                blocks.append(b)
            modules.append(nn.Sequential(*blocks))
        self.block_modules = nn.Sequential(*modules)

        if attention:
            self.attention_module = AttentionModule(out_channels, num_classes, drop_rate=drop_rate)
        # Output.
        self.avgpool = nn.AvgPool3d((6,7,6), stride=(6,6,6))
        
        # self.fc = nn.Linear(out_channels, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                init.constant_(m.bias, 0)
        # Zero initialize the last batchnorm in each residual branch.
        for m in self.modules():
            if isinstance(m, BottleneckBlock):
                init.constant_(m.out_conv.bn.weight, 0)

    def forward(self, x):
        print('----------ResNet forward-------------')
        x = self.conv1(x)
        # print('conv1 out: ', x.size())
        x = self.conv2(x)
        # print('conv2 out: ', x.size())
        x = self.conv3(x)
        # print('conv3 out: ', x.size())
        x = self.maxpool(x)
        # print('maxpool out: ', x.size())
        x = self.block_modules(x)
        
        return x

def _ResNet(module_sizes, module_channels, num_classes=2, attention=False):
    return ResNet(BottleneckBlock, 
                    module_sizes, module_channels, num_classes, attention=attention)

def ResNet3D(depth=10, num_classes=2, attention=False):
    if depth == 10:
        net = _ResNet((1, 1, 1), (32, 64, 128), num_classes=num_classes, attention=attention)
    elif depth == 18:
        net = _ResNet((2, 2, 2, 2), (64, 128, 256, 512), num_classes=num_classes, attention=attention)
    elif depth == 50:
        net = _ResNet((3, 4, 6, 3), (64, 128, 256, 512), num_classes=num_classes, attention=attention)
    else:
        raise ValueError('Depth must be 10,18 or 50!!!')
    # path = '/projectnb/ivc-ml/dlteif/SHAP4Med/brain2020/checkpoint_dir/NACC_NC_MCI_AD/run3_ResNet_train_NACC_NC_MCI_AD_balanced_batch_size_4_accum_iter10_CE_loss_weighted/cnn_best.pth'
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ckpt = torch.load(path, map_location=device)
    # try:
    #     load_state_dict(net, ckpt)
    # except:
    #     load_state_dict(net, ckpt['state_dict'])
    #     best_epoch = ckpt['epoch']
    #     best_acc = ckpt['accuracy']
    #     print(f'epoch: {best_epoch}, best acc: {best_acc}')
        
    # print(f"loaded checkpoint at {path}")
    return net

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate, kernel, pooling, BN=True, relu_type='leaky'):
        super().__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=False) if relu_type=='leaky' else nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(drop_rate, inplace=False) 
       
    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class CNN_GAP(nn.Module):
    def __init__(self, fil_num, drop_rate, num_classes=2, depth=4, instance_norm=False, input_size=(1,1,182,218,182), attention=False):
        super(CNN_GAP, self).__init__()
        self.instance_norm = instance_norm
        self.depth = depth
        self.norm = torch.nn.InstanceNorm3d(1, affine=True)

        self.block_modules = []
        in_channels = 1
        out_channels = fil_num
        for i in range(self.depth):
            self.block_modules.append(ConvLayer(in_channels, out_channels, drop_rate, (3, 1, 0), (2, 2, 0)))
            in_channels = out_channels
            out_channels = 2 * out_channels

        # self.block1 = ConvLayer(1, fil_num, drop_rate, (7, 2, 0), (3, 2, 0))
        # self.block2 = ConvLayer(fil_num, 2*fil_num, drop_rate, (4, 1, 0), (2, 2, 0))
        # self.block3 = ConvLayer(2*fil_num, 4*fil_num, drop_rate, (3, 1, 0), (2, 2, 0))
        # self.block4 = ConvLayer(4*fil_num, 8*fil_num, drop_rate, (3, 1, 0), (2, 1, 0))
        self.block_modules = nn.Sequential(*self.block_modules)
        self.attention = attention
        if attention:
            self.attention_module = AttentionModule(8*fil_num, num_classes, drop_rate=drop_rate)
            # self.attention_module = AttentionModule(8*fil_num, in_channels, drop_rate=drop_rate)

        dummy_inp = torch.rand(input_size)
        ic(dummy_inp.shape)
        dummy_feats = self.forward(dummy_inp, stage='get_features')
        # ic(dummy_feats.shape)
        if self.attention:
            dummy_feats = dummy_feats[0]
        ic(dummy_feats.shape)
        self.in_features = list(dummy_feats.shape)

        # self.gap = nn.AvgPool3d((6,7,6), stride=(6,6,6))

        self._init_weights()        

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                init.constant_(m.bias, 0)


    def forward(self, x, stage='normal'):
        if self.instance_norm:
            x = self.norm(x)
        for i in range(self.depth):
            x = self.block_modules[i](x)
            ic(x.size())
        # x = self.block1(x)
        # x = self.block2(x)
        # x = self.block3(x)
        # x = self.block4(x)
        
        if self.attention:
            att, feats = self.attention_module(x)
            return (att, feats)
        else:
            feats = x
            return feats

def load_state_dict(net, state_dict):
    print('----------load_state_dict----------')
    # print('net state_dict: ', net.state_dict().keys())
    # print('ckpt state_dict: ', state_dict.keys())
    try:
        print('--> try')
        net.load_state_dict(state_dict)
        print('try: loaded')
    except RuntimeError as e:
        print('--> except')
        if 'Missing key(s) in state_dict:' in str(e):
            net.load_state_dict({
                key.replace('module.', '', 1): value
                for key, value in state_dict.items()
            })
            print('except: loaded')

network_dict = {"alexnet": AlexNetFc,
                "resnet18": ResNet18Fc,
                "resnet34": ResNet34Fc,
                "resnet50": ResNet50Fc,
                "resnet101": ResNet101Fc,
                "resnet152": ResNet152Fc,
                "resnet3d": ResNet3D}
