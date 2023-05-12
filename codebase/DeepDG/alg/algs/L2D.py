mport numpy as np
import torch
import os
from torch import nn
from torch.nn import functional as F
from alg.opt import get_optimizer
from alg.algs.ERM import ERM
from network.common_network import AugNet3D
from utils.contrastive_loss import SupConLoss

class L2D(ERM):
    def __init__(self, args):
        super(L2D, self).__init__(args)
        self.args = args
        self.convertor = AugNet3D(1)
        self.convertor_opt = get_optimizer(self.convertor, self.args, True)
        self.centroids = 0
        self.d_representation = 0
        self.flag = False
        self.con = SupConLoss()

    def update(self, minibatches, opt, sch, scaler):
        device = list(self.parameters())[0].device
        self.weight = self.weight.to(device)
        with torch.cuda.amp.autocast():
        
        pass