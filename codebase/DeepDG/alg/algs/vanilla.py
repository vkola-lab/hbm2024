# coding=utf-8
from pyexpat import features
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.algs.base import Algorithm
import numpy as np
import re
from torch.cuda.amp import autocast

class Vanilla(Algorithm):
    """
    Vanilla Training
    """

    def __init__(self, args):
        super(Vanilla, self).__init__(args)
        if args.counts is not None:
            if isinstance(args.counts[0], list):
                args.counts = np.stack(args.counts, axis=0).sum(axis=0)
                print('counts: ', args.counts)
                total = np.sum(args.counts)
                print(total/args.counts)
                self.weight = total/torch.Tensor(args.counts)
            else:
                total = sum(args.counts)
                self.weight = torch.Tensor([total/c for c in args.counts])
        else:
            self.weight = None
        print('weight: ', self.weight)
        # device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id is not None else 'cpu')
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)
        # self.criterion = nn.CrossEntropyLoss().cuda()
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier)

        self.network = nn.Sequential(
            self.featurizer, self.classifier)
        self.acc_steps = 0
        self.accum_iter = args.accum_iter

    def update(self, minibatches, opt, sch, scaler):
        print('--------------def update----------------')
        # domain_labels = 
        device = list(self.parameters())[0].device
        loss = 0
        self.zero_grad()
        for data in minibatches:
            # each minibatch is coming from a different domain 
            x = data[1].to(device).float()
            y = data[2].to(device).long()
            with autocast():
                loss = self.criterion(self.predict(x), y)
            # loss = loss / len(minibatches)
            ic(loss.item())
            ic(y, self.weight)
            scaler.scale(loss / self.accum_iter).backward()
        
        self.acc_steps += 1
        if self.acc_steps == self.accum_iter:
            scaler.step(opt)        
            if sch:
                sch.step()
            scaler.update()
            self.zero_grad()
            self.acc_steps = 0
            torch.cuda.empty_cache()
        
        return {'class': loss.item()}, sch

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)
    
    def predict(self, x, stage='normal', attention=False):
        # print('network device: ', list(self.network.parameters())[0].device)
        # print('x device: ', x.device)
        if stage == 'get_features':
            feats = self.network[0](x, attention=attention)
            output = self.network[1](feats[-1] if attention else feats)
            return feats, output
        else:
            return self.network(x)

    def load_checkpoint(self, state_dict):
        try:
            self.load_checkpoint_helper(state_dict)
        except:
            featurizer_dict = {}
            net_dict = {}
            for key,val in state_dict.items():
                if 'featurizer' in key:
                    featurizer_dict[key] = val
                elif 'network' in key:
                    net_dict[key] = val
            self.featurizer.load_state_dict(featurizer_dict)
            self.classifier.load_state_dict(net_dict)

    def load_checkpoint_helper(self, state_dict):
        try:
            self.load_state_dict(state_dict)
            print('try: loaded')
        except RuntimeError as e:
            print('--> except')
            if 'Missing key(s) in state_dict:' in str(e):
                state_dict = {
                    key.replace('module.', '', 1): value
                    for key, value in state_dict.items()
                }
                state_dict = {
                    key.replace('featurizer.', '', 1).replace('classifier.','',1): value
                    for key, value in state_dict.items()
                }
                state_dict = {
                    re.sub('network.[0-9].', '', key): value
                    for key, value in state_dict.items()
                }
                try:
                    del state_dict['criterion.weight']
                except:
                    pass
                self.load_state_dict(state_dict)
                
                print('except: loaded')

