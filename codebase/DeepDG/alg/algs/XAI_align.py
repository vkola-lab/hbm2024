# coding=utf-8
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch.autograd as autograd
from torch.autograd import Variable
from torchmetrics import PeakSignalNoiseRatio as PSNR


from skimage.transform import resize

from alg.algs.ERM import ERM

class TorchNorm(nn.Module):
    def __init__(self, p=2, weight=None):
        super(TorchNorm, self).__init__()
        self.p = p
        self.weight = weight
    
    def forward(self, a, b, labels=None):
        if labels is not None:
            device = a.device
            weights = torch.tensor([self.weight[l] for l in labels]).to(device)
            return (torch.norm((a*b), p=self.p) * weights).mean()
        return torch.norm((a*b), p=self.p)


class AlignCriterion(nn.Module):
    def __init__(self, alpha=2, weight=None):
        self.alpha = alpha
        self.weight = weight

    def forward(self, x, y, labels=None):
        if labels is not None:
            device = a.device
            weights = torch.tensor([self.weight[l] for l in labels]).to(device)
            return ((x - y).norm(p=2, dim=1).pow(self.alpha) * weights).mean()
        return (x - y).norm(p=2, dim=1).pow(self.alpha).mean()    

class XAIalign(ERM):
    def __init__(self, args) -> None:
        super(XAIalign, self).__init__(args)
        ic(self.criterion, self.weight)
        self.gamma = args.align_gamma
        self.lmbda = args.align_lambda 
        self.num_classes = args.num_classes
        self.layer = args.layer
        self.net = args.net
        self.align_cohort = args.align_cohort
        if (('resnet3d' in args.net) or ('attcnn' in args.net)) and args.attention:
            self.base_dir = f'/data_1/dlteif/shap_maps/9x11x9/ERM/{args.net}_{args.classifier}_attention_{args.cohort}_minmax_normalize_lr{args.lr}_{args.fold}/{args.cohort}/classifier/'
        else:
            if os.uname().nodename == 'echo':
                self.base_dir = f'/data_1/dlteif/shap_maps/ERM/{args.net}_{args.classifier}_pretrained_attention_{args.cohort}aug_minmax_normalize_lr{args.lr}_{args.fold}/{args.cohort}/classifier/'
            else:
                self.base_dir = f'output_dir/shap_maps/ERM/{args.net}_{args.classifier}_pretrained_attention_{args.cohort}aug_minmax_normalize_lr{args.lr}_{args.fold}/{args.cohort}/classifier/'

        self.classes = ['NC','MCI','AD']
        print(self.base_dir)
        
        if self.layer == 'input':
            all_x = torch.zeros(1,1,182,218,182)
            all_f = self.featurizer(all_x, stage='get_features')
            if isinstance(all_f, tuple) and self.net == 'attcnn':
                att, feats = all_f
            elif 'resnet3d' in self.net:
                feats, pool_out = all_f
                att = None
            else:
                feats = all_f
            self.feat_size = feats.shape
            print('feat size: ', self.feat_size)

        avg_shap_maps = {}
        if not args.eval:
            for i, cls in enumerate(self.classes):
                ic(f'{self.base_dir}{cls}_avg.npy')
                assert os.path.exists(f'{self.base_dir}{cls}_avg.npy'), 'SHAP Priors do not exist!!'
                try:
                    shap_map = np.load(f'{self.base_dir}{cls}_avg.npy')
                    #  resize in self.update() if needed
                    if self.layer == 'input':
                        shap_map = torch.from_numpy(resize(shap_map, self.feat_size[2:]))
                    else:
                        shap_map = torch.from_numpy(shap_map)
                except:
                    shap_map = None
                avg_shap_maps[i] = shap_map                
                
            self.ref_maps = avg_shap_maps
        
        if self.align_cohort:
            cohort_avg_shap = np.load('/projectnb/ivc-ml/dlteif/SHAP4Med/brain2020/shap_maps/CNN_train_UNION_NC_MCI_AD_COHORTS/UNION/UNION_NC_MCI_AD_splits/layer_input/NACC_avg.npy')
            # resize in self.update() if needed
            if self.layer == 'input':
                cohort_avg_shap = torch.from_numpy(resize(cohort_avg_shap, self.feat_size[2:]))
            else:
                cohort_avg_shap = torch.from_numpy(cohort_avg_shap, self.feat_size[2:])
            self.cohort_ref_maps = torch.stack([cohort_avg_shap for i in range(self.num_classes)])

        if args.align_loss == 'cos':
            self.sim_loss = nn.CosineSimilarity(dim=0, eps=1e-6)
        elif args.align_loss == 'mse':
            self.sim_loss = nn.MSELoss()
        elif args.align_loss == 'psnr':
            self.sim_loss = PSNR()
        elif args.align_loss == 'l2':
            self.sim_loss = TorchNorm(p=2, weight=self.weight)
        elif args.align_loss == 'align':
            self.sim_loss = AlignCriterion(weight=self.weight)

        self.accum_iter = args.accum_iter
        self.acc_steps = 0

    def update(self, minibatches, opt, sch, scaler):
        print('--------------def update----------------')
        device = list(self.parameters())[0].device
        ic(device)
        with torch.cuda.amp.autocast():
            all_x = torch.cat([data[1].to(device).float() for data in minibatches])
            all_y = torch.cat([data[2].to(device).long() for data in minibatches])
            print('all_x: ', all_x.size())


            # if self.net == 'unet3d':
                # all_x = all_x.half()
            
            # all_probs =
            all_f = self.featurizer(all_x, stage='get_features')
            att = None
            if isinstance(all_f, tuple):
                att, feats = all_f
            else:
                feats = all_f
                
            all_p = self.classifier(feats).float()
            
            print('feats: ', feats.size())
            print('all_p: ', all_p.size())
            
            # CE loss between predicted and target labels
            with autocast():
                CE_loss = self.criterion(all_p, all_y)
                ic(CE_loss.item())

                # get classes whose avg SHAP maps do not exist
                none_indices = [i for i in range(len(self.ref_maps)) if self.ref_maps[i] is None]
                ic(none_indices) 
                comparable_y_idxs = [i for i in range(all_y.size(0)) if all_y[i] not in none_indices]
                ic(all_y, comparable_y_idxs)
                sim_loss = torch.Tensor([0.]).to(device).float()
                if len(comparable_y_idxs) > 0:
                    if self.layer == 'input':
                        target_avg_maps = torch.stack([self.ref_maps for i in range(all_x.size(0))])
                        if self.align_cohort:
                            target_cohort_maps = torch.stack([self.cohort_ref_maps for i in range(all_x.size(0))])
                    else:
                        target_avg_maps = torch.stack([self.ref_maps[int(all_y[i].item())] for i in comparable_y_idxs])
                        # target_avg_maps = torch.unsqueeze(target_avg_maps, 0)
                    print('target_avg_maps: ', target_avg_maps.size(), type(target_avg_maps[0]))


                    maps = Variable(target_avg_maps.detach().type(torch.FloatTensor), requires_grad=True)
                    print('maps min: ', maps.min(), ', max: ', maps.max())
                    maps = F.softmax(maps)
                    # maps = normalize(maps, att.min().item(), att.max().item())
                    # maps = normalize(maps, feats.min().item(), feats.max().item())
                    print('Softmax maps min: ', maps.min(), ', max: ', maps.max())
                    print('feats min: ', feats.min(), ', max: ', feats.max())
                    if att != None:
                        print('att min: ', att.min(), ', max: ', att.max())
                        if isinstance(self.sim_loss, nn.CosineSimilarity):
                            sim_loss = - self.sim_loss(torch.flatten(att), torch.flatten(maps.to(device)))
                        elif isinstance(self.sim_loss, PSNR):
                            sim_loss = - self.sim_loss(feats.float(), maps.to(device))
                        else:
                            ic(att.size(), maps.size())
                            ic(all_y, all_y[comparable_y_idxs])
                            sim_loss = self.sim_loss(att[comparable_y_idxs], maps.to(device))

                    else:
                        if isinstance(self.sim_loss, nn.CosineSimilarity):
                            sim_loss = - self.sim_loss(torch.flatten(feats), torch.flatten(maps.to(device)))
                        elif isinstance(self.sim_loss, PSNR):
                            sim_loss = - self.sim_loss(feats, maps.to(device))
                        else:
                            sim_loss = self.sim_loss(feats, maps.to(device))

                ic(sim_loss)
                # weights = torch.tensor([self.weight[y] for y in comparable_y_idxs]).to(device)
                # ic(weights.size(), all_y.size(), len(comparable_y_idxs))
                # sim_loss = sim_loss * weights
                # ic(sim_loss)
                ic((self.lmbda*sim_loss) / CE_loss)
                # LOSS 1
                loss = CE_loss + self.lmbda * sim_loss
                if sim_loss < 0:
                    loss = loss + 1
                # LOSS 2
                if self.align_cohort:
                    cohort_maps = Variable(target_cohort_maps.detach().type(torch.FloatTensor), requires_grad=True)
                    print('Downsampled cohort_maps: ', cohort_maps.size(), cohort_maps.type())
                    print('Cohort maps min: ', cohort_maps.min(), ', max: ', cohort_maps.max())
                    cohort_maps = F.softmax(cohort_maps)
                    print('Softmax Cohort maps min: ', cohort_maps.min(), ', max: ', cohort_maps.max())

                    if att != None:
                        sim_loss_2 =  self.cos(torch.flatten(att), torch.flatten(cohort_maps.to(device)))
                    else:
                        sim_loss_2 =  self.cos(torch.flatten(feats), torch.flatten(cohort_maps.to(device)))
                    
                    print('sim loss 2: ', sim_loss_2.item())

                    
                    loss = loss + self.gamma * sim_loss_2
            ic(loss.device)
            self.acc_steps += 1
            print('class: ', loss.item())

            scaler.scale(loss / self.accum_iter).backward()
            
            if self.acc_steps == self.accum_iter:
                scaler.step(opt)
                if sch:
                    sch.step()
                scaler.update()
                self.zero_grad()
                self.acc_steps = 0
                torch.cuda.empty_cache()
        
        return {'class': CE_loss.item(), 'align': (sim_loss.item() + sim_loss_2.item()) if self.align_cohort else sim_loss.item(), 'total': loss.item()}, sch