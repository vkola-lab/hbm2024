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

from alg.pl_algs.ERM import ERM

class TorchNorm(nn.Module):
    def __init__(self, p=2):
        super(TorchNorm, self).__init__()
        self.p = p
    
    def forward(self, a, b):
        return torch.norm((a*b), p=self.p)


class XAIalign(ERM):
    def __init__(self, args) -> None:
        super(XAIalign, self).__init__(args)
        self.gamma = args.align_gamma
        self.lmbda = args.align_lambda 
        self.num_classes = args.num_classes
        self.layer = args.layer
        self.net = args.net
        self.align_cohort = args.align_cohort
        if 'resnet3d' in args.net and args.attention:
            self.base_dir = f'output_dir/shap_maps/9x11x9/ERM/{args.net}_{args.classifier}_attention_{args.cohort}_minmax_normalize_lr{args.lr}_{args.fold}/{args.cohort}/classifier/'
        else:    
            self.base_dir = f'/data_1/dlteif/shap_maps/9x11x9/ERM/{args.net}_{args.classifier}_attention_{args.cohort}_minmax_normalize_lr{args.lr}_{args.fold}/{args.cohort}/classifier/'
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
                ic(os.path.exists(f'{self.base_dir}{cls}_avg.npy'))
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
            self.sim_loss = TorchNorm(p=2)

        self.accum_iter = args.accum_iter
        self.acc_steps = 0

    def training_step(self, batch, batch_idx, train_idx=0):
        print('--------------def update----------------')
        all_x = torch.cat([data[1].float() for _,data in batch.items()])
        all_y = torch.cat([data[2].long() for _,data in batch.items()])
        ic(all_x.size(), all_y.size())
        
        all_f = self.featurizer(all_x, stage='get_features')
        att = None
        if isinstance(all_f, tuple):
            att, feats = all_f
        else:
            feats = all_f
            
        all_p = self.classifier(feats)
        preds = all_p.argmax(1)
        acc = self.accuracy(preds, all_y) 
        
        print('feats: ', feats.size())
        print('all_p: ', all_p.size())
        
        # CE loss between predicted and target labels
        CE_loss = self.criterion(all_p, all_y)
        ic(CE_loss.item())

        # get classes whose avg SHAP maps do not exist
        # none_indices = [i for i in range(len(self.ref_maps)) if self.ref_maps[i] is None]
        # ic(none_indices) 
        # comparable_y_idxs = [i for i in range(all_y.size(0)) if all_y[i] not in none_indices]
        # ic(all_y, comparable_y_idxs)
        sim_loss = torch.Tensor([0.]).cuda().float()
        # if len(comparable_y_idxs) > 0:
        if self.layer == 'input':
            target_avg_maps = torch.stack([self.ref_maps for i in range(all_x.size(0))])
            if self.align_cohort:
                target_cohort_maps = torch.stack([self.cohort_ref_maps for i in range(all_x.size(0))])
        else:
            target_avg_maps = torch.stack([self.ref_maps[int(all_y[i].item())] for i in range(all_x.size(0))])
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
                sim_loss = - self.sim_loss(torch.flatten(att), torch.flatten(maps.cuda()))
            elif isinstance(self.sim_loss, PSNR):
                sim_loss = - self.sim_loss(feats, maps.cuda())
            else:
                ic(att.size(), maps.size())
                # ic(all_y, all_y[comparable_y_idxs])
                # sim_loss = self.sim_loss(att[comparable_y_idxs], maps.cuda())
                sim_loss = self.sim_loss(att, maps.cuda())

        else:
            if isinstance(self.sim_loss, nn.CosineSimilarity):
                sim_loss = - self.sim_loss(torch.flatten(feats), torch.flatten(maps.cuda()))
            elif isinstance(self.sim_loss, PSNR):
                sim_loss = - self.sim_loss(feats, maps.cuda())
            else:
                sim_loss = self.sim_loss(feats, maps.cuda())

        ic(sim_loss)
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
                sim_loss_2 =  self.cos(torch.flatten(att), torch.flatten(cohort_maps.cuda()))
            else:
                sim_loss_2 =  self.cos(torch.flatten(feats), torch.flatten(cohort_maps.cuda()))
            
            print('sim loss 2: ', sim_loss_2.item())

            
            loss = loss + self.gamma * sim_loss_2
        ic(loss)
        # else:
        #     loss = CE_loss
        
        return {'train_acc': acc, 'class': CE_loss, 'align': (sim_loss + sim_loss_2) if self.align_cohort else sim_loss, 'loss': loss}

    def training_epoch_end(self, outputs, train_idx=0):
        ic(outputs[0].keys())
        results = {}
        results['mean_train_loss'] = sum([o['loss'] for o in outputs]) / len(outputs)
        results['train_class_loss'] = sum([o['class'] for o in outputs]) / len(outputs)
        results['train_align_loss'] = sum([o['align'] for o in outputs]) / len(outputs)
        train_acc_mean = self.accuracy.compute()
        results['train_acc_mean'] = train_acc_mean
        # train_cm = torch.stack([o['cm'] for o in outputs], dim=0).squeeze()
        # ic(outputs[0]['cm'], train_cm.size())
        # show val_acc in progress bar but only log val_loss
        self.log_dict(results, sync_dist=True, on_step=False, on_epoch=True)

        results = {'progress_bar': {'train_acc': train_acc_mean.item()}, 'log': results,
                   'train_loss':  results['mean_train_loss'].item()}
        # self.accuracy.reset()