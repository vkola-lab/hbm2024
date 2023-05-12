# coding=utf-8
from pyexpat import features
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from alg.modelopera import get_fea
from network import common_network
from alg.pl_algs.base import Algorithm
import numpy as np
import re
import sys
sys.path.append('../')
from utils.util import allreduce_tensor, get_cuda_device
from alg.opt import get_optimizer, get_scheduler
from datautil.getdataloader import get_img_dataloader
import torchmetrics
from pytorch_lightning.trainer.supporters import CombinedLoader
import wandb 

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, args):
        super(ERM, self).__init__(args)
        print('ERM __init__')
        # args.counts = np.stack(args.counts, axis=0).sum(axis=0)
        # print(args.counts)
        # total = np.sum(args.counts)
        # print(total/torch.Tensor(args.counts))
        self.args = args
        # device = get_cuda_device()
        # self.criterion = nn.CrossEntropyLoss(weight=total/torch.Tensor(args.counts)).to(device)
        self.featurizer = get_fea(args)
        self.classifier = common_network.feat_classifier(
            args.num_classes, self.featurizer.in_features, args.classifier)

        self.network = nn.Sequential(
            self.featurizer, self.classifier)
        self.num_classes = args.num_classes
        self.classes = ['NC', 'MCI', 'AD'] if self.num_classes == 3 else ['NC', 'AD']
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=args.num_classes)
        self.confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=args.num_classes)
    
    def update(self, minibatches, opt, sch, rank=None):
        print('--------------def update----------------')
        device = get_cuda_device() if not rank else rank
        all_x = torch.cat([data[0].to(device).float() for data in minibatches])
        all_y = torch.cat([data[1].to(device).long() for data in minibatches])
        print('all_x: ', all_x.size())
        # all_p = self.predict(all_x)
        # all_probs =  
        loss = self.criterion(self.predict(all_x), all_y)
        if self.dist:
            loss = allreduce_tensor(loss.data)
        print('class: ', loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()
        if sch:
            sch.step()
        return {'class': loss.item()}, sch

    def configure_optimizers(self):
        print('------------------ configure_optimizers -----------------')
        opt = get_optimizer(self, self.args)
        sch = get_scheduler(opt, self.args)
        sch.optimizer = opt
        return [opt], [sch]

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)
    
    def predict(self, x):
        # print('network device: ', list(self.network.parameters())[0].device)
        # print('x device: ', x.device)
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

    def get_dataloaders(self):
        self.train_loaders, self.val_loaders, self.test_loaders = get_img_dataloader(self.args)
        self.weights, self.counts = [], []
        for name, ldr in self.train_loaders.items():
            weights, count = ldr.dataset.get_sample_weights()
            self.weights.append(weights)
            self.counts.append(count)
        ic(self.counts)
        train_length = sum([len(ldr) for _,ldr in self.train_loaders.items()])
        ic(train_length)
        val_length = sum([len(ldr) for ldr in self.val_loaders])
        ic(val_length)
        test_length = sum([len(ldr) for ldr in self.test_loaders])
        ic(test_length)
        self.counts = np.stack(self.counts, axis=0).sum(axis=0)
        total = np.sum(self.counts)
        self.criterion = nn.CrossEntropyLoss(weight=total/torch.Tensor(self.counts))
        self.val_accs = nn.ModuleList([torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes) for i in range(len(self.val_loaders))])
        self.test_accs = nn.ModuleList([torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes) for i in range(len(self.test_loaders))])
        # exit()

    def train_dataloader(self):
        print('------train_dataloader---------')
        if not hasattr(self, 'train_loaders'):
            self.get_dataloaders()

        return CombinedLoader(self.train_loaders, mode='max_size_cycle')

    def val_dataloader(self):
        print('------val_dataloader---------')
        if not hasattr(self, 'val_loaders'):
            self.get_dataloaders()
        return self.val_loaders

    def test_dataloader(self):
        print('------test_dataloader---------')
        if not hasattr(self, 'test_loaders'):
            self.get_dataloaders()
        return self.test_loaders 

    def training_step(self, batch, batch_idx, train_idx=0):
        ic('-------------- training_step ------------------')
        ic(train_idx, type(batch), len(batch))
        all_x = torch.cat([data[1].float() for _,data in batch.items()])
        all_y = torch.cat([data[2].long() for _,data in batch.items()])
        ic(all_x.size(), all_y.size())
        label_list = all_y.tolist()
        count = float(len(label_list))
        ic(count)
            
        uniques = sorted(list(set(label_list)))
        ic(uniques)
        counts = [float(label_list.count(i)) for i in uniques]
        ic(counts)
        
        weights = [count / c for c in counts]
        ic(weights)

        output = self.predict(all_x)
        preds = output.argmax(1)

        loss = self.criterion(output, all_y)
        acc = self.accuracy(preds, all_y)
        # cm = self.confmat(preds, all_y)
        # self.log('train_loss', loss, sync_dist=True)
        # self.log('train_acc', acc, sync_dist=True)
        return {'loss': loss, 'train_acc': acc}

    def training_epoch_end(self, outputs, train_idx=0):
        ic(outputs[0].keys())
        train_loss_mean = sum([o['loss'] for o in outputs]) / len(outputs)
        train_acc_mean = self.accuracy.compute()
        # train_cm = torch.stack([o['cm'] for o in outputs], dim=0).squeeze()
        # ic(outputs[0]['cm'], train_cm.size())
        # show val_acc in progress bar but only log val_loss
        self.log('mean_train_loss', train_loss_mean.item(), sync_dist=True, on_step=False, on_epoch=True)
        self.log('mean_train_acc', train_acc_mean, sync_dist=True, on_step=False, on_epoch=True)

        results = {'progress_bar': {'train_acc': train_acc_mean}, 'log': {'train_loss': train_loss_mean.item(), 'train_acc': train_acc_mean},
                   'train_loss': train_loss_mean.item()}
        # self.accuracy.reset()
        # self.confmat.reset()
        # return results

    def validation_step(self, batch, batch_idx, val_idx=0):
        ic('------validation_step-------')
        ic(len(batch))
        _, x, y, _ = batch
        output = self.predict(x)
        preds = output.argmax(1)
        acc = self.val_accs[val_idx](preds, y)
        ic(acc)
        cm = self.confmat(preds, y)
        loss = self.criterion(output, y)
        
        # self.log('val_loss', loss, rank_zero_only=True)
        # self.log('val_acc', acc, rank_zero_only=True)

        return {'val_loss': loss, 'val_acc': acc, 'cm': cm}

    def test_step(self, batch, batch_idx, test_idx=0):
        ic('--------test_step--------')
        _, x, y, _ = batch
        output = self.predict(x)
        preds = output.argmax(1)
        acc = self.test_accs[test_idx](preds, y)
        ic(acc)
        loss = self.criterion(output, y)
        cm = self.confmat(preds, y)
        
        # self.log('test_loss', loss, rank_zero_only=True)
        # self.log('test_acc', acc, rank_zero_only=True)

        return {'test_loss': loss, 'test_acc': acc, 'cm': cm}

    def test_epoch_end(self, outputs, test_idx=0):
        ic(len(outputs), len(outputs[0]))
        results = {}
        
        for idx, dset_output in enumerate(outputs):
            results[f'test_loss_{idx}'] = sum([o['test_loss'] for o in dset_output]) / len(dset_output)
            results[f'test_acc_{idx}'] = sum([o['test_acc'] for o in dset_output]) / len(dset_output)
            cm = torch.sum(torch.stack([o['cm'] for o in dset_output], dim=0).squeeze(), dim=0)
            cm = cm.cpu().numpy()
            fig = plt.figure(figsize=(10,5))
            df_cm = pd.DataFrame(cm, columns=range(self.num_classes), index=range(self.num_classes))
            df_cm.index.name = 'Actual'
            df_cm.columns.name = 'Predicted'
            ic(df_cm)
            plot = sns.heatmap(df_cm, annot=True, fmt='g', annot_kws={'size': 16})
            self.logger.experiment.log({f'confusion_matrix_{idx}': wandb.Image(plt)})
            FP = cm.sum(axis=0) - np.diag(cm) 
            FN = cm.sum(axis=1) - np.diag(cm)
            TP = np.diag(cm)
            TN = cm.sum() - (FP + FN + TP)
            FP = FP.astype(float)
            FN = FN.astype(float)
            TP = TP.astype(float)
            TN = TN.astype(float)
            # Sensitivity, hit rate, recall, or true positive rate
            TPR = TP/(TP+FN)
            # Specificity or true negative rate
            TNR = TN/(TN+FP) 
            # Precision or positive predictive value
            PREC = TP/(TP+FP)
            # Negative predictive value
            NPV = TN/(TN+FN)
            # Fall out or false positive rate
            FPR = FP/(FP+TN)
            # False negative rate
            FNR = FN/(TP+FN)
            # False discovery rate
            FDR = FP/(TP+FP)
            # Overall accuracy for each class
            ACC = (TP+TN)/(TP+FP+FN+TN)
            TOTAL_ACC = np.diag(cm).sum() / cm.sum()
            results[f'CLS_BALANCED_ACC_{idx}'] = TPR.sum() / 3 # num of classes

            F1 = 2 * PREC * TPR / (PREC + TPR)
            MCC = (TP*TN - FP*FN) / (np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)) + 0.000000001)
            for i in range(self.num_classes):
                results[f'{self.classes[i]}_F1_{idx}'] = F1[i].item()
                results[f'{self.classes[i]}_MCC_{idx}'] = MCC[i].item()
                results[f'{self.classes[i]}_TPR_{idx}'] = TPR[i].item()
                results[f'{self.classes[i]}_TNR_{idx}'] = TNR[i].item()
            
            results[f'MACRO_F1_{idx}'] = F1.sum() / self.num_classes
            results[f'WEIGHTED_F1_{idx}'] = (F1 * cm.sum(axis=0)).sum() / cm.sum() 
            
            MACRO_MCC = MCC.sum() / self.num_classes
            results[f'MACRO_MCC_{idx}'] = MACRO_MCC
            WEIGHTED_MCC = (MCC * cm.sum(axis=0)).sum() / cm.sum()
            results[f'WEIGHTED_MCC_{idx}'] = WEIGHTED_MCC
            # self.log(f'mean_test_loss_{idx}', results[f'mean_test_loss_{idx}'], sync_dist=True)
            # self.log(f'mean_test_acc_{idx}', results[f'mean_test_acc_{idx}'], sync_dist=True)

        test_acc = [v for k,v in results.items() if 'acc' in k]
        results['mean_test_acc'] = sum(test_acc) / len(test_acc)
        test_loss = [v for k,v in results.items() if 'loss' in k]
        results['mean_test_loss'] = sum(test_loss) / len(test_loss)
        metric_test_acc = 0.0
        for i in range(len(self.test_loaders)):
            results[f'{self.test_loaders[i].dataset.domain_name}/metric_test_acc'] = self.test_accs[i].compute()
            metric_test_acc += results[f'{self.test_loaders[i].dataset.domain_name}/metric_test_acc']
        results['metric_test_acc'] = metric_test_acc / len(self.test_loaders)
        
        metric_test_confmat = self.confmat.compute()
        ic(metric_test_confmat.size())
        cm = metric_test_confmat.cpu().numpy()
        fig = plt.figure(figsize=(10,5))
        df_cm = pd.DataFrame(cm, columns=range(self.num_classes), index=range(self.num_classes))
        df_cm.index.name = 'Actual'
        df_cm.columns.name = 'Predicted'
        ic(df_cm)
        plot = sns.heatmap(df_cm, annot=True, fmt='g', annot_kws={'size': 16})
        self.logger.experiment.log({f'metric_confusion_matrix': wandb.Image(plt)})
        self.log_dict(results, sync_dist=True, on_step=False, on_epoch=True)
        results = {'progress_bar': {'mean_test_acc': results['mean_test_acc']}, 'log': results,
                   'mean_test_loss': results['mean_test_loss']}

        # self.accuracy.reset()
        # self.confmat.reset()
        return results

    
    def validation_epoch_end(self, outputs):
        results = {}
        for idx, dset_output in enumerate(outputs):
            results[f'val_loss_{idx}'] = sum([o['val_loss'] for o in dset_output]) / len(dset_output)
            results[f'val_acc_{idx}'] = sum([o['val_acc'] for o in dset_output]) / len(dset_output)
        # show val_acc in progress bar but only log val_loss
        val_acc = [v for k,v in results.items() if 'acc' in k]
        results['mean_val_acc'] = sum(val_acc) / len(val_acc)
        val_loss = [v for k,v in results.items() if 'loss' in k]
        results['mean_val_loss'] = sum(val_loss) / len(val_loss)
        metric_val_acc = 0.0
        for i in range(len(self.val_loaders)):
            results[f'{self.val_loaders[i].dataset.domain_name}/metric_val_acc'] = self.val_accs[i].compute()
            metric_val_acc += results[f'{self.val_loaders[i].dataset.domain_name}/metric_val_acc']
        results['metric_val_acc'] = metric_val_acc / len(self.val_loaders)

        self.log_dict(results, sync_dist=True, on_step=False, on_epoch=True)
        results = {'progress_bar': {'mean_val_acc':  results['mean_val_acc']}, 'log': results,
                   'mean_val_loss': results['mean_val_loss']}
        # self.accuracy.reset()
        # self.confmat.reset()
        return results