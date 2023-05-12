import sys
sys.path.append('../')
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from matplotlib.colors import LinearSegmentedColormap

import alg
from alg import alg
from utils.util import set_random_seed, print_args
from train import get_args
import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

import seaborn as sns
from icecream import ic
from torchsummary import summary

from torchmetrics import PeakSignalNoiseRatio as PSNR
from utils.plot_brains import plot_glass_brain, upsample
import nibabel as nib
from nilearn import datasets
from functools import reduce
from operator import concat
from collections import defaultdict
import pandas as pd

class TorchNorm(nn.Module):
    def __init__(self, p=2):
        super(TorchNorm, self).__init__()
        self.p = p
    
    def forward(self, a, b):
        return torch.norm((a*b), p=self.p)

colors = []
for l in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255,l))
for l in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,l))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)


class ModifiedFeaturizer(nn.Module):
    def __init__(self, featurizer, shap_priors) -> None:
        super().__init__()
        self.featurizer = featurizer
        self.shap_priors = F.softmax(shap_priors)

    def forward(self, x):
        if self.featurizer.instance_norm:
            x = self.featurizer.norm(x)

        x = self.featurizer.block_modules(x)
        ic(x.size())

        if self.featurizer.attention:
            feats = self.featurizer.attention_module.conv(x)
            ic(feats.size(), self.shap_priors.size())
            self.att = self.featurizer.attention_module.attention(x)
            out = feats * F.softmax(self.att)
            mod_out = feats * self.shap_priors
            return out, mod_out
        else:
            self.att = None
            out = x
            return out


if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)
    s = print_args(args, [])

    print('data dir: ', args.data_dir)
    names = args.img_dataset[args.dataset]
    domain_label = names.index(args.cohort.split('_')[0])
    
    if 'train' in args.data_dir:
        transform = imgutil.image_train(args.dataset)
    else:
        transform = imgutil.image_test(args.dataset)

    dataset = ImageDataset(args.dataset, args.task, args.data_dir,
                                           args.cohort, domain_label, transform=transform, indices=None, test_envs=args.test_envs, filename=args.data_dir, num_classes=args.num_classes)
    
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.N_WORKERS, drop_last=False, shuffle=True)
    

    print('loaded data.')
    
    args.weights, args.counts = dataset.get_sample_weights()
    print(type(args.counts))

    priors_path = os.path.join(f'output_dir/shap_maps/9x11x9/ERM/{args.net}_{args.classifier}_attention_{args.cohort}_minmax_normalize_lr0.0008_{args.fold}', args.cohort, args.layer)
    ic(priors_path)
    classes = ['NC','AD'] if args.num_classes == 2 else ['NC', 'MCI', 'AD']

    shap_priors = []
    for i, cls in enumerate(classes):
        assert os.path.exists(f'{priors_path}/{cls}_avg.npy')
        shap_priors.append(torch.from_numpy(np.load(f'{priors_path}/{cls}_avg.npy')[i]))


    shap_priors = torch.stack(shap_priors)
    ic(shap_priors.shape)

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    args.device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id is not None else 'cpu')
    algorithm = algorithm_class(args).to(args.device)
    algorithm.eval()
    

    # ic(algorithm.network.state_dict().keys())
    # ic(torch.nn.Sequential(*list(algorithm.network[0].modules())[:-2]).state_dict().keys())
    
    summary(algorithm.network[0], input_size=(1,182,218,182))
    # exit()

    # replace attention maps by SHAP priors
    if args.resume:
        print('args.resume: ', args.resume)
        ckpt = torch.load(args.resume, map_location='cpu')
        state_dict = ckpt['model_dict']
        # ic(state_dict.keys())
        # ic(algorithm.state_dict().keys())
        algorithm.load_state_dict(state_dict, strict=True)
       
        
        print('loaded checkpoint at ', args.resume)
            
        try:
            best_epoch = ckpt['epoch']
            best_valid_acc = ckpt['accuracy']
            print(f'epoch: {best_epoch}, best acc: {best_valid_acc}')
        except:
            pass


    input = torch.randn((1,1,182,218,182))

    ic(type(algorithm.network[0]))

    # net_copy = copy.deepcopy(algorithm.network[0])
    ic(args.batch_size)
    algorithm.network[0] = ModifiedFeaturizer(algorithm.network[0], shap_priors.to(args.device))
    
    if args.align_loss == 'cos':
        sim_loss = nn.CosineSimilarity(dim=0, eps=1e-6)
    elif args.align_loss == 'mse':
        sim_loss = nn.MSELoss()
    elif args.align_loss == 'psnr':
        sim_loss = PSNR()
    elif args.align_loss == 'l2':
        sim_loss = TorchNorm(p=2)

    mri_infolist = defaultdict(list)
    final_count = 0
    mri_count = len(dataloader)
    correct, total = 0, 0
    class_name = {0: 'NC', 1: 'MCI', 2: 'AD'}
    label_list, preds_list, mod_preds_list = [], [], []
    for idx, data in enumerate(tqdm(dataloader)):
        if idx == mri_count:
            break
        device = list(algorithm.parameters())[0].device
        filenames = data[0]
        inputs = data[1].float().to(device)
        labels = data[2].long()
        print('inputs: ', inputs.size())
        mri_infolist['filename'].extend(filenames)
        mri_infolist['label'].extend(labels.tolist())

        feats, mod_feats = algorithm.network[0](inputs)
        output, mod_output = algorithm.network[1](feats), algorithm.network[1](mod_feats)
        ic(output.size(), mod_output.size())
        probs = F.softmax(output, dim=1)
        mod_probs = F.softmax(mod_output, dim=1)
        ic(probs, mod_probs)
        pred_probs, pred_labels = probs.sort(dim=1, descending=True)
        mod_pred_probs, mod_pred_labels = mod_probs.sort(dim=1, descending=True)
        att = algorithm.network[0].att
        att = (att - att.min()) / (att.max() - att.min())
        # att = F.softmax(att)
        ic(att.min(), att.max())
        shap_priors = (shap_priors - shap_priors.min()) / (shap_priors.max() - shap_priors.min())
        # shap_priors = F.softmax(shap_priors)
        ic(shap_priors.min(), shap_priors.max())

        # mni = datasets.load_mni152_template()
        # index = 3
        # prediction = pred_labels[index,0].item()
        # mod_prediction = mod_pred_labels[index,0].item()
        # att_img = nib.Nifti1Image(upsample(att[0,prediction,:,:,:].detach().cpu().numpy(), target_shape=mni.get_fdata().shape), affine=mni.affine)
        # shap_img = nib.Nifti1Image(upsample(shap_priors[mod_prediction,:,:,:].detach().cpu().numpy(), target_shape=mni.get_fdata().shape), affine=mni.affine)

        classes = ['NC', 'MCI', 'AD']
        # plot_glass_brain(att[0,prediction,:,:,:].detach().cpu().numpy(), att_img, classes[prediction], os.path.dirname(args.resume), prefix='att', threshold=1/2)
        # plot_glass_brain(shap_priors[mod_prediction,:,:,:].detach().cpu().numpy(), shap_img, classes[mod_prediction], os.path.dirname(args.resume), prefix='shap_prior', threshold=1/2)
        # plot_glass_brain(shap_priors[labels[index].item(),:,:,:].detach().cpu().numpy(), shap_img, classes[labels[index].item()], os.path.dirname(args.resume), prefix='shap_prior_gt', threshold=1/2)
        # shap_priors = shap_priors.unsqueeze(0).repeat(args.batch_size, 1, 1, 1, 1)
        if args.align_loss == 'cos':
            ic(att.size(), shap_priors.size())
            loss = torch.stack([torch.stack([sim_loss(att[i,j,...].detach().cpu().flatten(), shap_priors[j,...].detach().cpu().flatten()) for j in range(3)]) for i in range(att.size(0))])
            ic(loss.size(), loss[0])
        else:
            loss = [sim_loss(att[i,...].detach().cpu(), shap_priors) for i in range(att.size(0))]
        loss_values, indices = loss.sort(dim=1, descending=True)
        ic(loss)
        
        print('pred_labels: ', pred_labels.size())
        ic(pred_labels[:,0], mod_pred_labels[:,0], labels)
        ic((pred_labels[:,0] == mod_pred_labels[:,0]).sum())
        
        mri_infolist['pred.'].extend([p[0] for p in pred_labels.cpu().tolist()])
        mri_infolist['SHAPtt pred.'].extend([p[0] for p in mod_pred_labels.cpu().tolist()])
        mri_infolist['cosine sim. with pred.'].extend([loss[i,pred_labels[i,0].item()].item() for i in range(loss.size(0))])
        mri_infolist['cosine sim. with gt'].extend([loss[i,labels[i].item()].item() for i in range(loss.size(0))])
        label_list.extend(labels.cpu().tolist())
        preds_list.extend([p[0] for p in pred_labels.cpu().tolist()])
        mod_preds_list.extend([p[0] for p in mod_pred_labels.cpu().tolist()])
        del output
        del probs
        del pred_probs
        del pred_labels
        
        final_count += 1
        # if final_count == 2:
        #     exit()


    df = pd.DataFrame(mri_infolist)
    df.to_csv(os.path.join(os.path.dirname(args.resume), f'{os.path.basename(args.data_dir)[:-4]}_test_time_adaptation_correctness.csv'))
    label_list = np.array(label_list).flatten()
    preds_list = np.array(preds_list).flatten()
    mod_preds_list = np.array(mod_preds_list).flatten()
    print(label_list.shape, preds_list.shape)
    cm = confusion_matrix(label_list, preds_list, labels=[0, 1, 2])
    mod_cm = confusion_matrix(label_list, mod_preds_list, labels=[0,1,2])
    print(cm, mod_cm)
    
    ax = sns.heatmap(cm, annot=True, fmt='g')
    ax.set_title('Confusion matrix, acc={:.2f}'.format(np.sum(np.diag(cm)) * 100.0 / np.sum(cm)))
    ax.set_xticklabels(['NC', 'MCI', 'AD'])
    ax.set_yticklabels(['NC', 'MCI', 'AD'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.savefig(os.path.join(os.path.dirname(args.resume),f"confusion_matrix_{'_'.join(args.data_dir.split('/')[-3:])[:-4]}.png"), dpi=150)
    plt.clf()
    
    ax = sns.heatmap(mod_cm, annot=True, fmt='g')
    ax.set_title('Confusion matrix, acc={:.2f}'.format(np.sum(np.diag(mod_cm)) * 100.0 / np.sum(mod_cm)))
    ax.set_xticklabels(['NC', 'MCI', 'AD'])
    ax.set_yticklabels(['NC', 'MCI', 'AD'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.savefig(os.path.join(os.path.dirname(args.resume),f"confusion_matrix_shap_{'_'.join(args.data_dir.split('/')[-3:])[:-4]}.png"), dpi=150)

    print('final count: ', final_count)
        
