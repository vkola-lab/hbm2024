#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 13:03:58 2021
@author: cxue2
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import pandas as pd
import numpy as np
import tqdm
from tsnecuda import TSNE
from cupyx.scipy.ndimage import zoom
import cupy

import sys
sys.path.append('../DeepDG')
from network import img_network, common_network
from train import get_args
import alg
from alg import alg, modelopera
import torch
import icecream
from icecream import install
install()

# 'mri', 'emb'
_input = 'emb'
_legend = True

fn = '/home/dlteif/csv_files/NACC_ADNI_AIBL_FHS_NC_MCI_AD.csv'

df = pd.read_csv(fn)
# df = df[df['dataset'] == 'NACC']

# checkpoint_dir = '/data_1/dlteif/echo_output_dir/ERM/unet3d_gap_pretrained_attention_NACC_ADNI_NC_MCI_ADaug_minmax_normalize_lr0.0008_0/'
# checkpoint_dir = '/data_1/dlteif/echo_output_dir/XAIalign/unet3d_gap_pretrained_attention_NACC_ADNI_NC_MCI_ADaug_minmax_normalize_lr0.0008layer_feats_classifier_l2_start0_lambda1e-05_gamma0.0_0/'
args = get_args()
args.counts = None
# algorithm = 'Vanilla'
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
ic(torch.cuda.device_count())
algorithm_class = alg.get_algorithm_class(args.algorithm)

algorithm = algorithm_class(args)
state_dict = torch.load(args.resume, map_location='cpu')['model_dict']
del state_dict['criterion.weight']
ic(state_dict.keys())
algorithm.load_state_dict(state_dict)
algorithm.to(device)
algorithm.eval()

if _input == 'emb':
    x = np.zeros((len(df), 39204))
    for i in tqdm.tqdm(range(len(df))):
        fpath = os.path.join(os.path.dirname(args.resume), 'embedding', df.loc[i, 'filename'].replace('.nii', '.npy'))
        mri = np.load(fpath)
        x[i,:] = cupy.asnumpy(cupy.ravel(mri))

        # fpath = df.loc[i,'path'] + df.loc[i, 'filename'].replace('.nii', '.npy')
        # if not os.path.exists(os.path.join(os.path.dirname(args.resume), 'embedding')):
        #     os.makedirs(os.path.join(os.path.dirname(args.resume), 'embedding'))
        # mri = np.load(fpath)[None,None,...]
        # with torch.cuda.amp.autocast():
        #     feats, _ = algorithm.predict(torch.from_numpy(mri).float().to(device), stage='get_features')
        #     ic(feats[-1].size())
        #     ic(os.path.join(os.path.dirname(args.resume),'embedding', df.loc[i, 'filename'].replace('.nii', '.npy')))
        #     np.save(os.path.join(os.path.dirname(args.resume), 'embedding', df.loc[i, 'filename'].replace('.nii', '.npy')), feats[-1].detach().cpu().numpy())
        # x[i,:] = cupy.asnumpy(cupy.ravel(feats[-1].detach().cpu().numpy()))
    
    x_embedded = TSNE(n_components=2, perplexity=40, n_iter=4000, verbose=True).fit_transform(x)

elif _input == 'mri':
    x = np.zeros((len(df), 114264))
    for i in tqdm.tqdm(range(len(df))):
        _ = df.loc[i, 'path'] + df.loc[i, 'filename'].replace('.nii', '.npy')
        _ = cupy.load(_)
        _ = zoom(_, (1/4, 1/4, 1/4))
        x[i,:] = cupy.asnumpy(cupy.ravel(_))
        # print(x[i,:].shape)

    x_embedded = TSNE(n_components=2, perplexity=30, n_iter=3000, verbose=True).fit_transform(x)

df['tsne_0'] = x_embedded[:, 0]
df['tsne_1'] = x_embedded[:, 1]
# df['tsne_2'] = x_embedded[:, 2]

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(14, 6)})
sns.set(style='white')

# ax = plt.figure(figsize=(16,10)).gca(projection='3d')
# ax.scatter(df['tsne_0'], df['tsne_1'], df['tsne_2'], c=df['dataset'], cmap='tab10')
style='cohort'
g = sns.scatterplot(data=df, x='tsne_0', y='tsne_1', hue='cohort', s=36, palette='tab10', legend=_legend, style=style) #style='diagnosis'
if _legend: g.legend(loc='center left', bbox_to_anchor=(1, 1), fontsize=18)
g.set(yticklabels=[], xticklabels=[], xlabel=None, ylabel=None)
g.axis('off')
fig = g.get_figure()
fig.tight_layout()
# colorPalette = sns.color_palette('tab10', n_colors=4)
# cohorts = {'NACC': 0, 'ADNI': 1, 'AIBL': 2, 'FHS': 3}
# markers = ['^', 'v', '.']
# fig, axes = plt.subplots(figsize=(11.7, 8.27))
# for i, diag in enumerate(['NC','MCI','AD']):
#     for cohort, cidx in cohorts.items():
#         plot_df = df[(df['cohort'] == cohort) & (df['diagnosis'] == diag)]
#         sns.scatterplot(plot_df['tsne_0'], plot_df['tsne_1'], marker=markers[i], c=cohort, cmap='tab10')
        # plt.annotate(i, xy=(row['tsne_0'], row['tsne_1']), horizontalalignment='center', verticalalignment='center', size=7, color=colorPalette[cohorts[row['cohort']]])
        # plt.plot(row['tsne_0'], row['tsne_1'], markers[i], markersize=5, color=colorPalette[cohorts[row['cohort']]], label=row['cohort'])
# plt.axis('off')
# plt.legend(list(cohorts.keys()), bbox_to_anchor=(1, 1), loc='upper right')

plt.show()
fig.savefig('./{}_SDG_tsne_{}_{}_cohort_style_{}.pdf'.format(args.algorithm, _input, _legend, style))
plt.clf()

# g = sns.scatterplot(data=df, x='tsne_0', y='tsne_1', hue='diagnosis', s=16, palette='tab20', legend=_legend)
# if _legend: g.legend(loc='center left', bbox_to_anchor=(1, 1))
# g.set(yticklabels=[], xticklabels=[], xlabel=None, ylabel=None)
# g.axis('off')
# fig = g.get_figure()
# fig.tight_layout()
# fig.savefig('./tsne_{}_{}_diag.png'.format(_input, _legend))
# plt.show()

g = sns.scatterplot(data=df, x='tsne_0', y='tsne_1', hue='scanner_brand', s=16, palette='tab10', legend=_legend)
if _legend: g.legend(loc='center left', bbox_to_anchor=(1, 1))
g.set(yticklabels=[], xticklabels=[], xlabel=None, ylabel=None)
g.axis('off')
fig = g.get_figure()
fig.tight_layout()
fig.savefig('./{}_SDG_tsne_{}_{}_scanner_brand.png'.format(args.algorithm, _input, _legend))
plt.show()