import shap
import argparse
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from matplotlib.colors import LinearSegmentedColormap

import alg
from alg import alg, modelopera
from utils.util import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names
from datautil.getdataloader import get_img_dataloader
from train import get_args
import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from alg.modelopera import write_scores

import seaborn as sns
from icecream import ic

brain_regions = {1.:'TL hippocampus R',
2.:'TL hippocampus L',
3.:'TL amygdala R',
4.:'TL amygdala L',
5.:'TL anterior temporal lobe medial part R',
6.:'TL anterior temporal lobe medial part L',
7.:'TL anterior temporal lobe lateral part R',
8.:'TL anterior temporal lobe lateral part L',
9.:'TL parahippocampal and ambient gyrus R',
10.:'TL parahippocampal and ambient gyrus L',
11.:'TL superior temporal gyrus middle part R',
12.:'TL superior temporal gyrus middle part L',
13.:'TL middle and inferior temporal gyrus R',
14.:'TL middle and inferior temporal gyrus L',
15.:'TL fusiform gyrus R',
16.:'TL fusiform gyrus L',
17.:'cerebellum R',
18.:'cerebellum L',
19.:'brainstem excluding substantia nigra',
20.:'insula posterior long gyrus L',
21.:'insula posterior long gyrus R',
22.:'OL lateral remainder occipital lobe L',
23.:'OL lateral remainder occipital lobe R',
24.:'CG anterior cingulate gyrus L',
25.:'CG anterior cingulate gyrus R',
26.:'CG posterior cingulate gyrus L',
27.:'CG posterior cingulate gyrus R',
28.:'FL middle frontal gyrus L',
29.:'FL middle frontal gyrus R',
30.:'TL posterior temporal lobe L',
31.:'TL posterior temporal lobe R',
32.:'PL angular gyrus L',
33.:'PL angular gyrus R',
34.:'caudate nucleus L',
35.:'caudate nucleus R',
36.:'nucleus accumbens L',
37.:'nucleus accumbens R',
38.:'putamen L',
39.:'putamen R',
40.:'thalamus L',
41.:'thalamus R',
42.:'pallidum L',
43.:'pallidum R',
44.:'corpus callosum',
45.:'Lateral ventricle excluding temporal horn R',
46.:'Lateral ventricle excluding temporal horn L',
47.:'Lateral ventricle temporal horn R',
48.:'Lateral ventricle temporal horn L',
49.:'Third ventricle',
50.:'FL precentral gyrus L',
51.:'FL precentral gyrus R',
52.:'FL straight gyrus L',
53.:'FL straight gyrus R',
54.:'FL anterior orbital gyrus L',
55.:'FL anterior orbital gyrus R',
56.:'FL inferior frontal gyrus L',
57.:'FL inferior frontal gyrus R',
58.:'FL superior frontal gyrus L',
59.:'FL superior frontal gyrus R',
60.:'PL postcentral gyrus L',
61.:'PL postcentral gyrus R',
62.:'PL superior parietal gyrus L',
63.:'PL superior parietal gyrus R',
64.:'OL lingual gyrus L',
65.:'OL lingual gyrus R',
66.:'OL cuneus L',
67.:'OL cuneus R',
68.:'FL medial orbital gyrus L',
69.:'FL medial orbital gyrus R',
70.:'FL lateral orbital gyrus L',
71.:'FL lateral orbital gyrus R',
72.:'FL posterior orbital gyrus L',
73.:'FL posterior orbital gyrus R',
74.:'substantia nigra L',
75.:'substantia nigra R',
76.:'FL subgenual frontal cortex L',
77.:'FL subgenual frontal cortex R',
78.:'FL subcallosal area L',
79.:'FL subcallosal area R',
80.:'FL pre-subgenual frontal cortex L',
81.:'FL pre-subgenual frontal cortex R',
82.:'TL superior temporal gyrus anterior part L',
83.:'TL superior temporal gyrus anterior part R',
84.:'PL supramarginal gyrus L',
85.:'PL supramarginal gyrus R',
86.:'insula anterior short gyrus L',
87.:'insula anterior short gyrus R',
88.:'insula middle short gyrus L',
89.:'insula middle short gyrus R',
90.:'insula posterior short gyrus L',
91.:'insula posterior short gyrus R',
92.:'insula anterior inferior cortex L',
93.:'insula anterior inferior cortex R',
94.:'insula anterior long gyrus L',
95.:'insula anterior long gyrus R',
}


colors = []
for l in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255,l))
for l in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,l))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)


def initialize_shap(model, background_cases, args):
    # class_name = {0: 'NACC', 1: 'ADNI', 2: 'OASIS', 3: 'AIBL'}
    if args.layer != 'input':

        e = shap.DeepExplainer((model, model[1].layer),background_cases.to(args.device))

        # e = shap.DeepExplainer((model.module, model.module.block1.conv) if args.cuda else (model, model.block1.conv),background_cases)
    else:
        e = shap.DeepExplainer(model if args.gpu_id is not None else model, background_cases.to(args.device))

    return e


def save_avg_map(base_dir, classes):
    for cls in classes:
        ic(os.path.join(base_dir, cls))
        ic(os.path.exists(os.path.join(base_dir, cls)))
        if not os.path.exists(os.path.join(base_dir, cls)):
            continue
        maps = []
        for f in os.listdir(os.path.join(base_dir,cls)):
            shap = np.load(os.path.join(base_dir,cls,f))
            print('shap: ', shap.shape)
            # break
            maps.append(shap)
        np_maps = np.asarray(maps)
        print(f'np_maps for class {cls}: ', np_maps.shape)
        
        avg_map = np.mean(np_maps, axis=0)
        print('Mean: ', avg_map.shape)
        np.save(os.path.join(base_dir,f'{cls}_avg.npy'), avg_map)
        print(os.path.exists(os.path.join(base_dir,f'{cls}_avg.npy')))


if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)
    s = print_args(args, [])

    print('data dir: ', args.data_dir)
    names = args.img_dataset[args.dataset]
    domain_label = names.index(args.cohort.split('_')[0])
    args.save_path = os.path.join(args.output, args.algorithm, os.path.basename(os.path.dirname(args.resume)), os.path.basename(args.data_dir)[:-4])
    print('save_path: ', args.save_path)
    # save_avg_map(args.save_path, ['NC', 'MCI', 'AD'])
    
    # exit()
    
    if 'train' in args.data_dir:
        transform = imgutil.image_train(args.dataset)
    else:
        transform = imgutil.image_test(args.dataset)

    dataset = ImageDataset(args.dataset, args.task, args.data_dir,
                                           args.cohort, domain_label, transform=transform, indices=None, test_envs=args.test_envs, filename=args.data_dir, num_classes=args.num_classes)
    

    
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=args.N_WORKERS, drop_last=False, shuffle=True)

    if 'train' in args.data_dir:
        bg_dataset = ImageDataset(args.dataset, args.task, args.data_dir.replace('train', 'val'),
                                           args.cohort, domain_label, transform=transform, indices=None, test_envs=args.test_envs, filename=args.data_dir.replace('train','val'))
        bg_dataloader = DataLoader(dataset=bg_dataset, batch_size=1, num_workers=args.N_WORKERS, drop_last=False, shuffle=True) 
        print(len(bg_dataloader))
        for data in bg_dataloader:
            background_cases = data[1]
            break
    else:
        background_cases = next(iter(dataloader))[1]
    
    print('background_cases: ', background_cases.size())

    print('loaded data.')
    
    args.weights, args.counts = dataset.get_sample_weights()
    print(type(args.counts))

    args.save_path = os.path.join(args.output, args.algorithm, os.path.basename(os.path.dirname(args.resume)), args.data_dir.split('/')[-1][:-4])
    ic(args.save_path)
    # exit()

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    args.device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id is not None else 'cpu')
    algorithm = algorithm_class(args).to(args.device)
    algorithm.eval()

    if args.resume:
        print('args.resume: ', args.resume)
        ckpt = torch.load(args.resume, map_location='cpu')
        state_dict = ckpt['model_dict']
        try:
            algorithm.load_state_dict(state_dict, strict=True)
        except:
            try:
                del state_dict['q']
                algorithm.load_state_dict(state_dict, strict=False)
            except:
                del state_dict['criterion.weight']
                algorithm.load_state_dict(state_dict, strict=False)
       
        
        print('loaded checkpoint at ', args.resume)
            
        try:
            best_epoch = ckpt['epoch']
            best_valid_acc = ckpt['accuracy']
            print(f'epoch: {best_epoch}, best acc: {best_valid_acc}')
        except:
            pass

    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)
    os.makedirs(args.save_path)

    mri_infolist = []
    final_count = 0
    mri_count = len(dataloader)
    correct, total = 0, 0
    class_name = {0: 'NC', 1: 'MCI', 2: 'AD'}
    label_list, preds_list = [], []
    for idx, data in enumerate(tqdm(dataloader)):
        if idx == mri_count:
            break
        device = list(algorithm.parameters())[0].device
        filenames = data[0]
        inputs = data[1].float().to(device)
        labels = data[2].long()
        print('inputs: ', inputs.size())

        feats, output = algorithm.predict(inputs, stage='get_features', attention=True)
        att_maps = feats[0].detach().cpu().numpy()
        ic(att_maps.shape, att_maps.min(), att_maps.max())
        ic(np.squeeze(att_maps)[2].min(), np.squeeze(att_maps)[2].max())
        probs = F.softmax(output, dim=1)
        pred_probs, pred_labels = probs.sort(dim=1, descending=True)
        print('pred_labels: ', pred_labels.size())
        with open(f'{args.save_path}/raw_scores.txt', 'a') as f:
            write_scores(f, probs, labels)

        prediction = pred_labels.squeeze(0)[0].item()
        label_list.extend(labels.cpu().tolist())
        preds_list.extend([p[0] for p in pred_labels.cpu().tolist()])
        ic(f"Predicted: {pred_labels}, true label: {labels[0]}")
        del output
        del probs
        del pred_probs
        del pred_labels
        # if prediction != labels[0]:
        #     print('Wrong prediction, skipping...')
        #     mri_count += 1
        #     continue

        parent_dir = os.path.join(args.save_path, class_name[prediction]) 
        if not os.path.exists(parent_dir): os.makedirs(parent_dir)
        ic(filenames[0], os.path.join(parent_dir, os.path.basename(filenames[0])))
        np.save(os.path.join(parent_dir, os.path.basename(filenames[0])), 
                np.squeeze(att_maps)[2])    # save AD att map for all cases
        
        final_count += 1

    label_list = np.array(label_list).flatten()
    preds_list = np.array(preds_list).flatten()
    print(label_list.shape, preds_list.shape)
    cm = confusion_matrix(label_list, preds_list, labels=[0, 1, 2])
    print(cm)
    
    ax = sns.heatmap(cm, annot=True, fmt='g')
    ax.set_title('Confusion matrix')
    ax.set_xticklabels(['NC', 'MCI', 'AD'])
    ax.set_yticklabels(['NC', 'MCI', 'AD'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    # plt.show()
    ic(os.path.join(args.save_path,f"confusion_matrix_{'_'.join(args.data_dir.split('/')[-3:])[:-4]}.png"))
    plt.savefig(os.path.join(args.save_path,f"confusion_matrix_{'_'.join(args.data_dir.split('/')[-3:])[:-4]}.png"), dpi=150)

    print('final count: ', final_count)
        