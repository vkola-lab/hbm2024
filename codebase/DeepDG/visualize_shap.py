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

import seaborn as sns
from icecream import ic
import gc

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


def save_avg_map(base_dir, classes, subfolders):
    for cls in classes:
        ic(os.path.join(base_dir, cls))
        ic(os.path.exists(os.path.join(base_dir, cls)))
        
        maps = []
        for subfolder in subfolders:
            if not os.path.exists(os.path.join(base_dir, subfolder, cls)):
                continue
            for f in os.listdir(os.path.join(base_dir, subfolder,cls)):
                shap = np.load(os.path.join(base_dir, subfolder, cls, f))
                ic(shap.shape, shap.min(), shap.max())
                shap = (shap - shap.min()) / (shap.max() - shap.min())
                ic(shap.shape, shap.min(), shap.max())
                # break
                maps.append(shap)
        
        np_maps = np.asarray(maps)
        print(f'np_maps for class {cls}: ', np_maps.shape)
        
        avg_map = np.mean(np_maps, axis=0)
        print('Mean: ', avg_map.shape)
        np.save(os.path.join(base_dir,f'{cls}_avg.npy'), avg_map)
        print(os.path.exists(os.path.join(base_dir,f'{cls}_avg.npy')))


def plot_shap_stacks(subject, shap_values, filename):
    subject_mri = np.squeeze(subject.numpy())
    shap_copy = np.squeeze(shap_values.copy())
#     shap_copy = shap_values.copy()
#     print('shap_copy: ', len(shap_copy))
    shap_lst = []
    shap_lst.append(shap_copy)
#     print('shap_lst: ', len(shap_lst))
    # slice count
    image_slice = 0
    
    # for each 2D slice, plot shap explanation
    while(image_slice < 182):
        arr = []
        m_arr = []
        # make a slice of shap value data
        for i in range(len(shap_lst)):
            print(f'shap_lst[{i}]: {shap_lst[i].shape}, {shap_lst[i][:, :, image_slice].shape}')
            copy_arr = shap_lst[i][:, :, image_slice].transpose((1, 0))[::-1, :]
            copy_arr = np.expand_dims(copy_arr, axis=0)
            arr.append(copy_arr)
            
            m_arr.append(arr)
        
        final = []
        for each in arr:
            each = np.expand_dims(each, axis=3)
            final.append(each)
            
        # make a slice of original data
        to_explain_img = subject_mri[:, :, image_slice].transpose((1,0))[::-1, :]
        to_explain_copy = np.expand_dims(to_explain_img, axis=2)
        test_np_copy = []
        test_np_copy.append(to_explain_copy)
        test_np_copy = np.asarray(test_np_copy)
        # make a subplot of shap explanation on the original data
        # implementation from https://github.com/slundberg/shap/blob/master/shap/plots/_image.py
        x = test_np_copy
        figure = plt.figure()
        
        x_curr = x[0].copy()
        x_curr = x_curr.reshape(x_curr.shape[:2])
        x_curr /= 255.
        x_curr_gray = x_curr
        x_curr_disp = x_curr
        abs_vals = np.stack([np.abs(final[i].sum(-1)) for i in range(len(final))], 0).flatten()
        sv = final[0][0] if len(final[0][0].shape) == 2 else final[0][0].sum(-1)
        orig_fig = plt.imshow(x_curr_disp, cmap=plt.get_cmap('gray'), extent=(-1, sv.shape[1], sv.shape[0], -1));
        max_val = np.nanpercentile(abs_vals, 99)
        # manually adjust the scaling
        shap_map = plt.imshow(sv, cmap=red_transparent_blue, vmin=-max_val, vmax=max_val);
        directory = '3D_SHAP_images/' + filename
        os.makedirs(directory, exist_ok=True)
        plt.axis('off');
        fig_path = f'3D_SHAP_images/{filename}/slice_{str(image_slice)}.png'
        figure.savefig(fig_path,dpi=150);
        plt.close()
        image_slice = image_slice + 1
         
    return None
    
_cache_ = {
  'memory_allocated': 0,
  'max_memory_allocated': 0,
  'memory_reserved': 0,
  'max_memory_reserved': 0,
}

def _get_memory_info(info_name, device, unit):
    tab = '\t'
    if info_name == 'memory_allocated':
        current_value = torch.cuda.memory.memory_allocated(device)
    elif info_name == 'max_memory_allocated':
        current_value = torch.cuda.memory.max_memory_allocated(device)
    elif info_name == 'memory_reserved':
        tab = '\t\t'
        current_value = torch.cuda.memory.memory_reserved(device)
    elif info_name == 'max_memory_reserved':
        current_value = torch.cuda.memory.max_memory_reserved(device)
    else:
        raise ValueError()

    divisor = 1
    if unit.lower() == 'kb':
        divisor = 1024
    elif unit.lower() == 'mb':
        divisor = 1024*1024
    elif unit.lower() == 'gb':
        divisor = 1024*1024*1024
    else:
        raise ValueError()

    diff_value = current_value - _cache_[info_name]
    _cache_[info_name] = current_value

    return f"{info_name}: \t {current_value} ({current_value/divisor:.3f} {unit.upper()})" \
            f"\t diff_{info_name}: {diff_value} ({diff_value/divisor:.3f} {unit.upper()})"


@profile
def run_shap(algorithm, explainer, dataloader, args):
    mri_infolist = []
    final_count = 0
    mri_count = len(dataloader)
    correct, total = 0, 0
    class_name = {0: 'NC', 1: 'MCI', 2: 'AD'}
    label_list, preds_list = [], []
    torch.cuda.empty_cache()
    with torch.cuda.amp.autocast():
        for idx, data in enumerate(tqdm(dataloader)):
            if idx == mri_count:
                break
            device = list(algorithm.parameters())[0].device
            filenames = data[0]
            inputs = data[1].half().to(device)
            labels = data[2].long().half()
            print('inputs: ', inputs.size())

            output = algorithm.predict(inputs)
            probs = F.softmax(output, dim=1)
            pred_probs, pred_labels = probs.sort(dim=1, descending=True)
            print('pred_labels: ', pred_labels.size())
            prediction = pred_labels.squeeze(0)[0].item()
            label_list.extend(labels.detach().cpu().tolist())
            preds_list.extend([p[0] for p in pred_labels.detach().cpu().tolist()])
            ic(f"Predicted: {pred_labels}, true label: {labels[0]}")
            # ic(torch.cuda.memory_allocated(device) / 1024. / 1024.)
            output = output.detach().cpu()
            probs = probs.detach().cpu()
            pred_probs = pred_probs.detach().cpu()
            pred_labels = pred_labels.detach().cpu()
            del output
            del probs
            del pred_probs
            del pred_labels
            # gc.collect()
            # print(_get_memory_info('memory_allocated', device, 'gb'))
            # print(_get_memory_info('max_memory_allocated', device, 'gb'))
            # print(_get_memory_info('memory_reserved', device, 'gb'))
            if prediction != labels[0]:
                print('Wrong prediction, skipping...')
                mri_count += 1
                continue
            
            
            shap_values = explainer.shap_values(inputs)
            print('shap values: ', len(shap_values), shap_values[0].shape)

            parent_dir = os.path.join(args.save_path, class_name[prediction]) 
            if not os.path.exists(parent_dir): os.makedirs(parent_dir)
            ic(filenames[0], os.path.join(parent_dir, os.path.basename(filenames[0])))
            np.save(os.path.join(parent_dir, os.path.basename(filenames[0])), 
                    np.squeeze(shap_values[prediction]))
            
            final_count += 1
            # shap_values = [s.detach().cpu() for s in shap_values]
            inputs = inputs.detach().cpu()
            del shap_values
            del inputs
            del labels
            del prediction
            gc.collect()
            torch.cuda.empty_cache()
            # print(_get_memory_info('memory_allocated', device, 'gb'))
            # print(_get_memory_info('max_memory_allocated', device, 'gb'))
            # print(_get_memory_info('memory_reserved', device, 'gb'))


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
    ic(os.path.join(args.save_path,f"confusion_matrix_shap_{'_'.join(args.data_dir.split('/')[-3:])[:-4]}.png"))
    plt.savefig(os.path.join(args.save_path,f"confusion_matrix_shap_{'_'.join(args.data_dir.split('/')[-3:])[:-4]}.png"), dpi=150)

    print('final count: ', final_count)


if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)
    s = print_args(args, [])

    print('data dir: ', args.data_dir)
    names = args.img_dataset[args.dataset]
    domain_label = names.index(args.cohort.split('_')[0])
    args.save_path = os.path.join(args.output, args.algorithm, os.path.basename(os.path.dirname(args.resume)), args.cohort, args.layer, '_'.join([args.data_dir.split('/')[-3], str(args.fold), args.data_dir.split('/')[-1][:-4]]))
    # args.save_path = os.path.join(args.output, args.algorithm, os.path.basename(os.path.dirname(args.resume)), args.cohort, args.layer)
    # ic(args.save_path)
    # save_avg_map(args.save_path, ['NC', 'MCI', 'AD'], [f'NACC_NC_MCI_AD_folds_{args.fold}_train', f'ADNI_NC_MCI_AD_folds_{args.fold}_train'])
    
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
                                           args.cohort, domain_label, transform=transform, indices=None, test_envs=args.test_envs, filename=args.data_dir.replace('train','val'), num_classes=args.num_classes)
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

    # args.save_path = os.path.join(args.output, args.algorithm, os.path.basename(os.path.dirname(args.resume)), args.cohort, args.layer)
    # ic(args.save_path)
    # save_avg_map(args.save_path, ['NC', 'MCI', 'AD'], [f'NACC_NC_MCI_AD_folds_{args.fold}_train',f'ADNI_NC_MCI_AD_folds_{args.fold}_train'])
    
    
    args.save_path = os.path.join(args.output, args.algorithm, os.path.basename(os.path.dirname(args.resume)), args.cohort, args.layer, '_'.join([args.data_dir.split('/')[-3], str(args.fold), args.data_dir.split('/')[-1][:-4]]))
    ic(args.save_path)
    # exit()
    # args.gpu_id = None
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    args.device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id is not None else 'cpu')
    ic(args.gpu_id, args.device)
    # exit()
    print(_get_memory_info('memory_allocated', args.device, 'gb'))
    print(_get_memory_info('max_memory_allocated', args.device, 'gb'))
    print(_get_memory_info('memory_reserved', args.device, 'gb'))
    algorithm = algorithm_class(args).to(args.device)
    algorithm.eval()
    print(_get_memory_info('memory_allocated', args.device, 'gb'))
    print(_get_memory_info('max_memory_allocated', args.device, 'gb'))
    print(_get_memory_info('memory_reserved', args.device, 'gb'))
    # exit()
    if args.resume:
        print('args.resume: ', args.resume)
        ckpt = torch.load(args.resume, map_location='cpu')
        state_dict = ckpt['model_dict']
        ic(state_dict.keys())
        ic(algorithm.state_dict().keys())
        # try:
        #     del state_dict['criterion.weight']
        # except:
        #     pass
        algorithm.load_state_dict(state_dict, strict=True)
       
        
        print('loaded checkpoint at ', args.resume)
            
        try:
            best_epoch = ckpt['epoch']
            best_valid_acc = ckpt['accuracy']
            print(f'epoch: {best_epoch}, best acc: {best_valid_acc}')
        except:
            pass
    ic('before initializing shap')
    print(_get_memory_info('memory_allocated', args.device, 'gb'))
    explainer = initialize_shap(algorithm.network.half(), background_cases.half(), args)
    ic('after initializing shap')
    print(_get_memory_info('memory_allocated', args.device, 'gb'))

    
    if os.path.exists(args.save_path):
        shutil.rmtree(args.save_path)

    
    run_shap(algorithm, explainer, dataloader, args)        
    # save_avg_map(args.save_path, ['NC', 'MCI', 'AD'])
    args.save_path = os.path.join(args.output, args.algorithm, os.path.basename(os.path.dirname(args.resume)), args.cohort, args.layer)
    ic(args.save_path)
    # save_avg_map(args.save_path, ['NC', 'MCI', 'AD'], [f'NACC_NC_MCI_AD_folds_{args.fold}_train',f'ADNI_NC_MCI_AD_folds_{args.fold}_train'])
    if args.cohort == 'NACC_NC_MCI_AD':
        save_avg_map(args.save_path, ['NC', 'MCI', 'AD'], [f'NACC_NC_MCI_AD_folds_{args.fold}_train'])
    
