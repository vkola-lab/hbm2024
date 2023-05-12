import numpy as np
import csv
import pandas as pd
import os
from argparse import ArgumentParser
from utils.util import print_environ
import nibabel as nib
from scipy.ndimage import zoom
from icecream import ic
from tqdm import tqdm

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

prefixes = ['CG_1', 'FL_mfg_7', 'FL_pg_10', 'TL_stg_32', 'PL_ag_20', 'Amygdala_24',
            'TL_hippocampus_28', 'TL_parahippocampal_30', 'c_37', 'bs_35', 'sn_48', 'th_49',
            'pal_46', 'na_45X', 'cn_36C', 'OL_17_18_19OL']

prefix_idx = {'CG_1':[24],
              'FL_mfg_7': [28],
              'FL_pg_10': [50],
              'TL_stg_32': [82],
              'PL_ag_20': [32],
              'Amygdala_24': [4],
              'TL_hippocampus_28': [2],
              'TL_parahippocampal_30': [10],
              'c_37': [18],
              'bs_35': [19],
              'sn_48': [74],
              'th_49': [40],
              'pal_46': [42],
              'na_45X': [36],
              'cn_36C': [34],
              'OL_17_18_19OL': [64, 66, 22]}

def get_args():
    parser = ArgumentParser(description='SHAP Region Scores')
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--shap_dir', type=str)
    parser.add_argument('--seg_dir', type=str)
    parser.add_argument('--output_dir', type=str, help='CSV file output dir')
    parser.add_argument('--cohort', type=str, help='Name of cohort to which the shap maps belong.')
    parser.add_argument('--postfix', type=str, default='', help="postfix to append to the CSV filename")
    parser.add_argument('--layer', type=str, help="Layer at which SHAP maps were computed")
    args = parser.parse_args()
    print_environ()
    return args

def upsample(heat, target_shape=(182, 218, 182), margin=0):
    background = np.zeros(target_shape)
    x, y, z = heat.shape
    X, Y, Z = target_shape
    X, Y, Z = X - 2 * margin, Y - 2 * margin, Z - 2 * margin
    data = zoom(heat, (float(X)/x, float(Y)/y, float(Z)/z), mode='nearest')
    background[margin:margin+data.shape[0], margin:margin+data.shape[1], margin:margin+data.shape[2]] = data
    return background

def pool_by_region(heatmap, seg, normalize=True):
    ic(heatmap.min(), heatmap.max())
    if normalize:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + np.finfo(float).eps)
    ic(heatmap.min(), heatmap.max())

    pool = [[] for _ in range(96)]
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            for k in range(seg.shape[2]):
                pool[int(seg[i, j, k])].append(heatmap[i, j, k])
    return pool

def get_region_scores(pool):
    ans = []
    for i in range(1, 96):
        ic(f'region {i}: {len(pool[i])} pixels')
        val = sum(pool[i])/len(pool[i])
        # val = (val - val.min()) / (val.max() - val.min() + np.finfo(float).eps)
        ic(val)
        ans.append(val)
    return np.array(ans)

def get_ABC_scores(pool):
    A_regions = ['FL_mfg_7', 'PL_ag_20', 'Amygdala_24', 'TL_parahippocampal_30', 'TL_stg_32']
    B_regions = ['TL_hippocampus_28', 'bs_35', 'Amygdala_24', 'TL_parahippocampal_30', 'sn_48', 'pal_46']
    C_regions = ['FL_mfg_7', 'PL_ag_20', 'TL_hippocampus_28', 'Amygdala_24', 'TL_parahippocampal_30', 'TL_stg_32']

    ans = []
    for regions in [A_regions, B_regions, C_regions]:
        vals = []
        for region in regions:
            vals.extend([p for i in prefix_idx[region] for p in pool[i]])
        ic(len(vals), type(vals[0]))
        ans.append(np.mean(vals))

    return np.array(ans)

if __name__ == '__main__':
    args = get_args()

    assert os.path.exists(args.shap_dir), 'Generate heatmaps first!!'
    score = 'attention'
    gt = None
    csv_fname = f'/home/dlteif/neuropath/{args.cohort}_{score}_scores_{args.layer}_layer_{args.algorithm}{args.postfix}.csv'
    f = open(csv_fname, 'w')
    # fieldnames = ['filename'] + [str(i) for i in range(1, 96)]  + ['avg_score', 'pos_avg_score', 'neg_avg_score'] # 95 regions
    fieldnames = ['filename'] + [str(i) for i in range(1, 96)]  + ['avg_score', 'A_score', 'B_score', 'C_score'] # 95 regions
    df = pd.read_csv('~/neuropath/FHS_NC_MCI_AD_np.csv')
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    classes = {'NC': 0, 'MCI': 1, 'AD': 2}
    for root, dirs, files in tqdm(os.walk(args.shap_dir)):
        path = root.split(os.sep)
        ic(root, dirs, files)
        print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            if '_avg' in file or '.npy' not in file:
                continue
            if gt is not None and df[df['filename'] == file][gt].values[0] == 0:
                continue
            ic(len(path) * '---', file)
            heatmap = np.load(os.path.join(root, file)).astype(np.float64)
            ic(heatmap.shape)
            if args.layer != "input":
                # heatmap = heatmap[classes[os.path.basename(root)],...]
                print('Before upsampling: ', heatmap.shape, heatmap.min(), heatmap.max())
                heatmap = upsample(heatmap)
                print('heatmap upsampled: ', heatmap.shape, heatmap.min(), heatmap.max())

            print('heatmap min,max: ', heatmap.min(), heatmap.max())
            seg_dir = '/data_1/NACC_ALL/seg' if 'mri' in file else args.seg_dir
            seg = nib.load(os.path.join(seg_dir, file.replace('.npy','.nii'))).get_data()
            print('heatmap: ', heatmap.shape, ', seg: ', seg.shape)
            pool = pool_by_region(heatmap, seg)
            regions = get_region_scores(pool)
            ABC_scores = get_ABC_scores(heatmap)
            print('regions: ', len(regions), regions)
            print('ABC scores: ', ABC_scores)
            case = {'filename': file}
            # if args.normalize:
            # heatmap = -1 + ((heatmap - heatmap.min())*2) / (heatmap.max() - heatmap.min() + np.finfo(float).eps)
            brain_shapvals = [v for region in pool for v in region]
            # pos_shap = heatmap[np.where(heatmap > 0)]
            # neg_shap = heatmap[np.where(heatmap < 0)]
            # pos_brain_shapvals = [v for region in pool for v in region if v > 0]
            # neg_brain_shapvals = [v for region in pool for v in region if v < 0]
            ic(len(brain_shapvals))
            for i in range(1, 96):
                case[str(i)] = regions[i-1]
            # case['avg_score'] = sum(brain_shapvals) / len(brain_shapvals)
            case['avg_score'] = np.mean(heatmap)
            for idx, score in enumerate(['A_score', 'B_score', 'C_score']):
                case[score] = ABC_scores[idx]
            # case['pos_avg_score'] = sum(pos_brain_shapvals) / len(brain_shapvals)
            # case['pos_avg_score'] = np.mean(pos_shap)
            # case['neg_avg_score'] = sum(neg_brain_shapvals) / len(brain_shapvals)
            # case['neg_avg_score'] = np.mean(neg_shap)
            
            # ic(case['avg_score'], case['pos_avg_score'], case['neg_avg_score'])
            writer.writerow(case)
    
    f.close()

            

