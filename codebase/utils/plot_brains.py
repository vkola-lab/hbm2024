################################################
#   A lot of this code is taken from Mike.
#   Thanks Mike!!
################################################

import sys
sys.path.append('/home/dlteif/mri-surv-dev/')
sys.path.append('/home/dlteif/mri-surv-dev/mri_surv/')
import pandas as pd
import re
import os
import numpy as np
import nibabel as nib
import abc
import colorcet as cc
import matplotlib.pyplot as plt
from matplotlib import gridspec, colors
import matplotlib.cm as cm
import seaborn as sns
import copy
import subprocess
from nilearn import plotting, datasets
from nilearn.image import resample_to_img, load_img
from typing import Dict, Tuple
from icecream import ic
from scipy.ndimage import zoom
from scipy.special import softmax
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from mri_surv.statistics.utilities \
    import CONFIG
from mri_surv.statistics.clustered_mlp_output_wrappers import load_any_long, \
    load_adni_clusters, load_adni_ad_clusters
from mri_surv.statistics.mlp_output_wrappers import load_roiname_to_roiid_map
from mri_surv.statistics.aggregate_utilities import cluster_average, \
    group_cortical_regions_by_subtype
from mri_surv.statistics.dataframe_validation import DataFrame, \
    ParcellationClusteredLongSchema, pa
from mri_surv.statistics.shap_viz.shap_plots import get_region_order

__all__ = [
    'load_brain_for_groups',
    'load_brain_for_groups_xyz_rank'
    ]

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

prefix_idx = {'CG_1':[24,25],
              'FL_mfg_7': [28,29],
              'FL_pg_10': [50, 51],
              'TL_stg_32': [82, 83],
              'PL_ag_20': [32, 33],
              'Amygdala_24': [3, 4],
              'TL_hippocampus_28': [1, 2],
              'TL_parahippocampal_30': [9, 10],
              'c_37': [17, 18],
              'bs_35': [19],
              'sn_48': [74, 75],
              'th_49': [40, 41],
              'pal_46': [42, 43],
              'na_45X': [36, 37],
              'cn_36C': [34, 35],
              'OL_17_18_19OL': [64, 65, 66, 67, 22, 23]
              }

class Brain(object):
    def __init__(self, props=None, rid='', train_or_test='ADNI'):
        if props is None:
            props = CONFIG
        self.props = props
        for key, value in self.props.items():
            setattr(self, key, value)
        """MICA -- you will want to plot the mask"""
        self.mask_path = os.path.join(
                self.image_directories[train_or_test]['atlas'],
                f'wneuromorphometrics_{rid}_mri.nii'
        )
        self.original_brain_path = os.path.join(
                self.image_directories[train_or_test]['basedir'],
                f'masked_brain_mri_{rid}.nii'
        )
        self.rid = rid
        self._load_mask() # loading mask here
        self._load_brain()
        self._recoded = False

    def _load_mask(self) -> None:
        try:
            data = nib.load(self.mask_path)
            self.mask = data.get_fdata()
            self.hdr = data.header
            self.affine = data.affine
            self.mask_img = data
        except FileNotFoundError as e:
            print(e)
            self.mask = np.nan
            self.hdr = np.nan
            self.affine = np.nan

    def _load_brain(self) -> None:
        try:
            data = nib.load(self.original_brain_path)
            self.original_brain = data.get_fdata()
            self.brain_img = data
        except FileNotFoundError as e:
            print(e)
            self.original_brain = np.array([])

    @abc.abstractmethod
    def recode_background(self):
        if not all([np.isnan(x) for x in self.mask.reshape(-1,1)]):
            mask = self.mask.copy()
            mask[mask == 1] = np.nan
            self.mask_img = nib.Nifti1Image(np.ma.masked_invalid(mask), self.affine)
            self._recoded = True

    def plot_brain(self, title='', cmap_nm='glasbey_dark', cut_coords=(range(
            -51,-1,20), range(-41,1,20), range(-21,21,20))):
        mask_img_path = './figures/figure_4'
        f_name = os.path.join(mask_img_path,
                              f'mri_parcellated_brain_{self.rid}.svg')
        os.makedirs(mask_img_path, exist_ok=True)
        if not self._recoded:
            self.recode_background()
        cmap = copy.copy(cc.cm[cmap_nm])
        cmap.set_bad('w', alpha=1)
        for idx, dim in enumerate(('x','y','z')):
            if idx == 0:
                title = f'RID {self.rid}'
            else:
                title = ''
            # please make background white or black, make regions
            # as distinct as possible, and can use 3x3 grid
            plotting.plot_img(self.mask_img,  # could replace with self.mask_img
                                display_mode=dim,
                                cut_coords=cut_coords[idx],
                                cmap = cmap,
                                colorbar=False,
                                annotate=False,
                                title=title,
                                axes=plt.subplot(3,1,idx+1),
                                )
        plt.savefig(f_name, dpi=300, transparent=True)
        plt.close()

class BrainSample(object):
    def __init__(self, props=None, brain_path='', title=''):
        if props is None:
            props = CONFIG
        self.props = props
        for key, value in self.props.items():
            setattr(self, key, value)
        self.title = title
        self.plot_img_path = './figures/test_figs'
        self.plot_img_prefix = os.path.join(self.plot_img_path,
                                f'mri_slice_{self.title}')
        self._brain_path = brain_path
        self._load_brain()

    def _load_brain(self) -> None:
        try:
            data = nib.load(self._brain_path)
            self.original_brain = data.get_fdata()
            self.brain_img = data
            self.affine = data.affine
            self.shape = data.header.get_data_shape()
            self.center = np.asarray(self.shape) // 2
            self.top_corner = nib.affines.apply_affine(
                    self.affine, self.shape[:-1])
            self.bottom_corner = nib.affines.apply_affine(
                    self.affine, [0, 0, 0])
            self.range = [[int(x),int(y)] for x,y in zip(self.bottom_corner,
                                               self.top_corner)]
        except FileNotFoundError as e:
            print(e)
            self.original_brain = np.array([])

    def plot_slices(self):
        os.makedirs(self.plot_img_path, exist_ok=True)
        for idx, dim in enumerate(('x','y','z')):
            for _slice in range(self.range[idx][0],self.range[idx][1],5):
                ax = plt.subplot(111)
                plotting.plot_anat(self.brain_img,
                                   axes=ax,
                                   display_mode=dim,
                                   cut_coords=[_slice])
                f_name = f'{self.plot_img_prefix}_{dim}_' \
                         f'{str(_slice).zfill(3)}.pdf'
                plt.savefig(f_name, dpi=300, transparent=True)
                plt.close()

    def concatenate_slices(self):
        fi = os.listdir(self.plot_img_path)
        fi = [os.path.join(self.plot_img_path, x) for x in fi]
        for idx, dim in enumerate(('x','y','z')):
            matching_fi = list(filter(lambda x: re.match(
                    f'^{self.plot_img_prefix}_{dim}_[0-9]+\.pdf',x), fi))
            matching_fi = ' '.join(matching_fi)
            cmd = f'pdftk {matching_fi} cat output ' \
                  f'{self.plot_img_prefix}_{dim}'
            subprocess.run(cmd, shell=True, capture_output=True)
            rm_cmd = f'rm {matching_fi}'
            subprocess.run(rm_cmd, shell=True)

# # from https://nipy.org/nibabel/coordinate_systems.html

class ShapBrain(Brain):
    def __init__(self, props=None, rid=None, shap_map=None, _bin=None,
                 threshold=0.001, name=''):
        if props is None:
            props = CONFIG
        if rid is not None:
            super().__init__(props=props, rid=rid, train_or_test='ADNI')
            self.path_config = None
        else:
            self.rid = name
            for prop, item in props.items():
                setattr(self, prop, item)
            self.mask_img = nib.load(
                    self.path_config['neuromorphometrics_mask']
            )
            self.mask = self.mask_img.get_fdata()
            self.hdr = self.mask_img.header
            self.affine = self.mask_img.affine
            self.original_brain = None
            self.brain_img = None
        self.shap_mask_img_path = os.path.join('.', 'figures',
                                                    'shap_glass_brain')
        self.shap_mask_path = os.path.join('.', 'metadata','data_processed'
                                                    'shap_masks')
        self.shap_map = shap_map
        self.bin = _bin
        self.threshold = threshold
        self._make_shap_mask()

    def _make_shap_mask(self) -> None:
        shap_map = self.shap_map
        mask = self.mask
        shap_mask = np.zeros_like(mask)
        for idx in shap_map.keys():
            if abs(shap_map[idx]) >= self.threshold:
                shap_mask[np.where(mask == idx)] = shap_map[idx]
        shap_mask[np.where(mask < 1)] = np.nan
        self.shap_mask = shap_mask
        self.shap_img = nib.Nifti1Image(shap_mask, self.affine,
                                             header=self.hdr)

    def make_shap_mask_select(self, region_idx) -> None:
        shap_map = self.shap_map
        mask = self.mask
        shap_mask = np.zeros_like(mask)
        for idx in region_idx:
            shap_mask[np.where(mask == idx)] = shap_map[idx]
        shap_mask[np.where(mask < 1)] = np.nan
        self.shap_mask = shap_mask
        self.shap_img = nib.Nifti1Image(shap_mask, self.affine,
                                             header=self.hdr)

    def save_brain(self) -> None:
        f_name = os.path.join(self.shap_mask_path,
                              f'shap_mask_{self.rid}_bin{self.bin}')
        os.makedirs(self.shap_mask_path, exist_ok=True)
        nib.save(self.shap_img, f_name + '.nii')

    def plot_brain(self, vmax=0.015, title=''):
        f_name = {}
        f_name['svg'] = os.path.join(self.shap_mask_img_path,
                              f'stat_map_{self.rid}_bin'
                              f'{self.bin}.svg')
        f_name['png'] = os.path.join(self.shap_mask_img_path,
                              f'stat_map_{self.rid}_bin'
                              f'{self.bin}.png')
        os.makedirs(self.shap_mask_img_path, exist_ok=True)
        for _, value in f_name.items():
            if self.brain_img is not None:
                brain_plt = plotting.plot_stat_map(self.shap_img,
                                display_mode='z',
                                cut_coords=range(-20,21,20),
                                bg_img=self.brain_img,
                                cmap = cc.cm.bmy,
                                colorbar=True,
                                annotate=False,
                                title=title
                                )
            else:
                brain_plt = plotting.plot_stat_map(self.shap_img,
                                display_mode='z',
                                cut_coords=range(-20,21,20),
                                cmap = cc.cm.bmy,
                                colorbar=True, vmax=vmax,
                                symmetric_cbar=True,
                                annotate=False,
                                title=title
                                )
            plt.savefig(value, dpi=300)
            plt.close()
    
    def plot_brain_new(self, vmax=0.015, title='', cut_coords=None):
        img_path = './figures'
        if cut_coords is None:
            cut_coords = []
            for c in ('x','y','z'):
                cut_coords.append(
                    plotting.find_cut_slices(
                        self.img,
                        direction=c,
                        n_cuts=1,
                        spacing='auto')
                    )
        f_name = os.path.join(img_path,
                              f'shap_cnn_brain_{self.rid}.png')
        os.makedirs(img_path, exist_ok=True)
        plotting.plot_stat_map(self.img,  # could replace with self.mask_img
                                bg_img=None,
                                cut_coords=cut_coords,
                                colorbar=True,
                                cmap=plt.cm.bwr,
                                annotate=False,
                                title=title,
                                draw_cross=False,
                                black_bg=True,
                                vmax=0.00025
                                )
        plt.savefig(f_name, dpi=300, transparent=True)
        plt.savefig(f_name[:-4] + '.svg', dpi=300, transparent=True)
        plt.close()
        return cut_coords

    def plot_brain_xyz(self, vmax=0.015, title='',
                       cut_coords=(range(-51,-1,20),
                                   range(-41,1,20),
                                   range(-21,21,20))):
        f_name = os.path.join(self.shap_mask_img_path,
                              f'xyz_{self.rid}.svg')
        os.makedirs(self.shap_mask_img_path, exist_ok=True)
        for idx, dim in enumerate(('x','y','z')):
            if idx == 1:
                colorbar = True
            else:
                colorbar = False
            plotting.plot_stat_map(self.shap_img,
                                display_mode=dim,
                                cut_coords=cut_coords[idx],
                                cmap = cc.cm.bmy,
                                colorbar=colorbar, vmax=vmax,
                                symmetric_cbar=True,
                                annotate=False,
                                title=title,
                                axes=plt.subplot(3,1,idx+1)
                                )
        plt.savefig(f_name, dpi=300)
        plt.close()

# class MriBrain(Brain):
#     def __init__(self, rid, cluster_idx=-1, dataset='ADNI'):
#         super().__init__(props=CONFIG, rid=rid, train_or_test=dataset)
#         self.mask_img_path = os.path.join('.','figures','mri_brain')
#         self.cluster_idx = cluster_idx

#     def plot_brain(self, vmax=0.015, title='',
#                    cut_coords=(
#                            range(-51,-1,20),
#                            range(-41,1,20),
#                            range(-21,21,20))):
#         f_name = os.path.join(self.mask_img_path,
#                               f'mri_cluster_{self.cluster_idx}_{self.rid}.pdf')
#         os.makedirs(self.mask_img_path, exist_ok=True)
#         for idx, dim in enumerate(('x','y','z')):
#             if idx == 0:
#                 title = f'Cluster {self.cluster_idx}, RID {self.rid}'
#             else:
#                 title = ''

#             plotting.plot_img(self.brain_img,
#                                 display_mode=dim,
#                                 cut_coords=cut_coords[idx],
#                                 cmap = cc.cm.gray,
#                                 colorbar=False,
#                                 annotate=False,
#                                 title=title,
#                                 axes=plt.subplot(3,1,idx+1)
#                                 )
#         plt.savefig(f_name, dpi=300)
#         plt.close()

# class ParcellatedBrain(Brain):
#     def __init__(self, rid, dataset='ADNI'):
#         super().__init__(props=CONFIG, rid=rid, train_or_test=dataset)
#         self.mask_img_path = os.path.join('.','figures','parcellated_brain')

#     def plot_brain(self, vmax=np.nan, title='', cut_coords=(-31, -31, -31)):  #[
#         f_name = os.path.join(self.mask_img_path,
#                               f'mri_parcellated_brain_{self.rid}.svg')
#         os.makedirs(self.mask_img_path, exist_ok=True)
#         for idx, dim in enumerate(('x','y','z')):
#             if idx == 0:
#                 title = f'RID {self.rid}'
#             else:
#                 title = ''
#             # please make background white or black, make regions
#             # as distinct as possible, and can use 3x3 grid
#             plotting.plot_img(self.mask_img,  # could replace with self.mask_img
#                                 display_mode=dim,
#                                 cut_coords=cut_coords[idx],
#                                 cmap = cc.cm.bmy,
#                                 colorbar=False,
#                                 annotate=False,
#                                 title=title,
#                                 axes=plt.subplot(3,1,idx+1)
#                                 )
#         plt.savefig(f_name, dpi=300)
#         plt.close()

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    return axes

def format_pathology_region_map(region_map_df: pd.DataFrame):
    region_map_df = region_map_df.copy()
    region_map_df.rename(columns={'Corresponding neuromorphometrics': 'idx'},
                         inplace=True)
    region_map_df.loc[:, 'idx'] = region_map_df.loc[:, 'idx'].apply(
            lambda x: list(map(int, x.split(','))))
    return region_map_df

def _generate_index_to_value_map(
        df: DataFrame[ParcellationClusteredLongSchema],
        value_set: str='Shap Value') -> pd.Series:
    region_average_over_group = cluster_average(df=df, value_set=value_set)
    name_to_id_map = load_roiname_to_roiid_map()
    region_average_over_group = region_average_over_group[['Region', 'Cluster Idx', value_set]]
    idx_to_value_map = region_average_over_group.groupby('Cluster Idx').apply(
        lambda x: _generate_neuromorph_dictionary(x, name_to_id_map, value_set)
    )
    return idx_to_value_map

# def _generate_neuromorph_dictionary(region_average_over_group: pd.DataFrame,
#                                     name_to_id_map: Dict[str,list], value_set:
#         str):
#     schema = pa.DataFrameSchema({
#         'Region' : pa.Column(str),
#         'Cluster Idx': pa.Column(str),
#         'Shap Value': pa.Column(float, required=False),
#         'Gray Matter Vol': pa.Column(float, required=False)
#     })
#     region_average_over_group = schema.validate(region_average_over_group)
#     region_values = region_average_over_group.set_index('Region', drop=True)[value_set]
#     region_values = region_values.squeeze().to_dict()
#     idx_to_value_dict = {}   # for each region label integer, assign value for that region in group
#     for region, region_indices in name_to_id_map.items():  # iterate over all regions and mask_idx for the neuromorphometric atlas
#         for region_idx in region_indices:
#             if region in region_values.keys():  # over all region labels in the list of mask labels   
#                 idx_to_value_dict[region_idx] = region_values[region]
#             else:
#                 idx_to_value_dict[region_idx] = np.nan
#     return idx_to_value_dict

# def load_brain_for_groups(dataset: str='NACC',
#                             value_set: str='Shap Value',
#                             threshold: float=2.5,
#                             _plt: bool=True,
#                             vmax: float=10.0,
#                             xyz: bool=False
#                           ):
#     df = load_any_long(dataset).copy()
#     idx_to_value_map = _generate_index_to_value_map(df, value_set)
#     shap_brains = []
#     value_set = value_set.replace(' ', '').lower()
#     for group, shap_map in idx_to_value_map.items():
#         if group == -1:
#             group = 'unclustered'
#         fname=f'{value_set}_for_cluster' \
#                 f'_{group}_{dataset}'
#         shap_brains.append(ShapBrain(
#                 props=CONFIG,
#                 rid=None,
#                 shap_map=shap_map,
#                 _bin='all',
#                 threshold=threshold,
#                 name=fname,
#                 )
#         )
#     if _plt:
#         for brain in shap_brains:
#             if xyz:
#                 brain.plot_brain_xyz(vmax=vmax)
#             else:
#                 brain.plot_brain(vmax=vmax)
#     return shap_brains

def load_brain_for_groups_xyz_rank(dataset='NACC',
                            value_set='Shap Value',
                            threshold=2.5,
                            _plt=True,
                            vmax=10, top_or_bottom='top'):
    if top_or_bottom not in ('top','bottom'):
        raise NotImplementedError
    df = load_any_long(dataset).copy()
    idx_to_value_map = _generate_index_to_value_map(df, value_set)
    top_and_bottom_regions = {}
    for cluster_idx, cluster_df in df.groupby('Cluster Idx'):
        top_and_bottom_regions[cluster_idx] = \
            get_region_order(cluster_df, by='corr')
    name_to_id_map = load_roiname_to_roiid_map()
    shap_brains = []
    value_set = value_set.replace(' ', '').lower()
    for cluster, shap_map in idx_to_value_map.items():
        regions = top_and_bottom_regions[cluster][top_or_bottom]
        idx = []
        [idx.append(_id) for region in regions for _id in name_to_id_map[region]]
        if cluster == -1:
            group = 'unclustered'
        fname=f'{value_set}_for_cluster' \
                f'_{cluster}_{dataset}_{top_or_bottom}'
        sb = ShapBrain(
                        props=CONFIG,
                        rid=None,
                        shap_map=shap_map,
                        _bin='all',
                        threshold=threshold,
                        name=fname,
                    )
        sb.make_shap_mask_select(idx)
        shap_brains.append(sb)
        if _plt:
            sb.plot_brain_xyz(vmax=vmax)
    return shap_brains

def plot_all_cortical_regions_by_subtype():
    df_adni = plot_cortical_regions_by_subtype('ADNI')
    df_adni['Time'] = 'MCI'
    df_adniad = plot_cortical_regions_by_subtype('ADNI_AD')
    df_adniad['Time'] = 'AD'
    df_combined = pd.concat([df_adni, df_adniad], axis=0)
    df_combined.to_csv('./results/cortical_regions_by_subtype_partial.csv', index=False)
    return df_combined

def plot_cortical_regions_by_subtype(dataset: str='ADNI'):
    plt.style.use('./statistics/styles/style.mplstyle')
    tbl = group_cortical_regions_by_subtype(dataset=dataset)
    tbl_all = tbl.copy()
    tbl = tbl.groupby(['Cortical Region','Subtype']).agg(np.mean)
    tbl.reset_index(inplace=True)
    tbl['Subtype'] = tbl['Subtype'].replace({'0': 'H', '1': 'IH', '2': 'IL', '3': 'L'})
    tbl = tbl.pivot('Cortical Region', 'Subtype', 'ZS Gray Matter Volume')
    sns.lineplot(
        data=tbl,
        ci=None,
        dashes=True,
        palette='flare',
        lw=5
    )
    plt.xticks(rotation=45, horizontalalignment='right')
    ax = plt.gca()
    ax.set_ylim([-1.5, 1.5])
    plt.savefig(f'figures/cortical_regions_by_subtype_{dataset}.svg')
    plt.savefig(f'figures/cortical_regions_by_subtype_{dataset}.png', bbox_inches='tight')
    plt.close()
    return tbl_all

# def generate_shap_brains() -> None:
#     load_brain_for_groups(_plt=True,
#                              dataset='NACC', value_set='Shap Value',
#                              threshold=0.001, vmax=0.005)
#     load_brain_for_groups(_plt=True,
#                              dataset='NACC', value_set='Gray Matter Vol',
#                              threshold=0.3, vmax=1.5)


# def generate_mri_brains_by_cluster() -> None:
#     adni_idx = load_adni_clusters()
#     for rid in adni_idx.index:
#         brain = MriBrain(rid, cluster_idx=adni_idx.loc[rid, 'Cluster Idx'], dataset='ADNI')
#         brain.plot_brain()
#     adni_ad_idx = load_adni_ad_clusters()
#     for rid in adni_ad_idx.index:
#         brain = MriBrain(rid, cluster_idx=adni_idx.loc[rid, 'Cluster Idx'], dataset='ADNI_AD')
#         brain.plot_brain()

# def generate_gmv_xyz() -> None:
#     load_brain_for_groups(_plt=True,
#                              dataset='NACC', value_set='Gray Matter Vol',
#                              threshold=0.3, vmax=1.5, xyz=True)
#     load_brain_for_groups(_plt=True,
#                              dataset='ADNI', value_set='Gray Matter Vol',
#                              threshold=0.3, vmax=1.5, xyz=True)
#     load_brain_for_groups(_plt=True,
#                             dataset='ADNI_AD', value_set='Gray Matter Vol',
#                             threshold=0.3, vmax=1.5, xyz=True)

# def generate_shap_values_xyz() -> None:
#     load_brain_for_groups(_plt=True,
#                             dataset='ADNI_AD', value_set='Gray Matter Vol',
#                             threshold=0.3, vmax=1.5,xyz=True)
#     load_brain_for_groups(_plt=True,
#                             dataset='ADNI', value_set='Gray Matter Vol',
#                             threshold=0.3, vmax=1.5, xyz=True)

# def generate_shap_values_by_cluster_ranked() -> None:
#     load_brain_for_groups_xyz_rank(_plt=True,
#                              dataset='NACC', value_set='Shap Value',
#                              threshold=0.001, vmax=0.005, top_or_bottom='top')
#     load_brain_for_groups_xyz_rank(_plt=True,
#                              dataset='NACC', value_set='Shap Value',
#                              threshold=0.001, vmax=0.005, top_or_bottom='bottom')


def plot_shap_priors(fpath):
    classes = {'NC': 0, 'MCI': 1, 'AD': 2}
    masks = []
    for cls in classes:
        mask_fpath = os.path.join(fpath, f'{cls}_avg1.npy')
        mask = np.load(mask_fpath)
        ic(mask.shape)
        mask = mask[classes[cls],:,:,:].astype(np.float32)
        mask = mask.astype(np.float32)
        ic(mask.shape, mask.min(), mask.max())
        masks.append(mask)

        # data = nib.load(mri_fpath)
        # affine = data.affine
        # ic(affine)

        # if not all([np.isnan(x) for x in mask.reshape(-1,1)]):
        #     mask_copy = mask.copy()
        #     mask[mask == 1] = np.nan
        #     mask_img = nib.Nifti1Image(np.ma.masked_invalid(mask), affine)

    mni = datasets.load_mni152_template()
    ic(mni.get_fdata().shape)
    min = np.min([mask.min() for mask in masks])
    max = np.max([mask.max() for mask in masks])
    ic(min, max)

    for mask, cls in zip(masks, classes):
        
        mask = upsample(mask, target_shape=mni.get_fdata().shape)
        # mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255 
        mask = mask * 255
        mask[mask < 0] = 0

        ic(mask.shape, mask.min(), mask.max())
        
        mask_img = nib.Nifti1Image(mask, affine=mni.affine)

        ic(mask_img.get_fdata().shape)

        cut_coords=(range(
                -51,-1,20), range(-41,1,20), range(-21,21,20))
        
        cmap = copy.copy(cc.cm['glasbey_dark'])
        cmap.set_bad('w', alpha=1)
        
        # save_path = f'plots/ERM_unet3d_gap_attention_NACC_ADNI_NC_MCI_ADaug/{fname[:-4]}'
        
        plot_glass_brain(mask, mask_img, cls, fpath, threshold=1/2)
        

def plot_glass_brain(mask, mask_img, cls, save_path, prefix=None, mri_img=None, threshold=2/3, bg_img=None,cut_coords=None):
    # display = plotting.plot_glass_brain(None, 
    #                     display_mode='ortho',
    #                     annotate=True,
    #                     colorbar=True,
    #                     # title=cls,
    #                     # threshold=mask.max()/2,
    #                     )
    # nib.save(mask_img, os.path.join(save_path, f'{cls}_avg.nii')) 
    # display.title(f'{cls}', size=14, bgcolor='white', color='black', weight='bold')
    # display.add_contours(mask_img, filled=True, threshold=mask.max()*threshold,
    #                     colorbar=True,
    # )
    fig = plt.figure(figsize=(20,14), constrained_layout=False)
    plt.tight_layout()
    axes = []
    images = []
    for idx, dim in enumerate(['z','x','y']):
        if idx == 0:
            title = 'Ours' if 'XAIalign' in prefix else prefix.replace('_MDG', '')
        else:
            title=''
        # title = cls
        if idx == 1:
            colorbar = True
        else:
            colorbar = False
        ic([len(cut_coords[idx]) for idx in range(len(cut_coords))])
        ic(len(cut_coords))
        # for slice_idx, slice in enumerate(cut_coords[idx]):
        gs = gridspec.GridSpec(len(cut_coords[idx]), 1, hspace=0.15, wspace=0, figure=fig)
        axes.append(fig.add_subplot(gs[idx]))
        images.append(plotting.plot_stat_map(mask_img,
                                bg_img=bg_img,
                                display_mode=dim,
                                threshold=threshold,
                                cut_coords=cut_coords[idx],
                                # threshold=mask.max()*threshold,
                                # vmin=0,
                                # plot_abs=False,
                                cmap = 'jet',
                                colorbar=False,
                                annotate=False,
                                black_bg=False,
                                figure=fig,
                                # draw_cross=True,
                                axes=axes[-1],
                                ))
        images[-1].title(title, size=40, weight='bold', bgcolor='w', color='black')
    fig.subplots_adjust(right=0.95)
    if 'XAIalign' in prefix:
        last_axes = plt.gca()
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes('right', size='4%', pad=0)
        cax.set_visible(False)
        iax = inset_axes(cax, width='50%', height='300%', loc='center right', borderpad=0)

        mappable = cm.ScalarMappable(cmap='jet')
        mappable.set_clim(vmin=-1.0, vmax=1.0)
        cbar = fig.colorbar(mappable, cax=iax, orientation='vertical', ticklocation='left', ticks=[-1,-0.5,0,0.5,1], extend='both')
        cbar.ax.tick_params(labelsize=30)
        # cbar.set_label('Correlation coefficients')
        plt.sca(last_axes)
    plt.savefig(os.path.join(save_path, f'{prefix}_{cls}_glassbrain.pdf'), dpi=300, transparent=False, bbox_inches='tight', pad_inches=0)
    plt.clf()
    plt.close()
    ic(os.path.join(save_path, f'{prefix}_{cls}_glassbrain.pdf'))



class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def plot_image(mri_img, mask_img, cls, save_path, prefix=None, threshold=1/2):
    display = plotting.plot_anat(mri_img, 
                        display_mode='mosaic',
                        annotate=True,
                        black_bg=False,
                        colorbar=True,
                        # threshold=mask.max()/2,
                        )
    # display.title(f'Ground truth: {gt}, predicted: {pred}', size=12, bgcolor=None)
    display.add_contours(mask_img, threshold=mask_img.get_fdata().max() * threshold, vmin=0, alpha=0.3,
                        colorbar=True, filled=True,
    )
    # plotting.plot_glass_brain(mask_img,  # could replace with self.mask_img
    #                         display_mode='lyrz',
    #                         threshold=2*mask.max()/3,
    #                         vmin=0,
    #                         # plot_abs=False,
    #                         # cmap = ,
    #                         colorbar=True,
    #                         annotate=True,
    #                         black_bg=False,
    #                         title=title,
    #                             )
    plt.savefig(os.path.join(save_path, f'{prefix}_{cls}_glassbrain.png'), dpi=300, transparent=False)
    plt.close()

def aggregate_regions(shap_map, parcellation, region_dict):
    pass

def plot_corr_glass_brain(csv_fpath, save_path, prefix):
    df = pd.read_csv(csv_fpath)
    mni = datasets.load_mni152_template()
    mni = datasets.load_mni152_template()
    ic(mni.get_fdata().shape)
    ic(mni.get_fdata().min(), mni.get_fdata().max())
    seg_fpath = ('/home/dlteif/neuropath/FHS_NC_MCI_AD_np_seg/1-1359_20050419.nii')
    seg = nib.load(seg_fpath)
    seg_img = seg.get_fdata().astype(np.float32)
    ic(seg_img.shape, seg_img.min(), seg_img.max())
    
    cut_coords=(
        (-32,-7,29),
        (-54,-46,17),
        (-37,-22,5)
    )
    for stain in set(df['Stain'].values):
        ic(stain)
        mask = np.zeros_like(seg_img)
        ic(mask.shape, mask.min(), mask.max())
        for idx, row in df[df['Stain'] == stain].iterrows():
            ic(row['Region'], row['Correlation'])
            if row['Correlation'] == 100:
                continue
            if row['Region'] not in prefix_idx:
                continue
            ic(prefix_idx[row['Region']])
            ic([brain_regions[idx] for idx in prefix_idx[row['Region']]])
            for pref_idx in prefix_idx[row['Region']]:
                reg_pix = np.argwhere(seg_img == pref_idx)
                # ic(reg_pix)
                for x,y,z in reg_pix:
                    mask[x,y,z] = row['Correlation']
        # mask = upsample(mask, target_shape=mni.get_fdata().shape)
        # seg_img = upsample(seg_img, target_shape=mni.get_fdata().shape)
        
        # bg_pix = np.argwhere(seg_img == 0)
        # for x,y,z in bg_pix:
        #     seg_img[x,y,z] = 255

        seg = nib.Nifti1Image(seg_img, affine=mni.affine)
        ic(mask.shape, mask.min(), mask.max())
        # mask = -1 + ((mask - mask.min()) * 2) / (mask.max()- mask.min() + np.finfo(float).eps)
        # mask[mask <= np.finfo(float).eps] = 0.
        ic(mask.shape, mask.min(), mask.max())
        # bg_pix = np.argwhere(mask == -1)
        # for x,y,z in bg_pix:
        #     mask[x,y,z] = 0
        mask = mask * 255
        # bg_pix = np.argwhere(mask == 0)
        # for x,y,z in bg_pix:
        #     mask[x,y,z] = 255
        mask_img = nib.Nifti1Image(mask, affine=mni.affine)
        plot_glass_brain(mask, mask_img, stain, save_path, prefix=prefix, threshold=10, bg_img=seg, cut_coords=cut_coords)
        # plot_image(seg, mask_img, stain, save_path, prefix=prefix, threshold=0)



def upsample(heat, target_shape=(182, 218, 182), margin=0):
    background = np.zeros(target_shape)
    x, y, z = heat.shape
    X, Y, Z = target_shape
    X, Y, Z = X - 2 * margin, Y - 2 * margin, Z - 2 * margin
    data = zoom(heat, (float(X)/x, float(Y)/y, float(Z)/z), mode='nearest')
    background[margin:margin+data.shape[0], margin:margin+data.shape[1], margin:margin+data.shape[2]] = data
    return background

if __name__ == '__main__':
    # figs = os.listdir('./figures/shuffled_mris_nii')
    # for fig in figs:
    #     brain = BrainSample(brain_path='./figures/shuffled_mris_nii/' + fig,
    #                         title=fig[:-4])
    #     brain.plot_slices()
    #     brain.concatenate_slices(
    # Brain(rid='0746').plot_brain()
    # generate_shap_brains()
    # generate_gmv_xyz()
    # generate_shap_values_xyz()
    # generate_shap_values_by_cluster_ranked()
    # plot_all_cortical_regions_by_subtype()

    # plot_shap_priors('/data_1/dlteif/shap_maps/ERM/unet3d_gap_pretrained_attention_NACC_NC_MCI_ADaug_minmax_normalize_lr0.0008_0/NACC_NC_MCI_AD/classifier')
    # plot_shap_priors('/data_1/dlteif/shap_maps/9x11x9/ERM/unet3d_gap_attention_NACC_ADNI_NC_MCI_ADaug_minmax_normalize_lr0.0008_0/NACC_ADNI_NC_MCI_AD/classifier')
    method = 'XAIalign_MDG'
    plot_corr_glass_brain(f'~/neuropath/1095days_FHS_attention_classifier_{method}_correlation.csv', '/home/dlteif/neuropath', method)
    exit()
    base_dir = '/projectnb/ivc-ml/dlteif/transferlearning/codebase/DeepDG/output_dir/attention_maps/9x11x9/ERM/unet3d_gap_attention_NACC_ADNI_NC_MCI_ADaug_minmax_normalize_lr0.0008_0/FHS_NC_MCI_AD_np/MCI'
    fnames = [f for f in os.listdir(base_dir) if '.npy' in f]
    ic(fnames)
    for fname in fnames:
        mri_fpath = f'/projectnb/ivc-ml/dlteif/ayan_datasets/FHS/npy/{fname[:-4]}.nii'
        cls='AD'
        classes = {'NC': 0, 'MCI': 1, 'AD': 2}
        mask_fpath = f'../DeepDG/output_dir/shap_maps/9x11x9/ERM/attcnn_gap_attention_NACC_ADNI_NC_MCI_AD_minmax_normalize_lr0.0008_0/NACC_ADNI_NC_MCI_AD/classifier/NACC_NC_MCI_AD_folds_0_train/{cls}_avg.npy'
        # mask_fpath = f'{base_dir}/{fname}'
        data = nib.load(mri_fpath)
        mri = data.get_fdata()
        hdr = data.header
        print(hdr)
        affine = data.affine
        ic(affine)
        mri_img = data
        df = pd.read_csv('/projectnb/ivc-ml/dlteif/ayan_datasets/neuropath/FHS_NC_MCI_AD_np.csv')
        ic(mri_fpath.split('/')[-1][:-4])
        row = df[df['filename'] == mri_fpath.split('/')[-1][:-4] + '.npy']
        ic(row)
        gt = 'NC' if int(row['NC']) == 1 else 'MCI' if int(row['MCI']) == 1 else 'AD'
        pred = os.path.basename(os.path.dirname(mask_fpath))
        ic(gt, pred)
        mask = np.load(mask_fpath)
        ic(mask.shape)
        mask = mask[classes[cls],:,:,:].astype(np.float32)
        # mask = mask.astype(np.float32)
        ic(mask.shape, mask.min(), mask.max())

        # data = nib.load(mri_fpath)
        # affine = data.affine
        # ic(affine)

        # if not all([np.isnan(x) for x in mask.reshape(-1,1)]):
        #     mask_copy = mask.copy()
        #     mask[mask == 1] = np.nan
        #     mask_img = nib.Nifti1Image(np.ma.masked_invalid(mask), affine)
        
        # seg_fpath = '/projectnb/ivc-ml/dlteif/ayan_datasets/NACC_ALL/seg/mri5422_1.2.840.113619.2.25.4.1415559.1367701865.832.nii'
        # seg_data = nib.load(seg_fpath)
        # seg = seg_data.get_fdata()
        # segaffine = seg_data.affine
        # ic(segaffine)

        ic(mask.shape)
        mni = datasets.load_mni152_template()
        ic(mni.get_fdata().shape)
        mask = upsample(mask, target_shape=mni.get_fdata().shape)
        # mask = mask * 100.0
        mask = (mask - mask.min()) / (mask.max() - mask.min()) * 255 
        ic(mask.shape, mask.min(), mask.max())
        

        # mask_img = load_img(mask_fpath)
        # mask_img = nib.Nifti1Image(mask, affine=data.affine)
        # mask_img = resample_to_img(mask_img, mni)
        mask_img = nib.Nifti1Image(mask, affine=mni.affine)
        mri_img = nib.Nifti1Image(upsample(mri_img.get_fdata(), target_shape=mni.get_fdata().shape), affine=mni.affine)

        ic(mask_img.get_fdata().shape)

        # shap_mask = np.zeros_like(mask)
        # for i in range(96):
        #     shap_mask[np.where(mask == i)] = 

        # cut_coords=(range(
        #         -51,-1,20), range(-41,1,20), range(-21,21,20))
        cut_coords=(range(
                -51,-1,20), range(-41,1,20), range(-21,21,20))
        
        cmap = copy.copy(cc.cm['glasbey_dark'])
        cmap.set_bad('w', alpha=1)
        
        # save_path = f'plots/ERM_unet3d_gap_attention_NACC_ADNI_NC_MCI_ADaug/{fname[:-4]}'
        save_path = f'plots/ERM_unet3d_gap_attention_NACC_ADNI_NC_MCI_ADaug/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # plot_image(None, mask_img, 'NC', 'NC', save_path)
        plot_glass_brain(mask, mask_img, cls, save_path)
        break