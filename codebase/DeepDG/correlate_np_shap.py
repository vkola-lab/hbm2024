import csv
import pandas as pd
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import collections
import math
import os
from collections import defaultdict
import copy

from icecream import ic

def boxplot(shap, neuropath, name, folder, method):
    x_scores = [0, 1, 2, 3]
    all_data = [[] for _ in range(len(x_scores))]
    for data, score in zip(shap, neuropath):
        all_data[x_scores.index(score)].append(data)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    ax.boxplot(all_data,
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=x_scores,  # will be used to label x-ticks
               zorder=0,
               showfliers=False)
    for x, data in zip(x_scores, all_data):
        ax.plot(np.random.normal(x+1, 0.1, size=len(data)), data, 'r.',
                alpha=0.6, zorder=10, markersize=8)
    c, p = stats.spearmanr(shap, neuropath)
    ax.set_title('corr={:.4f}'.format(c))
    ax.set_xlabel(name)
    ax.set_ylabel('Shap value')
    plt.savefig(f'{folder}/figures/{method}/{name}.png', dpi=200, bbox_inches='tight')
    plt.close()


def get_correlation(col_name, indexes, thres, folder, method, cohort, score='shap', layer='input', missing=100):
    sub_shap = {}
    sub_np = {}
    
    np_df = pd.read_csv(f'{folder}/{cohort}_NC_MCI_AD_np.csv')
    for idx, row in np_df.iterrows():
        ic(col_name)
        if row[col_name] and not pd.isnull(row[col_name]) and row[col_name] not in ['9', '9.0', 'nan']:
            sub_np[row['id']] = float(row[col_name])
    # print('np df: ', np_df.columns)
    # print(np_df['filename'].values)
    
    shap_df = pd.read_csv('{}/{}_{}_scores_{}_layer_{}.csv'.format(folder, cohort, score, layer, method))
    for idx, row in shap_df.iterrows():
        # print(row['filename'])
        np_row = np_df[np_df['filename'] == row['filename']].iloc[0]
        # print('np row: ', np_row)
        if row['filename'] in list(np_df['filename']) and np_row['diff_days'] <= thres:
            id = np_row['id']
            sub_shap[id] = 0
            for idx in indexes:
                sub_shap[id] += float(row[str(idx)])
    
    vec1, vec2 = [], []
    for key in sub_shap:
        print('key: ', key)
        if key in sub_np:
            vec1.append(sub_shap[key] * 100.0)
            vec2.append(sub_np[key])
    # ic(vec1)
    ic(vec2)
    print('-------------------------------')
    if len(vec1) < 2 or len(set(vec2)) == 1:
        return missing, None, 0
    boxplot(vec1, vec2, col_name, folder, f'{thres}days_{method}')
    c, p = stats.spearmanr(vec1, vec2)
    # ordinal_model = OrderedModel(vec2, vec1, distr='logit')
    # results = ordinal_model.fit(method='bfgs')
    # ic(results)
    # pvals = results.pvalues[0]
    # ic(pvals)
    # CI = results.conf_int(alpha=0.05, cols=None)
    # ic(CI)  # [0.025, 0.975]
    # return results.llf, pvals, len(vec1)
    return c, p, len(vec1)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    print('----------def heatmap-----------')
    if not ax:
        ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    print('data: ', data)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75, orientation="horizontal", **cbar_kw)
    cbar.set_label(cbarlabel, rotation=0, fontsize=12, fontweight='black')
    # cbar.ax.set_ylabel(cbarlabel, rotation=0, va="center", fontsize=12, fontweight='black')
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im, cbar

def plot_corr_heatmap(regions, region_names, stains, corre, annot, pval, filename, folder):
    from matplotlib import rc, rcParams
    rc('axes', linewidth=1)
    rc('font', weight='bold')
    rcParams.update({'font.size': 7})
    hm = np.zeros((len(stains), len(regions)))
    an = np.zeros((len(stains), len(regions)))
    ic(hm, an)
    ic(corre)
    df_rows = []
    for i in range(len(regions)):
        ic(i)
        for j in range(len(stains)):
            ic(corre[regions[i]])
            corr = corre[regions[i]][stains[j]]
            if corr > 1 or corr < -1:
                corr = 100
            hm[j, i] = corr
            if stains[j] != 'Average Corr.':
                an[j, i] = annot[regions[i]][stains[j]]
                pvalue = pval[regions[i]][stains[j]]
                df_rows.append([regions[i], stains[j], corr, an[j, i], pvalue])
    df = pd.DataFrame(df_rows, columns=['Region', 'Stain', 'Correlation', 'n', 'p-value'])
    df.to_csv(f'~/neuropath/{filename}_correlation.csv')
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = copy.copy(matplotlib.cm.get_cmap("bwr"))
    cmap.set_over('grey')
    im, cbar = heatmap(hm, stains, region_names, ax=ax, vmin=-1, vmax=1,
                       cmap=cmap, cbarlabel=f"{' '.join(filename.split('_')[5:])} correlation")
    plt.savefig(folder + 'corrmap_{}.png'.format(filename), dpi=300, bbox_inches='tight')
    plt.close()


prefixes = ['CG_1', 'FL_mfg_7', 'FL_pg_10', 'TL_stg_32', 'PL_ag_20', 'Amygdala_24',
            'TL_hippocampus_28', 'TL_parahippocampal_30', 'c_37', 'bs_35', 'sn_48', 'th_49',
            'pal_46', 'na_45X', 'cn_36C', 'OL_17_18_19OL']

regions = ['CG anterior cingulate gyrus', 'FL middle frontal gyrus',
           'FL precentral gyrus', 'TL superior temporal gyrus anterior part',
           'PL angular gyrus', 'Amygdala', 'Hippocampus', 'parahippocampal and ambient gyrus',
           'cerebellum', 'brainstem excluding substantia nigra', 'substantia nigra', 'thalamus',
           'pallidum', 'nucleus accumbens', 'caudate nucleus',
           'OL cuneus, lateral remainder, lingual gyrus']



def abbre(text):
    replacements = {'temporal': 'temp', 'occipital': 'occi', 'ventricle': 'vent', 'brainstem': 'stem', 'excluding': 'ex.',
                    'middle': 'mid', 'posterior': 'post', 'anterior': 'ant', 'inferior': 'inf', 'superior': 'sup', 'Lateral': 'lat',
                    'lateral': 'lat', 'frontal': 'fron', 'and': '&', 'parahippocampal': 'parahippo', 'medial': 'med',
                    'TL ': '', 'FL ': '', 'PL ': '', 'OL ': '', 'CG ': ''}
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text

regions = [abbre(reg) for reg in regions]
print(regions)
regions = ['Ant cingulate gyrus',
           'Mid fron gyrus',
           'Precentral gyrus',
           'Sup temp gyrus ant part',
           'Angular gyrus',
           'Amygdala',
           'Hippocampus',
           'Parahippocampus',
           'Cerebellum',
           'Brainstem',
           'Substantia nigra',
           'Thalamus',
           'Pallidum',
           'Nucleus accumbens',
           'Caudate nucleus',
           'Cuneus']

# prefix_idx = {'CG_1':[24],
#               'FL_mfg_7': [28],
#               'FL_pg_10': [50],
#               'TL_stg_32': [82],
#               'PL_ag_20': [32],
#               'Amygdala_24': [4],
#               'TL_hippocampus_28': [2],
#               'TL_parahippocampal_30': [10],
#               'c_37': [18],
#               'bs_35': [19],
#               'sn_48': [74],
#               'th_49': [40],
#               'pal_46': [42],
#               'na_45X': [36],
#               'cn_36C': [34],
#               'OL_17_18_19OL': [64, 66, 22]}

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

stains = ['AB_DP', 'TAU_NFT', 'TAU_NP', 'SILVER_NFT']


def correlate_ABC(method, cohort, layer, filename, folder, thres, score='attention'):
    np_df = pd.read_csv(f'{folder}/{cohort}_NC_MCI_AD_np.csv')
    scores_df = pd.read_csv('{}/{}_{}_scores_{}_layer_{}.csv'.format(folder, cohort, score, layer, method))
    
    df_rows = []
    for var in ['A_score', 'B_score', 'C_score']:
        print(var)
        np_df[var].replace('', np.nan)
        var_df = np_df.dropna(subset=[var])
        ic(len(var_df.values))
        scores = []
        neuropath = []
        for idx, row in var_df.iterrows():
            if row['diff_days'] > thres:
                ic('skipping..')
                continue
            neuropath.append(row[var])
            scores_row = scores_df[scores_df['filename'] == row['filename']].iloc[0]
            if var == 'A_score':
                stain = 'AB_DP'
                regions = ['FL_mfg_7', 'PL_ag_20', 'Amygdala_24', 'TL_parahippocampal_30', 'TL_stg_32']
            elif var == 'B_score':
                stain = 'SILVER_NFT'
                regions = ['TL_hippocampus_28', 'bs_35', 'Amygdala_24', 'TL_parahippocampal_30', 'sn_48', 'pal_46']
            else:
                stain = 'TAU_NP'
                regions = ['FL_mfg_7', 'PL_ag_20', 'TL_hippocampus_28', 'Amygdala_24', 'TL_parahippocampal_30', 'TL_stg_32']
            var_score = []
            for region in regions:
                var_score.extend([scores_row[i] * 1000. for i in prefix_idx[region]])
            print('var_score: ', np.mean(var_score))
            scores.append(np.mean(var_score))
            # scores.append(scores_row[var])
        c, p = stats.spearmanr(scores, neuropath)
        # ordinal_model = OrderedModel(neuropath, scores, distr='logit')
        # results = ordinal_model.fit(method='bfgs')
        # ic(results)
        # p = results.pvalues[0]
        # c = results.llf
        # CI = results.conf_int(alpha=0.05, cols=None)
        # ic(CI)  # [0.025, 0.975]
        boxplot(scores, neuropath, var, folder, f'{thres}days_{method}')
        ic(var, c, p)
        df_rows.append([var, c, len(scores), p])

    df = pd.DataFrame(df_rows, columns=['Score', 'Correlation', 'n', 'p-value'])
    ic(f'~/neuropath/{filename}_ABC_correlation.csv')
    df.to_csv(f'~/neuropath/{filename}_ABC_correlation.csv')

if __name__ == '__main__':
    years = 2
    time_threshold = 365 * years
    plot_path = '/home/dlteif/neuropath'
    method = 'XAIalign_SDG_1'
    # method = 'XAIalign_MDG'
    cohort='FHS'
    score = 'attention'
    layer = 'classifier'

    figs_path = os.path.join(plot_path, 'figures', f'{time_threshold}days_{method}')
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)
    
    # correlate_ABC(method, cohort, layer, '{}days_{}_{}_{}_{}'.format(time_threshold, cohort, score, layer, method), plot_path, time_threshold)
    # exit()
    ordered_regions, ordered_prefixes = [], [] 

    corre = defaultdict(dict)
    annot = defaultdict(dict)
    pval = defaultdict(dict)
    region_to_corre = defaultdict(list)

    layername = 'classifier'
    for region in prefixes:
        for stain in stains:
            ic(f'{region}_{stain}')
            corr, pvalue, n = get_correlation(f'{region}_{stain}', prefix_idx[region], time_threshold, plot_path, method, cohort, score=score, layer=layername)
            ic(corr, pvalue, n)
            corre[region][stain] = corr
            annot[region][stain] = n
            pval[region][stain] = pvalue
            if corr >= -1 and corr<= 1:
                region_to_corre[region].append(corr)
                

    print(region_to_corre)

    pool = [(sum(region_to_corre[key]) / len(region_to_corre[key]) if len(region_to_corre[key]) > 0 else sum(region_to_corre[key]), key, region) for key, region in zip(prefixes, regions)]
    pool.sort(reverse=False)
    print('pool: ', pool)

    # if not ordered_prefixes:
    ordered_prefixes = [p[1] for p in pool if p[0] != 0]
    ordered_regions = [p[2] for p in pool if p[0] != 0]

    print('region_to_corre: ', region_to_corre)
    ic(len(ordered_prefixes))            
    ic(len(ordered_regions))            

    plot_corr_heatmap(ordered_prefixes, ordered_regions, stains, corre, annot, pval, '{}days_{}_{}_{}_{}'.format(time_threshold, cohort, score, layername, method), figs_path)
    exit()

    pooled_corre = defaultdict(dict)
    pooled_regions, pooled_prefixes = [], []
    avg = 0.0
    for p in pool:
        ic(p[1], p[2])
        avg += p[0]
        if p[0] == 0:
            continue
        pooled_corre[p[1]]['Average Corr.'] = p[0]
        pooled_prefixes.append(p[1])
        pooled_regions.append(p[2])

    avg /= len(pool)
    ic(avg)
    
    ic(pooled_corre)
    stains_corre = defaultdict(list)
    for region, stains_dict in corre.items():
        ic(region, stains_dict)
        for stain, corr in stains_dict.items():
            if corr >= -1 and corr <= 1:
                stains_corre[stain].append(corr)
        # pooled_stains[stain] = 

    pooled_stains = defaultdict(float)
    for stain, corrs in stains_corre.items():
        pooled_stains[stain] = sum(corrs) / len(corrs)
    
    ic(pooled_stains)


    plot_corr_heatmap(pooled_prefixes, pooled_regions, ['Average Corr.'], pooled_corre, annot, '{}days_{}_pooled_{}_{}_{}'.format(time_threshold, cohort, score, layername, method), plot_path+'/figures/')
    # plot_corr_heatmap(pooled_prefixes, pooled_regions, ['Average Corr.'], pooled_stains, annot, '{}days_{}_pooledstains_shap_{}_{}'.format(time_threshold, cohort, layername, method), plot_path+'/figures/')