import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os
from scipy import stats
from plot_brains import brain_regions, prefixes, prefix_idx
from collections import defaultdict
import copy
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from icecream import ic

regions = {24.: 'Ant cingulate gyrus',
           28.: 'Mid fron gyrus',
           50.: 'Precentral gyrus',
           82.: 'Sup temp gyrus ant part',
           32.: 'Angular gyrus',
           3.: 'Amygdala',
           1.: 'Hippocampus',
           9.: 'Parahippocampus',
           17.: 'Cerebellum',
           19.: 'Brainstem',
           74.: 'Substantia nigra',
           40.: 'Thalamus',
           42.: 'Pallidum',
           36.: 'Nucleus accumbens',
           34.: 'Caudate nucleus',
           64.: 'Cuneus, lateral remainder, lingual gyrus'}



def plot_bargraph(x, Y, methods, bar_width=0.25):
    # Data for the bars
    offset = 2
    # Create the bar graph
    colors = ['r','g']
    fig = plt.subplots(figsize=(12, 8))
    br1 = np.arange(len(Y[0]))
    methods = ['ERM', 'Ours']
    x = ['AB deposits', 'Tau NFT', 'Tau NP', 'Silver NFT']
    for idx, y in enumerate(Y):
        plt.bar(br1 + bar_width * idx, y, width=bar_width, align='center', label=methods[idx], color=colors[idx], edgecolor='grey')

    

    # Add labels and title
    plt.xlabel('Stain', fontsize=24, fontweight='bold')
    plt.ylabel('Mean rank correlation coefficient', fontsize=24, fontweight='bold')
    plt.xticks([r + bar_width/2 for r in range(len(Y[0]))], x, fontsize=24)
    plt.yticks(fontsize=20)
    # Add a legend
    plt.legend(fontsize=24)

    # Display the graph
    plt.show()
    plt.savefig('MDG_bargraph.pdf', dpi=300, transparent=False, bbox_inches='tight')


def boxplot(x, scores, neuropath, save_path, labels):
    x_scores = [0, 1, 2, 3]
    all_data = [[] for _ in range(len(x_scores))]
    # all_maximums = [y.max(axis=1).values for y in Y]
    # data_maximums = [max(m) for m in all_maximums]
    # y_max = max(data_maximums)

    # all_minimums = [y.min(axis=1).values for y in Y]
    # data_minimums = [max(m) for m in all_minimums]
    # y_min = min(data_minimums)
    # y_range = y_max - y_min

    # x_pos_range = np.arange(len(Y)) / (len(Y) - 1)
    # x_pos = (x_pos_range * 0.5) + 0.75

    for score, data in enumerate(scores, neuropath):
        all_data[x_scores.index(score)].append(data)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
    # fig.suptitle("Neurofibrillary Tangles", fontweight='normal', fontsize=10)
    fig.subplots_adjust(top=0.85)
    ax.set_title('Neurofibrillary Tangles')
    ax.boxplot(all_data,
               vert=True,  # vertical box alignment
               patch_artist=True,  # fill with color
               labels=x_scores,  # will be used to label x-ticks
               zorder=0,
               showfliers=False)
    for x, data in zip(x_scores, all_data):
        ax.plot(np.random.normal(x+1, 0.1, size=len(data)), data, 'r.',
                alpha=0.6, zorder=10, markersize=8)
    c, p = stats.spearmanr(scores, neuropath)
    ax.set_title('corr={:.4f}'.format(c))
    ax.set_xlabel(name)
    ax.set_ylabel('Shap value')
    plt.savefig(f'{folder}/figures/{method}/{name}.png', dpi=200, bbox_inches='tight')
    plt.close()

def get_pooled_regionscores(method):
    fname = f'~/neuropath/FHS_attention_scores_classifier_layer_{method}.csv'
    df = pd.read_csv(fname)
    rows = []
    regions = prefix_idx.keys()
    for idx, row in df.iterrows():
        scores_row = [row['filename']]
        for region, indices in prefix_idx.items():
            print(indices)
            scores_row.append(np.mean([row[i] for i in indices]))

        rows.append(scores_row)

    pooled_regionscores_df = pd.DataFrame(rows, columns=['filename'] + list(regions))
    pooled_regionscores_df.to_csv(f'~/neuropath/FHS_pooled_attention_scores_classifier_layer_{method}.csv')

def region_boxplot(method, region):
    fname = f'~/neuropath/FHS_pooled_attention_scores_classifier_layer_{method}.csv'
    df = pd.read_csv(fname)
    scores = []
    for idx, row in df.iterrows():
        scores.append(row[region])

    neuropath_df = pd.read_csv('~/neuropath/FHS_NC_MCI_AD_np.csv')
    neuropath = defaultdict(list)
    for idx, row in neuropath_df.iterrows():
        for stain in ['AB_DP', 'TAU_NFT', 'SILVER_NFT', 'TAU_NP']:
            neuropath[stain].append(row[f'{region}_{stain}'])

    stain_corr = {}
    for stain in neuropath.keys():
        corr, pvalue = stats.spearmanr(scores, neuropath[stain])
        pass
    

def heatmap(data, row_labels, col_labels, significant_matrix, ax=None, extra_label="", cbar_kw={}, cbarlabel="", **kwargs):
    empty_idx = []
    for col in range(data.shape[1]):
        ic(set(data[:,col]))
        if len(set(data[:,col])) == 1 and list(set(data[:,col]))[0] == 100.:
            empty_idx.append(col)
    ic(empty_idx)
    if len(empty_idx) > 0:
        data = np.delete(data, empty_idx, axis=1)
        significant_matrix = np.delete(significant_matrix, empty_idx, axis=1)
        col_labels = list(col_labels)
        empty_idx.sort(reverse=True)
        for i in empty_idx:
            col_labels.pop(i)

    col_labels = [regions[prefix_idx[reg][0]] for reg in col_labels]
    
    if not ax:
        ax = plt.gca()
    im = ax.imshow(data, **kwargs)
    if cbarlabel != "":
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.5, orientation="horizontal", **cbar_kw)
        cbar.set_label(cbarlabel, rotation=0, fontsize=20)
    else:
        cbar = None
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    
    ax.set_xticklabels(col_labels, fontweight="normal", fontsize=20)
    ax.set_yticklabels(row_labels, fontweight="normal", fontsize=20)
    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=20, ha="left",
             rotation_mode="anchor")
    
    for edge, spine in ax.spines.items():
        spine.set_visible(False)
    
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            print(i, j)
            text = ax.text(
                j + 0.02,
                i + 0.03,
                significant_matrix[i,j],
                ha="center",
                va="center",
                color="black",
            )
            text.set_fontsize(10)

    # text = ax.text(
    #         -4,
    #         0.8,
    #         extra_label,
    #         ha="left",
    #         va="center",
    #         # rotation=90,
    #         wrap=True,
    #         bbox=dict(boxstyle="square", fc='w', alpha=0, ec='w')
    #         # transform=ax.get_yaxis_transform(),
    # )
    # text._get_wrap_line_width = lambda : 50 if 'Neuritic' in extra_label else 100
    # text.set_fontsize(10)
    
    return im, cbar

def corr_significance_heatmap(methods, cohort, setting, days, csv_dir, save_path='~/neuropath/figures'):
    from matplotlib import rc, rcParams
    rc('axes', linewidth=1)
    rc('font', weight='bold')
    rcParams.update({'font.size': 7})
    dfs = [pd.read_csv(os.path.join(csv_dir, f'{days}days_{cohort}_attention_classifier_{method}{setting}_correlation.csv')) for method in methods]
    stains = ['AB_DP', 'SILVER_NFT', 'TAU_NP']
    final_regions = set(dfs[0]['Region'].values)
    print(final_regions)
    hm = np.zeros((len(stains) * len(methods), len(final_regions)))
    pvals = np.zeros((len(stains) * len(methods), len(final_regions)))
    for st_idx, stain in enumerate(stains):
        for idx, df in enumerate(dfs):
            for r_idx, region in enumerate(final_regions):  
                # corr_dict[methods[idx]][stain][row['Region']] = row['Correlation']
                print(stain, region)
                row = df.loc[(df['Stain'] == stain) & (df['Region'] == region)]
                print(row)
                if row.empty:
                    continue
                hm[st_idx*len(methods) + idx, r_idx] = row['Correlation'].values[0]
                pvals[st_idx*len(methods) + idx, r_idx] = row['p-value'].values[0]
    
    significant_matrix = np.array([["***" if (pvals[i,j] < 0.001) else "**" if (pvals[i,j] < 0.01) else "*" if (pvals[i,j] < 0.05) else "" for j in range(pvals.shape[1])] for i in range(pvals.shape[0])])
    # significant_matrix = np.array([["*" if (pvals[i,j] < 0.001) else "" for j in range(pvals.shape[1])] for i in range(pvals.shape[0])])
    ic(significant_matrix.shape, hm.shape)
    filename = f"{days}days_{cohort}_attention_{'_'.join(methods)}{setting}"
    # fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    fig, ax = plt.subplots(1, 1, figsize=(16, 8), constrained_layout=True, sharex=True, sharey=True)
    
    # plt.figure()
    cmap = copy.copy(matplotlib.cm.get_cmap("vlag"))
    cmap.set_over('grey')
    stains = ['ERM', 'Ours', 'ERM', 'Ours', 'ERM', 'Ours']
    
    # im, cbar = heatmap(hm.transpose(1,0), label_regions, stains, ax=ax, vmin=-1, vmax=1,
    #                    cmap=cmap, cbarlabel="Correlation")
    labels = methods[:-1] + ['Ours']

    # heatmap(hm[:len(methods),...], labels, label_regions, significant_matrix[:len(methods),...], ax=ax[0], extra_label="Amyloid beta deposits", vmin=-1, vmax=1, cmap=cmap, cbarlabel="")
    heatmap(hm[len(methods):len(methods)*2, ...], labels, final_regions, significant_matrix[len(methods):len(methods)*2,...], ax=ax, extra_label="Neurofibrillary tangles", vmin=-1, vmax=1, cmap=cmap, cbarlabel="Correlation")
    # heatmap(hm[len(methods)*2:, ...], labels, label_regions, significant_matrix[len(methods)*2:,...], ax=ax[2], extra_label="Neuritic plaques", vmin=-1, vmax=1, cmap=cmap, cbarlabel="Correlation")
    
    # heatmap(hm[len(methods):len(methods)*2, ...].transpose(), label_regions, labels, significant_matrix[len(methods):len(methods)*2,...].transpose(), ax=ax, extra_label="Neurofibrillary tangles", vmin=-1, vmax=1, cmap=cmap, cbarlabel="Correlation")
   

    # Save a high-res copy of the image to disk
    plt.margins(x=0)

    # custom_lines = [Line2D([0], [0], color=cmap(100.), lw=4)]
    custom_lines = [#plt.Rectangle((0,0), 1, 1, fc="gray", label='Not available'),
                    plt.scatter([], [], marker=r'$\ast$', label="p < .05", color='black', linestyle='None', linewidths=0.5),
                    # plt.scatter([], [], marker=r'$\ast\ast$', label="p < .01", color='black', linestyle='None', linewidths=0.8),
                    # plt.scatter([], [], marker=r'$\ast\ast\ast$', label="p < .001", color='black', linestyle='None', linewidths=1),
                    ]
    fig.legend(handles=custom_lines, loc="upper right", prop=dict(weight="normal", size=16), handlelength=2, handleheight=2, bbox_to_anchor=(0.2,1))

    plt.savefig('./corr_significance_{}.pdf'.format(filename), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    methods = ['ERM', 'MMD', 'RSC', 'VREx', 'GroupDRO', 'XAIalign']
    # methods = ['ERM', 'RSC', 'XAIalign']
    cohort = 'FHS'
    setting = '_MDG'
    days = 3 * 365
    csv_dir = '~/neuropath/'
    corr_significance_heatmap(methods, cohort, setting, days, csv_dir)
    exit()
    dfs = [pd.read_csv(os.path.join(csv_dir, f'{days}days_{cohort}_attention_classifier_{method}{setting}_correlation.csv')) for method in methods]
    x = ['AB_DP', 'TAU_NFT', 'TAU_NP', 'SILVER_NFT']
    Y = [[] for _ in range(len(methods))]
    for stain in x:
        for idx, df in enumerate(dfs):
            vals = [v for v in list(df.loc[df['Stain'] == stain]['Correlation'].values) if v != 100]
            Y[idx].append(np.mean(vals))

    print(x)
    print(Y)

    plot_bargraph(x, Y, methods)
    # region_boxplot('ERM_MDG', '')