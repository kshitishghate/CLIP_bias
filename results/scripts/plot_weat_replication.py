import math
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


all_results = pd.read_csv(os.path.join('results','data','weat_replication.csv'))


all_results['Test'] = ('WEAT '
                       + all_results["weat_num"].astype(str) + ' ('
                       + all_results["X"] + " and "
                       + all_results['Y'] + " / "
                       + all_results['A'] + " and "
                       + all_results['B'] + ')')
all_results['Test'] = all_results['Test'].str.strip()


all_results = all_results.sort_values([
    'weat_num'
])



def plot_overall(df, y_axis_col):
    plt.clf()
    df = df.copy()

    def jitter(y, jitter_size=0.25):
        return y + random.uniform(0, jitter_size * 2) - jitter_size



    weat_clip_means = {k:v for k,v in df.groupby(y_axis_col)['effect_size'].mean().iteritems()}
    weat_glove_means = {'WEAT 1 (Flowers and Insects / Pleasant and Unpleasant)':1.50,
       'WEAT 2 (Instruments and Weapons / Pleasant and Unpleasant)':1.53,
       'WEAT 3 (EA names and AA names / Pleasant and Unpleasant)':1.41,
       'WEAT 4 (EA names and AA names / Pleasant and Unpleasant)':1.50,
       'WEAT 5 (EA names and AA names / Pleasant and Unpleasant)':1.28,
       'WEAT 6 (Male names and Female names / Career and Family)':1.81,
       'WEAT 7 (Math and Arts / Male and Female)':1.06,
       'WEAT 8 (Science and Arts / Male and Female)':1.24,
       'WEAT 9 (Mental and Physical disease / Temporary and Permanent)':1.38,
       'WEAT 10 (Young and Old / Pleasant and Unpleasant)':1.53
    }
    type_ids = {k:(y_val, weat_clip_means[k],weat_glove_means[k]) for k,y_val in
                zip(df[y_axis_col].unique(),
                    range(len(df[y_axis_col].unique())-1, -1, -1))}

    df[y_axis_col] = df[y_axis_col].apply(lambda x: type_ids[x][0])
    df[y_axis_col] = df[y_axis_col].apply(lambda x: jitter(x))


    fig, (ax) = plt.subplots(1, 1, figsize=(10, 4))


    plt.scatter(y=df[y_axis_col],
                x=df['effect_size'],
                c='#AA4499',
                marker='+')





    plt.axvline(x=0, color='black')

    for mean_y, weat_clip, weat_glove in type_ids.values():
        plt.axhline(y=mean_y-0.5,c='gray',alpha=0.2)
        plt.scatter([weat_clip], mean_y, marker='s', c='#882255',s=100,zorder=10)
        plt.scatter([weat_glove], mean_y, marker='v', c='#44AA99',s=100,zorder=10)


    # label y axis
    plt.yticks(range(len(type_ids)))
    labs = list(type_ids.keys())
    labs.reverse()
    plt.gca().set_yticklabels(labs)



    # add legend
    handles = [
        plt.scatter([-10], [-100], marker='+', c='#AA4499', label='WEAT $d$ for \nIndivid. CLIP Model', s=50),
        plt.scatter([-10], [-100], marker='s',  c='#882255', label='Mean WEAT $d$ for CLIP Models', s=100),
               plt.scatter([-10], [-100], marker='v', c='#44AA99', label='WEAT $D$ for GloVe', s=100)]
    ax.legend(handles=handles,
              # bbox_to_anchor=(0,0.85),
              loc='upper left')

    plt.xlim(-1.99, 1.99)
    plt.ylim(-0.5, len(type_ids)-0.5)

    plt.tight_layout()
    plt.savefig(os.path.join('results','plots','weat_replication.pdf'))


def plot_cat(df, y_axis_col, hue_col):
    plt.clf()
    df = df.copy()

    def jitter(y, jitter_size=0.25):
        return y + random.uniform(0, jitter_size * 2) - jitter_size

    type_ids = {k:v for k,v in zip(df[y_axis_col].unique(), range(len(df[y_axis_col].unique())-1,
                                                                  -1,
                                                                  -1))}

    df[y_axis_col] = df[y_axis_col].apply(lambda x: type_ids[x])
    df[y_axis_col] = df[y_axis_col].apply(lambda x: jitter(x))

    colors = ['#332288','#88CCEE','#DDCC77','#882255','#117733','#E83C59','#88CCEE','#AA4499']
    markers = ['o','P','s','v','X', 'D']
    mmap = {k:[c,m] for k, c, m in zip(df[hue_col].unique(),
                                       colors[:len(df[hue_col].unique())],
                                       markers[:len(df[hue_col].unique())],
                                       )}

    fig, (ax) = plt.subplots(1, 1, figsize=(10, 4))

    for v, (c, m) in mmap.items():
        plt.scatter(y=df[y_axis_col][df[hue_col] == v],
                    x=df['SpEAT d'][df[hue_col] == v],
                    c=c,
                    marker = m)

    # add legend
    handles = [plt.Line2D([0], [0], marker=m, color='w', markerfacecolor=c, label=v, markersize=8) for v, (c,m) in
               mmap.items()]
    ax.legend(title=hue_col, handles=handles,
              # bbox_to_anchor=(0,0.85),
              loc='upper right')


    plt.yticks(range(len(type_ids)))
    labs = list(type_ids.keys())
    labs.reverse()
    plt.gca().set_yticklabels(labs)

    plt.axvline(x=0, color='black')

    for mean_y in range(len(type_ids) - 1):
        plt.axhline(y=mean_y+0.5,c='gray',alpha=0.2)


    plt.xlim(-1.99, 1.99)

    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'eats', 'images', f'{hue_col.lower().replace(" ","_")}.pdf'))


if __name__ == '__main__':
    plot_overall(all_results, 'Test')









