import math
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


ieats = pd.read_csv(os.path.join('results','data','ieat_replication.csv'))
ieats['Test'] = ieats['Target'] + '/' + ieats['A'] + ' vs. ' + ieats['B']
ieats['Test'] = ieats['Test'].str.replace('Pleasant vs. Unpleasant', 'Valence',regex=False)
ieats['Test'] = ieats['Test'].str.replace('Weapon/Tool vs. Weapon', 'Race/Tool vs. Weapon',regex=False)
ieats['Test'] = ieats['Test'].str.replace('Weapon/Tool-modern vs. Weapon-modern', 'Race/Tool vs. Weapon (Modern)',regex=False)
ieats = ieats.sort_values([
    'Test'
])

original_results = pd.read_csv(os.path.join('ieat','output','results.csv'))
original_results = original_results.rename(columns={'d':'effect_size'})
original_results['Test'] = original_results['Test'].replace(
    {'Disability':'Disabled',
     'Gender-Career':'Gender',
     'Gender-Science':'Gender',
     'Weapon-Race':'Race'
     }
)
original_results['Test'] = (original_results['Test'] + '/' + original_results['A'].str.title()
                            + ' vs. ' + original_results['B'].str.title())
original_results['Test'] = original_results['Test'].replace({
    'Native/Us vs. World':'Native/US vs. World',
    'Weapon-Race (Modern)/Tool-Modern vs. Weapon-Modern':'Race/Tool vs. Weapon (Modern)'
})
original_results['Test'] = original_results['Test'].str.replace('Pleasant vs. Unpleasant', 'Valence',regex=False)
original_results['Test'] = original_results['Test'].str.replace('Weapon/Tool vs. Weapon', 'Race/Tool vs. Weapon',regex=False)
original_results['Test'] = original_results['Test'].str.replace('Weapon/Tool-modern vs. Weapon-modern', 'Race/Tool vs. Weapon (Modern)',regex=False)

original_results = original_results[original_results['Test'].isin(ieats['Test'])]
ieats = ieats[ieats['Test'].isin(original_results['Test'])]

def plot_overall(df, original_df, y_axis_col):
    plt.clf()
    df = df.copy()

    def jitter(y, jitter_size=0.25):
        return y + random.uniform(0, jitter_size * 2) - jitter_size



    ieat_clip_means = {k:v for k,v in df.groupby(y_axis_col)['effect_size'].mean().iteritems()}
    ieat_original_means = {k:v for k,v in original_df.groupby(y_axis_col)['effect_size'].mean().iteritems()}

    assert len([k for k in ieat_clip_means.keys() if k in ieat_original_means.keys()]) == len(ieat_clip_means)
    assert len([k for k in ieat_clip_means.keys() if k in ieat_original_means.keys()]) == len(ieat_original_means)

    aggregated_info = {k:(y_val, ieat_clip_means[k], ieat_original_means[k]) for k, y_val in
                zip(df[y_axis_col].unique(),
                    range(len(df[y_axis_col].unique())-1, -1, -1))}

    df[y_axis_col] = df[y_axis_col].apply(lambda x: aggregated_info[x][0])
    df[y_axis_col] = df[y_axis_col].apply(lambda x: jitter(x))

    original_df[y_axis_col] = original_df[y_axis_col].apply(lambda x: aggregated_info[x][0])
    original_df[y_axis_col] = original_df[y_axis_col].apply(lambda x: jitter(x))


    fig, (ax) = plt.subplots(1, 1, figsize=(10, 6))


    plt.scatter(y=df[y_axis_col],
                x=df['effect_size'],
                c='#AA4499',
                marker='+')
    plt.scatter(y=original_df[y_axis_col],
                x=original_df['effect_size'],
                c='#117733',
                marker='2')


    plt.axvline(x=0, color='black')

    for mean_y, seat_clip, seat_text in aggregated_info.values():
        plt.axhline(y=mean_y-0.5,c='gray',alpha=0.2)
        plt.scatter([seat_clip], mean_y, marker='s', c='#882255',s=100,zorder=10)
        plt.scatter([seat_text], mean_y, marker='v', c='#44AA99',s=100,zorder=10)

    # label y axis
    plt.yticks(range(len(aggregated_info)))
    labs = list(aggregated_info.keys())
    labs.reverse()
    plt.gca().set_yticklabels(labs)



    # add legend
    handles = [
        plt.scatter([-10], [-100], marker='+', c='#AA4499', label='SEAT $d$ for \nIndivid. CLIP Model', s=50),
        plt.scatter([-10], [-100], marker='s', c='#882255', label='Mean SEAT $d$ for CLIP Models', s=100),
        plt.scatter([-10], [-100], marker='2', c='#117733', label='iEAT $d$ for \nIndivid. Image Model', s=50),
        plt.scatter([-10], [-100], marker='v', c='#44AA99', label='Mean iEAT $d$ for Image Models', s=100),
    ]
    ax.legend(handles=handles,
              # bbox_to_anchor=(0,0.85),
              loc='upper left')

    plt.xlim(-1.99, 1.99)
    plt.ylim(-0.5, len(aggregated_info)-0.5)

    plt.tight_layout()
    plt.savefig(os.path.join('results','plots','ieat_replication.pdf'))


if __name__ == '__main__':
    plot_overall(ieats, original_results, 'Test')









