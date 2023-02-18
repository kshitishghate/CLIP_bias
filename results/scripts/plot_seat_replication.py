import math
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


all_results = pd.read_csv(os.path.join('results','data','seat_replication.csv'))


all_results['Test'] = all_results["test_name"].str.replace('.jsonl','',regex=False).str.replace('_',' ',regex=False).str.title()
all_results['Test'] = all_results['Test'].str.strip()


all_results = all_results.sort_values([
    'test_name'
])



def plot_overall(df, y_axis_col):
    plt.clf()
    df = df.copy()

    def jitter(y, jitter_size=0.25):
        return y + random.uniform(0, jitter_size * 2) - jitter_size



    seat_clip_means = {k:v for k,v in df.groupby(y_axis_col)['effect_size'].mean().iteritems()}
    type_ids = {k:(y_val, seat_clip_means[k]) for k,y_val in
                zip(df[y_axis_col].unique(),
                    range(len(df[y_axis_col].unique())-1, -1, -1))}

    df[y_axis_col] = df[y_axis_col].apply(lambda x: type_ids[x][0])
    df[y_axis_col] = df[y_axis_col].apply(lambda x: jitter(x))


    fig, (ax) = plt.subplots(1, 1, figsize=(10, 10))


    plt.scatter(y=df[y_axis_col],
                x=df['effect_size'],
                c='gray')


    plt.axvline(x=0, color='black')

    for mean_y, seat_clip in type_ids.values():
        plt.axhline(y=mean_y-0.5,c='gray',alpha=0.2)
        plt.scatter([seat_clip], mean_y, marker='s', c='#D81B60',s=100,zorder=10)


    # label y axis
    plt.yticks(range(len(type_ids)))
    labs = list(type_ids.keys())
    labs.reverse()
    plt.gca().set_yticklabels(labs)



    # add legend
    handles = [
        plt.scatter([-10], [-100], marker='o', c='gray', label='SEAT $d$ for \nIndivid. CLIP Model', s=50),
        plt.scatter([-10], [-100], marker='s',  c='#D81B60', label='Mean SEAT $d$ for CLIP Models', s=100)]
    ax.legend(handles=handles,
              # bbox_to_anchor=(0,0.85),
              loc='upper left')

    plt.xlim(-1.99, 1.99)
    plt.ylim(-0.5, len(type_ids)-0.5)

    plt.tight_layout()
    plt.savefig(os.path.join('results','plots','seat_replication.pdf'))


if __name__ == '__main__':
    plot_overall(all_results, 'Test')









