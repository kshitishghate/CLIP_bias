import math
import os
import random

from CLIP import clip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_all_results(models=None, openai_only=False):
    all_results = pd.read_csv(os.path.join('results','data','weat_replication.csv'))
    open_ai_model_names = clip.available_models()
    if openai_only:
        all_results = all_results[all_results['model'].isin(open_ai_model_names)]
    else:
        all_results = all_results[~all_results['model'].isin(open_ai_model_names)]

    if models is not None:
        all_results = all_results[all_results['model'].isin(models)]

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

    return all_results



def plot_overall(df, y_axis_col, openai_only):
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
    if openai_only:
        plt.savefig(os.path.join('results','plots','openai_models_only','weat_replication.pdf'))
    else:
        plt.savefig(os.path.join('results','plots','weat_replication.pdf'))





if __name__ == '__main__':
    openai_only = True
    all_results = load_all_results(openai_only=openai_only)
    plot_overall(all_results, 'Test', openai_only=openai_only)









