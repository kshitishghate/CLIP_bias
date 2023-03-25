import math
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from CLIP import clip


def load_all_results(models=None, openai_only=False):
    all_results = pd.read_csv(os.path.join('results','data','seat_replication.csv'))
    open_ai_model_names = [m.replace('/','') for m in clip.available_models()] + clip.available_models()
    if openai_only:
        all_results = all_results[all_results['model'].isin(open_ai_model_names)]
    else:
        all_results = all_results[~all_results['model'].isin(open_ai_model_names)]

    if models is not None:
        all_results = all_results[all_results['model'].isin(models)]

    all_results['Test'] = all_results["test_name"].str.replace('.jsonl','',regex=False).str.replace('_',' ',regex=False).str.title()
    all_results['Test'] = all_results['Test'].str.strip()
    all_results = all_results[all_results['Test'].str.contains('Weat')]


    all_results = all_results.sort_values([
        'test_name'
    ])

    all_results['Test'] = all_results['Test'].replace({
        'Weat10': 'Age/Valence',
        'Weat1': 'Flower/Valence',
        'Weat2': 'Instruments/Valence',
        'Weat3': 'EA-AA Names/Valence',
        'Weat3B': 'EA - AA Terms / Valences',
        'Weat4': 'EA-AA Names/Valence',
        'Weat5': 'EA-AA Names/Valence',
        'Weat5B': 'EA-AA Terms/Valence',
        'Weat6': 'Gendered Names/Career',
        'Weat6B': 'Gendered Terms/Career',
        'Weat7': 'Math/Gendered Terms',
        'Weat7B': 'Math/Gendered Names',
        'Weat8': 'Science/Gendered Terms',
        'Weat8B': 'Science/Gendered Names',
        'Weat9': 'Physical Disease/Permanent',

        'Sent-Weat10': 'Sentences: Age/Valence',
        'Sent-Weat1': 'Sentences: Flower/Valence',
        'Sent-Weat2': 'Sentences: Instruments/Valence',
        'Sent-Weat3': 'Sentences: EA-AA Names/Valence',
        'Sent-Weat3B': 'Sentences: EA - AA Terms / Valences',
        'Sent-Weat4': 'Sentences: EA-AA Names/Valence',
        'Sent-Weat5': 'Sentences: EA-AA Names/Valence',
        'Sent-Weat5B': 'Sentences: EA-AA Terms/Valence',
        'Sent-Weat6': 'Sentences: Gendered Names/Career',
        'Sent-Weat6B': 'Sentences: Gendered Terms/Career',
        'Sent-Weat7': 'Sentences: Math/Gendered Terms',
        'Sent-Weat7B': 'Sentences: Math/Gendered Names',
        'Sent-Weat8': 'Sentences: Science/Gendered Terms',
        'Sent-Weat8B': 'Sentences: Science/Gendered Names',
        'Sent-Weat9': 'Sentences: Physical Disease/Permanent',

    })



    original_results = pd.read_csv(os.path.join('data','seat_results','results.tsv'), sep='\t')
    original_results['Test'] = original_results["test"].str.replace('.jsonl','',regex=False).str.replace('_',' ',regex=False).str.title()
    original_results['Test'] = original_results['Test'].str.strip()
    original_results = original_results[original_results['Test'].str.contains('Weat')]
    original_results['Test'] = original_results['Test'].replace({
        'Weat10': 'Age/Valence',
        'Weat1': 'Flower/Valence',
        'Weat2': 'Instruments/Valence',
        'Weat3': 'EA-AA Names/Valence',
        'Weat3B': 'EA - AA Terms / Valences',
        'Weat4': 'EA-AA Names/Valence',
        'Weat5': 'EA-AA Names/Valence',
        'Weat5B': 'EA-AA Terms/Valence',
        'Weat6': 'Gendered Names/Career',
        'Weat6B': 'Gendered Terms/Career',
        'Weat7': 'Math/Gendered Terms',
        'Weat7B': 'Math/Gendered Names',
        'Weat8': 'Science/Gendered Terms',
        'Weat8B': 'Science/Gendered Names',
        'Weat9': 'Physical Disease/Permanent',

        'Sent-Weat10': 'Sentences: Age/Valence',
        'Sent-Weat1': 'Sentences: Flower/Valence',
        'Sent-Weat2': 'Sentences: Instruments/Valence',
        'Sent-Weat3': 'Sentences: EA-AA Names/Valence',
        'Sent-Weat3B': 'Sentences: EA - AA Terms / Valences',
        'Sent-Weat4': 'Sentences: EA-AA Names/Valence',
        'Sent-Weat5': 'Sentences: EA-AA Names/Valence',
        'Sent-Weat5B': 'Sentences: EA-AA Terms/Valence',
        'Sent-Weat6': 'Sentences: Gendered Names/Career',
        'Sent-Weat6B': 'Sentences: Gendered Terms/Career',
        'Sent-Weat7': 'Sentences: Math/Gendered Terms',
        'Sent-Weat7B': 'Sentences: Math/Gendered Names',
        'Sent-Weat8': 'Sentences: Science/Gendered Terms',
        'Sent-Weat8B': 'Sentences: Science/Gendered Names',
        'Sent-Weat9': 'Sentences: Physical Disease/Permanent',

    })

    return all_results, original_results

def plot_overall(df, original_df, y_axis_col, openai_only):
    plt.clf()
    df = df.copy()

    def jitter(y, jitter_size=0.25):
        return y + random.uniform(0, jitter_size * 2) - jitter_size



    seat_clip_means = {k:v for k,v in df.groupby(y_axis_col)['effect_size'].mean().iteritems()}
    seat_original_means = {k:v for k,v in original_df.groupby(y_axis_col)['effect_size'].mean().iteritems()}
    assert len([k for k in seat_clip_means.keys() if k in seat_original_means.keys()]) == len(seat_clip_means)
    assert len([k for k in seat_clip_means.keys() if k in seat_original_means.keys()]) == len(seat_original_means)

    aggregated_info = {k:(y_val, seat_clip_means[k],seat_original_means[k]) for k, y_val in
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
        plt.scatter([-10], [-100], marker='2', c='#117733', label='SEAT $d$ for \nIndivid. Text Model', s=50),
        plt.scatter([-10], [-100], marker='v', c='#44AA99', label='Mean SEAT $d$ for Text Models', s=100),
    ]
    ax.legend(handles=handles,
              # bbox_to_anchor=(0,0.85),
              loc='upper left')

    plt.xlim(-1.99, 1.99)
    plt.ylim(-0.5, len(aggregated_info)-0.5)

    plt.tight_layout()
    if openai_only:
        plt.savefig(os.path.join('results','plots','openai_models_only','seat_replication.pdf'))
    else:
        plt.savefig(os.path.join('results','plots','seat_replication.pdf'))


if __name__ == '__main__':
    openai_only = False
    all_results, original_results = load_all_results(openai_only=openai_only)
    plot_overall(all_results, original_results, 'Test', openai_only=openai_only)









