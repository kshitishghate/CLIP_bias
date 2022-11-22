import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from results.scripts.load_results import load_results


def plot_results():

    for dataset in ['fairface', 'cfd']:
        results = load_results(dataset)
        save_dir = os.path.join('results','plots',f'categorical_synonyms_{dataset}')
        os.makedirs(save_dir,exist_ok=True)
        order = results['X vs. Y'].unique()
        order.sort()
        hue_order = results['Comparison Method (Context)'].unique()
        hue_order.sort()
        for emotion, data in results.groupby('emotion'):
            sns.catplot(y='X vs. Y',x='effect_size', data=data, order=order, aspect=1.5, hue='Comparison Method (Context)',
                        kind='box', hue_order=hue_order)
            plt.axvline(0, c='black')
            # plt.title(emotion.title())
            plt.xlim(-2,2)
            # plt.tight_layout()

            plt.savefig(os.path.join(save_dir,emotion + '.jpg'))
            plt.clf()
            plt.close()


    results_fairface = load_results('fairface')
    results_cfd = load_results('cfd')

    merged_results = results_fairface.merge(results_cfd, on=['Title', 'X', 'Y', 'A', 'B', 'model', 'na', 'npermutations',
                                                             'comparison','X vs. Y', 'Comparison Method (Context)',
                                                             'context','emotion'], suffixes=('_fairface', '_cfd'))
    merged_results = merged_results.rename(columns={'effect_size_fairface':'Effect Size (FairFace)',
                                            'effect_size_cfd':'Effect Size (Chicago Face)'})

    sns.jointplot(data=merged_results, x='Effect Size (FairFace)',y='Effect Size (Chicago Face)', hue='Title')


    plt.savefig(os.path.join('results','plots','fairface_cfd_comparison.png'))
    print(merged_results.groupby('Title')[['Effect Size (FairFace)','Effect Size (Chicago Face)']].corr())



if __name__ == '__main__':
    plot_results()
