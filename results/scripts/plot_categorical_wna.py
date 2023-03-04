import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def plot_results():
    results = pd.read_csv(os.path.join('results','data','categorical_wna.csv'))
    results['emotion'] = np.where(
        results['A'].str.contains('anger'), 'anger',
    np.where(
        results['A'].str.contains('sadness'), 'sadness',
    np.where(
        results['A'].str.contains('anticipation'), 'anticipation',
    np.where(
        results['A'].str.contains('disgust'), 'disgust',
    np.where(
        results['A'].str.contains('fear'), 'fear',
    np.where(
        results['A'].str.contains('joy'), 'joy',
    np.where(
        results['A'].str.contains('surprise'), 'surprise',
    np.where(
        results['A'].str.contains('trust'), 'trust',
        'ERROR'
    ))))))))
    results['comparison'] = np.where(results['B'].str.contains('apathy'), 'apathy','negation')




    results['X'] = results['X'].map(lambda x: 'European-American' if x == 'Euro' else x, na_action='ignore')
    results['Y'] = results['Y'].map(lambda x: 'Native-American' if x == 'Native' else x, na_action='ignore')
    results['X'] = results['X'].map(lambda x: 'Light-Skin' if x == 'Light' else x, na_action='ignore')
    results['Y'] = results['Y'].map(lambda x: 'Dark-Skin' if x == 'Dark' else x, na_action='ignore')
    results['X vs. Y'] = results['X'] + ' vs. ' + results['Y']

    results = results.rename(columns={'comparison':'Comparison'})
    main_model_results = results[results['model'].isin(['ViT-L/14@336px'])]

    save_dir = os.path.join('results','plots','categorical_wna')
    os.makedirs(save_dir,exist_ok=True)
    order = results['X vs. Y'].unique()
    order.sort()
    hue_order = results['Comparison'].unique()
    hue_order.sort()
    for emotion, data in results.groupby('emotion'):
        main_model_props = {
            'medianprops': {'color': 'red','linewidth':3},
        }
        main_model_data = main_model_results[main_model_results['emotion'] == emotion]
        sns.boxplot(y='X vs. Y',x='effect_size', data=data, order=order,  hue='Comparison',
                   hue_order=hue_order)
        sns.boxplot(y='X vs. Y', x='effect_size', data=main_model_data, order=order, hue='Comparison',
                        hue_order=hue_order, **main_model_props)
        plt.axvline(0, c='black')
        # plt.title(emotion.title())
        plt.xlim(-2,2)
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir,emotion + '.jpg'))
        plt.clf()
        plt.close()


if __name__ == '__main__':
    plot_results()
