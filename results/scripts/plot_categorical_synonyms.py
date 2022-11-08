import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def plot_results():
    results = pd.read_csv(os.path.join('results','data','categorical_synonyms_results.csv'))
    results = results[~results['B'].str.contains('does not makes me')]
    results['context'] = np.where(
        results['A'].str.contains('a person who is conveying'), 'a person who is conveying',
    np.where(
        results['A'].str.contains('a person who is feeling'), 'a person who is feeling',
    np.where(
        results['A'].str.contains('a person who makes me feel'), 'a person who makes me feel',
        'None'
    )))
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

    results = results[results['context'] != 'a person who is conveying']
    for comparison in ['apathy', 'negation']:
        data = results[results['comparison'] == comparison].copy()


        data = data[data['X'] != 'White']
        data['X'] = data['X'].map(lambda x: 'European-American' if x == 'Euro' else x, na_action='ignore')
        data['Y'] = data['Y'].map(lambda x: 'Native-American' if x == 'Native' else x, na_action='ignore')
        data['X'] = data['X'].map(lambda x: 'Light-Skin' if x == 'Light' else x, na_action='ignore')
        data['Y'] = data['Y'].map(lambda x: 'Dark-Skin' if x == 'Dark' else x, na_action='ignore')
        data['X/Y'] = data['X'] + '/' + data['Y']
        data['Y/X'] = data['Y'] + '/' + data['X']
        data['Y/X - emotion'] = data['Y'] + '/' + data['X'] + ' - ' + data['emotion'].str.title()

        # Plot the
        a = data.groupby(['X/Y','emotion'])['effect_size'].mean().reset_index()
        a = a.pivot_table(index='X/Y',columns='emotion',values='effect_size')

        save_dir = os.path.join('results','plots','categorical_synonyms', comparison)
        os.makedirs(save_dir,exist_ok=True)
        order = data['Y/X'].unique()
        order.sort()
        for emotion, data in data.groupby('emotion'):
            sns.catplot(y='Y/X',x='effect_size', data=data, order=order, aspect=1.5, hue='context', kind='box')
            plt.axvline(0, c='black')
            # plt.title(emotion.title())
            plt.xlim(-2,2)
            # plt.tight_layout()

            plt.savefig(os.path.join(save_dir,emotion + '.jpg'))
            plt.clf()
            plt.close()


if __name__ == '__main__':
    plot_results()
