import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



def plot_results(excluded_positive_negative = True):
    results = pd.read_csv(os.path.join('results','data','categorical_results.csv'))
    results = results.drop_duplicates()
    results = results[results['X'] != 'White']
    results = results[excluded_positive_negative == results['positive_negative_excluded']].copy()
    results['X'] = results['X'].map(lambda x: 'European-American' if x == 'Euro' else x, na_action='ignore')
    results['Y'] = results['Y'].map(lambda x: 'Native-American' if x == 'Native' else x, na_action='ignore')
    results['X'] = results['X'].map(lambda x: 'Light-Skin' if x == 'Light' else x, na_action='ignore')
    results['Y'] = results['Y'].map(lambda x: 'Dark-Skin' if x == 'Dark' else x, na_action='ignore')
    results['X/Y'] = results['X'] + '/' + results['Y']
    results['Y/X'] = results['Y'] + '/' + results['X']
    results = results.rename(columns={'A': 'emotion'})
    results['Y/X - emotion'] = results['Y'] + '/' + results['X'] + ' - ' + results['emotion'].str.title()



    # Plot the
    a = results.groupby(['X/Y','emotion'])['effect_size'].mean().reset_index()
    a = a.pivot_table(index='X/Y',columns='emotion',values='effect_size')

    save_dir = os.path.join('results','plots','categorical',
                            'confounding_valence' if excluded_positive_negative else 'holding_valence_constant')
    os.makedirs(save_dir,exist_ok=True)
    order = results['Y/X'].unique()
    order.sort()
    for emotion, data in results.groupby('emotion'):
        sns.catplot(y='Y/X',x='effect_size', data=data, order=order, aspect=1.5)
        plt.axvline(0, c='black')
        plt.title(emotion.title())
        plt.xlim(-2,2)
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir,emotion + '.jpg'))
        plt.clf()
        plt.close()


if __name__ == '__main__':
    plot_results(True)
    plot_results(False)