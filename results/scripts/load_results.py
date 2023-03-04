
import os

import numpy as np
import pandas as pd

def load_results(dataset):
    results = pd.read_csv(os.path.join('results','data',f'categorical_synonyms_{dataset}_results.csv'))
    results = results[results['model'].isin(['ViT-L/14@336px'])]
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




    results['X'] = results['X'].map(lambda x: 'European-American' if x == 'Euro' else x, na_action='ignore')
    results['Y'] = results['Y'].map(lambda x: 'Native-American' if x == 'Native' else x, na_action='ignore')
    results['X'] = results['X'].map(lambda x: 'Light-Skin' if x == 'Light' else x, na_action='ignore')
    results['Y'] = results['Y'].map(lambda x: 'Dark-Skin' if x == 'Dark' else x, na_action='ignore')
    results['X vs. Y'] = results['X'] + ' vs. ' + results['Y']
    results['Comparison Method (Context)'] = results['comparison'] + ' (' + results['context']  + ')'

    return results

