import os

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import statsmodels.api as sm

from results.scripts.linear_reg_diagnostics import Linear_Reg_Diagnostic
from scipy.stats import boxcox

def get_word_list(word):
    # frequency data taken from: https://anc.org/data/anc-second-release/frequency-data/
    # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

    hierarchy = ET.parse(os.path.join('data','wn-domains-3.2','wn-affect-1.1','a-hierarchy.xml'))
    prev_parent_len = -1
    parents = [word]
    while len(parents) > prev_parent_len:
        prev_parent_len = len(parents)
        parents = [word] + [w.attrib['name'] for w in hierarchy.getroot() if 'isa' in w.attrib.keys() and w.attrib['isa'] in parents]

    np.random.shuffle(parents)
    return parents

def load_results(largest_model_only=True, only_emotions=True, exclude_negations=True,
                 templates_to_filter_to=None):
    results_path = os.path.join('results','data','wna_scweat.csv')

    results = pd.read_csv(results_path)
    if largest_model_only:
        results = results[results['model'] == 'ViT-L/14@336px'].copy()
    results['A/B'] = results['A'] + '/' + results['B']

    if only_emotions:
        results = results[results['word'].isin([w.replace('-', ' ') for w in get_word_list('emotion')])].copy()

    results['negation'] = results['template'].str.contains('not',case=False)
    if exclude_negations:
        results = results[~results['negation']]
        results = results.drop(columns='negation')

    results['template'] = results['template'].str.replace('an','a').str.replace('An','A')
    results['template_type'] = np.where(
        (results['template'] == '{}'), 'Simple',
        np.where((results['template'] == 'not {}.'), 'Simple Negation',
        np.where(((results['template'].str.contains('feel') | results['template'].str.contains('convey'))
            & results['template'].str.contains('not')
        ), 'Feeling Negation',
        np.where(((results['template'].str.contains('feel') | results['template'].str.contains('convey'))
         ), 'Feeling',
        np.where(results['template'].str.contains('not'), 'Semantically Bleached Negation',
        'Semantically Bleached'
                 )))))

    results['template_base'] = results['template'].str.replace('not ','')

    if templates_to_filter_to:
        results = results[results['template_type'].isin(templates_to_filter_to)]

    results = results.drop(columns=['npermutations','na'])

    emotion_maps = [{
        e: emotion_type for e in get_word_list(f'{emotion_type}-emotion')
    } for emotion_type in ['positive', 'negative', 'ambiguous', 'neutral']]

    emotion_map = {}
    for em in emotion_maps:
        for k, v in em.items():
            if k in emotion_map.keys():
                print(k, v)
            emotion_map[k] = v

    results['emotion_type'] = results['word'].map(emotion_map)

    emotion_list = ['joy', 'anger','anticipation','disgust','fear','surprise','trust']
    emotion_maps = [{
        e: emotion_type for e in get_word_list(f'{emotion_type}')
    } for emotion_type in emotion_list]
    emotion_map = {}
    for em in emotion_maps:
        for k, v in em.items():
            if k in emotion_map.keys():
                print(k, v)
            emotion_map[k] = v

    results['emotion'] = results['word'].map(emotion_map)


    return results

def fit_regression(B, variables, use_boxcox = True):
    all_results = load_results(largest_model_only=False)
    mf = all_results[all_results['B'] == B].copy()
    mf['emotion'] = mf.emotion.fillna('NA')
    mf['emotion_type'] = mf['emotion_type'].fillna('NA')
    mf['template_type'] = mf['template_type'].fillna('NA')
    mf = pd.concat([mf] + [pd.get_dummies(mf[v], prefix=v, drop_first=True) for v in variables],axis=1)
    mf = mf.drop(columns=['Title','A','B','w','template','A/B','p','emotion_type','emotion','model','template_type','template_base','word'])
    if use_boxcox:
        target, lmbda = boxcox(mf['association_score'] - mf['association_score'].min() + np.abs(mf['association_score'].min() * 0.00001))
        reg = sm.OLS(target, sm.add_constant(mf.drop(columns='association_score')))
    else:
        reg = sm.OLS(mf['association_score'], sm.add_constant(mf.drop(columns='association_score')))
        lmbda = np.NaN

    res = reg.fit()
    diag = Linear_Reg_Diagnostic(res)

    return res, diag, lmbda

res0, diag0, lmbda0 = fit_regression(B='Female',variables=['model','template_base'])
res1, diag1, lmbda1 = fit_regression(B='Female',variables=['model','template_base','emotion_type'])
res2, diag2, lmbda2 = fit_regression(B='Female',variables=['model','template_base','emotion_type','emotion'])
res3, diag3, lmbda3 = fit_regression(B='Female',variables=['model','template_base','emotion_type','emotion','word'])
diag2.residual_plot()
diag2.qq_plot()

diag2.qq_plot()
# Does it understand negation?
#       Are negated templates opposite their non negated ones?
results = load_results(only_emotions=True,exclude_negations=False)
a = results.groupby(['A/B','template_type','negation']).mean().reset_index()
b = results.groupby(['A/B','negation']).mean().reset_index()


# Does it understand the templates?
#       Are there differences across the templates?
# Are specific demographic groups more associated
results = load_results(only_emotions=True,exclude_negations=True)
a = results.groupby(['A/B','template_type']).mean().reset_index()



results = load_results(templates_to_filter_to=['Simple'])
results = results[results['emotion_type'].isin(['positive','negative'])]
a = results.groupby(['A/B','emotion_type','template']).mean().reset_index()


results = load_results(templates_to_filter_to=['Simple', 'Feeling'])
a = results.groupby(['A/B','emotion','template']).mean().reset_index()
