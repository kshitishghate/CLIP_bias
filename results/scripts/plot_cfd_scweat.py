import os

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import statsmodels.api as sm

from results.scripts.linear_reg_diagnostics import Linear_Reg_Diagnostic
from scipy.stats import boxcox


def load_results():
    results_path = os.path.join('results','data','cfd_scweat.csv')

    results = pd.read_csv(results_path)
    results['image_fp'] = results['image_fp'].str.replace(os.path.join('data', 'CFD Version 3.0', 'Images') + '/', '',
                                                          regex=False)

    results['image_set'] = results['image_fp'].str.split('/').str[0]

    results['model'] = np.where(results['image_set'].isin(['CFD','CFD-MR']),
                                results['image_name'].str.split('-').str[1] + '-' + results['image_name'].str.split('-').str[2],
                                results['image_name'].str.split('-').str[1] + results['image_name'].str.split('-').str[2] + '-' + results['image_name'].str.split('-').str[3])

    results['emotion'] = results['image_name'].str.replace('.jpg','',regex=False).str.split('-').str[-1]


    image_sets_and_codebooks = [
        ('CFD', {'':'CFD U.S. Norming Data'}),
        ('CFD-MR', {'': 'CFD-MR U.S. Norming Data'}),
        ('CFD-INDIA', {'united_states_': 'CFD-I U.S. Norming Data', 'india_':'CFD-I INDIA Norming Data'})
    ]
    datasets = {}
    for image_set, codebooks in image_sets_and_codebooks:
        dataset = results[results['image_set'] == image_set]
        for codebook_locale, codebook_sheet_name in codebooks.items():
            codebook = pd.read_excel(os.path.join('data', 'CFD Version 3.0', 'CFD 3.0 Norming Data and Codebook.xlsx'),
                                     sheet_name=codebook_sheet_name,
                                     header=7,
                                     engine='openpyxl')
            codebook = codebook[~codebook['Model'].isna()]
            codebook = codebook.rename(columns=lambda x: 'Model' if x == 'Model' else codebook_locale + x)
            #drop columns with nas
            codebook = codebook.drop(columns=[c for c in codebook.columns if codebook[c].isna().mean() > 0])
            dataset = dataset.merge(codebook, left_on='model',right_on = 'Model')
        datasets[image_set] = dataset

    return datasets

def get_all_column_names(image_set):
    all_results = load_results()
    return all_results[image_set].columns

def fit_regression(image_set, target, predictors, use_boxcox = True):
    all_results = load_results()
    data = all_results[image_set].copy()
    for c in data.columns:
        try:
            data[c] = data[c].astype(float)
        except ValueError:
            pass

    to_dummy = [c for c in predictors if data[c].dtype == 'O']
    numeric = [c for c in predictors if data[c].dtype != 'O']
    data = pd.concat([data[[target] + numeric]] + [pd.get_dummies(data[v], prefix=v, drop_first=True) for v in to_dummy],axis=1)
    if use_boxcox:
        indep, lmbda = boxcox(data[target] - data[target].min() + np.abs(data[target].min() * 0.00001))
        reg = sm.OLS(indep, sm.add_constant(data.drop(columns=target)))
    else:
        reg = sm.OLS(data[target], sm.add_constant(data.drop(columns=target)))
        lmbda = np.NaN

    res = reg.fit()
    diag = Linear_Reg_Diagnostic(res)

    return res, diag, lmbda


# What interactions exist between age, race, and gender?
res, diag, lmbda = fit_regression('CFD','association_score',['emotion', 'EthnicitySelf', 'GenderSelf', 'AgeRated'],use_boxcox=True)
res1, diag1, lmbda1 = fit_regression('CFD','association_score',['emotion', 'EthnicitySelf', 'GenderSelf', 'AgeRated', 'Happy'],use_boxcox=True)

# Do perceptions or self reports better explain WEAT scores?

# Do castes relate to pleasantness?