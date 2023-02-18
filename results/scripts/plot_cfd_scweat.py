import os
import logging

import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from results.scripts.linear_reg_diagnostics import Linear_Reg_Diagnostic
from scipy.stats import boxcox, levene


global datasets
datasets = {}

logger = logging.getLogger()
logger.setLevel(logging.INFO)
model_log_path = os.path.join('results','logs','cfd_scweat.log')
fh = logging.FileHandler(model_log_path)
formatter = logging.Formatter(
    '%(asctime)s (%(levelname)s): %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


def load_results():
    if len(datasets) != 0:
        return datasets
    image_sets_and_codebooks = [
        ('CFD', {'':'CFD U.S. Norming Data'}),
        ('CFD-MR', {'': 'CFD-MR U.S. Norming Data'}),
        ('CFD-INDIA', {'united_states_': 'CFD-I U.S. Norming Data', 'india_':'CFD-I INDIA Norming Data'})
    ]

    results_path = os.path.join('results','data','cfd_scweat.csv')

    results = pd.read_csv(results_path)
    results['image_fp'] = results['image_fp'].str.replace(os.path.join('data', 'CFD Version 3.0', 'Images') + '/', '',
                                                          regex=False)

    results['image_set'] = results['image_fp'].str.split('/').str[0]

    results['model_id'] = np.where(results['image_set'].isin(['CFD','CFD-MR']),
                                results['image_name'].str.split('-').str[1] + '-' + results['image_name'].str.split('-').str[2],
                                results['image_name'].str.split('-').str[1] + results['image_name'].str.split('-').str[2] + '-' + results['image_name'].str.split('-').str[3])

    results['emotion'] = results['image_name'].str.replace('.jpg','',regex=False).str.split('-').str[-1]

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
            dataset = dataset.merge(codebook, left_on='model_id',right_on = 'Model')
        datasets[image_set] = dataset

    return datasets

def get_all_column_names(image_set):
    all_results = load_results()
    return all_results[image_set].columns


def create_interaction_frame(cat1, cat2):
    cat1_levels = cat1.unique()
    cat2_levels = cat2.unique()
    indicators = []
    for c1l in cat1_levels:
        for c2l in cat2_levels:
            indicators.append(
                pd.Series(((cat1 == c1l) & (cat2 == c2l)).astype(int),
                          name=f'{cat1.name}[{c1l}]&{cat2.name}[{c2l}]')
            )
    return pd.concat(indicators, axis=1)

def fit_regression(image_set, target, predictors, interactions=None, cubic_terms=None, threeway_interactions=None,
                   squared_terms=None, baselines=None, weights=None, use_boxcox = True,
                   cov_type='nonrobust'):
    all_results = load_results()
    data = all_results[image_set].copy()
    predictors = [p for p in predictors if p in data.columns]
    for c in data.columns:
        try:
            data[c] = data[c].astype(float)
        except ValueError:
            pass


    formula_start = '0 + model +' if 'model' in predictors else ''
    predictors = [p for p in predictors if p != 'model']

    to_dummy = [c for c in predictors if data[c].dtype == 'O' and c != 'model']
    for c in to_dummy:
        if baselines is not None and c in baselines.keys():
            data[c] = data[c].replace(baselines[c], '0'+baselines[c])

    if use_boxcox:
        data[target+'_transformed'], lmbda = boxcox(data[target] - data[target].min() + np.abs(data[target].min() * 0.00001))
        target = target + '_transformed'
    else:
        lmbda = np.NaN


    if interactions is not None:
        predictors = [p for p in predictors
                      if p not in [i[0] for i in interactions]
                         and p not in [i[1] for i in interactions]]

    formula = (f'{target} ~ {formula_start} '
               + ' + '.join(predictors))

    if squared_terms is not None:
        formula += ' + ' + ' + '.join([f'I({a}**2)' for a in squared_terms])

    if cubic_terms is not None:
        formula += ' + ' + ' + '.join([f'I({a}**3)' for a in squared_terms])

    if interactions is not None:
        formula += ' + ' + ' + '.join([f'{a}*{b}' for a, b in interactions])

    if threeway_interactions is not None:
        formula += ' + ' + ' + '.join([f'{a}*{b}*{c}' for a, b, c in threeway_interactions])

    if weights is None:
        reg = smf.ols(formula, data)
    else:
        reg = smf.wls(formula, data, weights=weights)


    res = reg.fit(cov_type=cov_type)
    diag = Linear_Reg_Diagnostic(res)


    return res, diag, lmbda, data



def fit_wls(num_iterations=None, *args, **kwargs):
    assert num_iterations > 1
    weighted_model, weighted_diag, weighted_lmbda, weighted_data = fit_regression(*args, **kwargs)

    for i in range(num_iterations - 1):
        predictors = sm.add_constant(
            pd.DataFrame({
                'Y_hat': weighted_model.fittedvalues,
                'Y_hat_sq': weighted_model.fittedvalues ** 2
            })
        )

        fitted_sds = sm.OLS(np.abs(weighted_model.resid), predictors).fit().fittedvalues
        weights2 = 1 / (fitted_sds ** 2)
        weighted_model, weighted_diag, weighted_lmbda, weighted_data = fit_regression(weights=weights2, *args, **kwargs)

    return weighted_model, weighted_diag, weighted_lmbda, weighted_data



# Identify columns
face_and_photo_info = [
    'LuminanceMedian', 'NoseWidth', 'NoseLength', 'LipThickness',
    'FaceLength',
    'EyeHeightR', 'EyeHeightL', #'EyeHeightAvg',
    'EyeWidthR', 'EyeWidthL', # 'EyeWidthAvg',
    'FaceWidthCheeks', 'FaceWidthMouth',
    'FaceWidthBZ', 'Forehead', 'UpperFaceLength2',
    'PupilTopR', 'PupilTopL', #'PupilTopAsymmetry',
    'PupilLipR', 'PupilLipL', # 'PupilLipAvg', 'PupilLipAsymmetry',
    'BottomLipChin',
    'MidcheekChinR', 'MidcheekChinL', # 'CheeksAvg',
    'MidbrowHairlineR', 'MidbrowHairlineL', # 'MidbrowHairlineAvg',
    'FaceShape', 'Heartshapeness', 'NoseShape',
    'LipFullness', 'EyeShape', 'EyeSize', 'UpperHeadLength',
    'MidfaceLength', 'ChinLength', 'ForeheadHeight', 'CheekboneHeight',
    'CheekboneProminence', 'FaceRoundness','fWHR2',
    'EyeDistance',
    'FaceColorRed', 'FaceColorGreen', 'FaceColorBlue',
    'HairColorRed', 'HairColorGreen', 'HairColorBlue',
    'HairLuminance', 'EyeLuminanceR', 'EyeLuminanceL',
    'EyeBrowThicknessR', 'EyeBrowThicknessL', # 'EyeBrowThicknessAvg',
    'EyeLidThicknessR', 'EyeLidThicknessL',# 'EyeLidThicknessAvg'
]

indep_rated_demo = [
    'MaleProb', #'FemaleProb',
    'BlackProb', 'LatinoProb', 'WhiteProb', 'OtherProb', 'MultiProb', #'AsianProb',
    'ChineseAsianProb', 'JapaneseAsianProb', 'IndianAsianProb', 'OtherAsianProb', 'MiddleEasternProb'
]

self_reported_demo = [
    'EthnicitySelf', 'GenderSelf'
]

self_reported_age = [
    'AgeSelf'
]

indep_rated_age = [
    'AgeRated'
]

indep_rated_emotions = [
    'Afraid', 'Angry','Disgusted','Dominant',  'Happy', 'Sad', 'Surprised', 'Trustworthy'
]

logger.info('We do not include suitability as it is not available for all images in CFD')
indep_rated_characteristics = [
    'Attractive', 'Babyfaced', 'Feminine', 'Masculine', 'Prototypic', 'Threatening', 'Unusual',
    'Warm', 'Competent', 'SocialStatus', 'Suitability',
]

ancestry_columns = [
    'AncestryPaternalGFatherSelf',
    'AncestryPaternalGMotherSelf',
    'AncestryMaternalGFatherSelf',
    'AncestryMaternalGMotherSelf',
    'AncestryFatherSelf',
    'AncestryMotherSelf',
    'AncestrySelf'
]

self_reported_emotion = [
    'emotion'
]

# Textbook: https://users.stat.ufl.edu/~winner/sta4211/ALSM_5Ed_Kutner.pdf
res, diag, lmbda, data = fit_regression('CFD','association_score',self_reported_demo + indep_rated_demo, interactions=None, use_boxcox=False)
a = data.corr()
logger.info('We do not use third-party rated race or gender info in the main CFD set, due to the high '
            'correlation with self-reported race and gender. For example, when including both terms in a regression to '
            'predict SC-WEAT, we find variance inflation factors significantly above 10.')


# Do race, gender, and age account for much of the variance?
full_model, full_diag, full_lmbda, full_data = fit_regression('CFD','association_score', ['model'] + self_reported_emotion, interactions=None, use_boxcox=False)

emotions = datasets['CFD']['emotion'].unique()
emotion_resids = [
    full_model.resid[datasets['CFD']['emotion'] == e] for e in emotions
]
levene_results = levene(*emotion_resids)
logger.info('When fitting a model based on expressed emotion, we find evidence for non-constant error variance. We '
            'perform a Brown-Forsythe Test (AKA modified Levene Test), for residuals from photos with differing '
            f'emotions, and find evidence that variance in residuals is not equal between groups '
            f'(W={levene_results.statistic}, p={levene_results.pvalue}')


full_model, full_diag, full_lmbda, full_data = fit_regression('CFD','association_score', ['model'] + self_reported_emotion, interactions=None, use_boxcox=True)

emotions = datasets['CFD']['emotion'].unique()
emotion_resids = [
    full_model.resid[datasets['CFD']['emotion'] == e] for e in emotions
]

levene_results = levene(*emotion_resids)
logger.info('We find that this issue is not corrected by using a Box-Cox transformation of the independent variable, ' 
            'and for this reason we elect to use weighted least squares '
            f'(W={levene_results.statistic}, p={levene_results.pvalue}')


# Estimate weights!
full_model, full_diag, full_lmbda, full_data = fit_regression(
    'CFD', 'association_score',
    ['model'] + self_reported_emotion + self_reported_demo + indep_rated_age,
    use_boxcox=False,
    interactions=[('GenderSelf', 'EthnicitySelf'), ('GenderSelf', 'AgeRated'), ('EthnicitySelf','AgeRated'),
                  ('model','GenderSelf'), ('model','EthnicitySelf'), ('model','AgeRated')
                  ],
    threeway_interactions=[('EthnicitySelf','AgeRated','GenderSelf')],
    cov_type='HC3',
    squared_terms=['AgeRated'],
    # cubic_terms=['AgeRated'],
    baselines={'GenderSelf': 'M',
             'EthnicitySelf': 'W',
             'emotion': 'N'}
)


merge_sets = [pd.DataFrame({'I':1,'AgeRated':[datasets['CFD']['AgeRated'].min(), datasets['CFD']['AgeRated'].max()]}),
              pd.DataFrame({'I':1,'GenderSelf':['0M','F']}),
              pd.DataFrame({'I':1,'model':datasets['CFD']['model'].unique()}),
              pd.DataFrame({'I':1,'emotion':['0N','A','HO','HC','F']}),
              pd.DataFrame({'I':1,'EthnicitySelf':['0W','L','B','A']})]
crossed = pd.merge(merge_sets[0], merge_sets[1], on='I')
for i in range(len(merge_sets) - 2):
    crossed = pd.merge(crossed, merge_sets[i + 2], on='I')
crossed = crossed.drop(columns='I')
crossed['predictions'] = full_model.predict(crossed)


min_age = crossed[crossed['AgeRated'] == crossed['AgeRated'].min()]
max_age = crossed[crossed['AgeRated'] == crossed['AgeRated'].max()]
ages = pd.merge(min_age,max_age, on = ['model','emotion','EthnicitySelf','GenderSelf'])
ages['dif'] = ages['predictions_x'] - ages['predictions_y']


min_gender = crossed[crossed['GenderSelf'] == crossed['GenderSelf'].min()]
max_gender = crossed[crossed['GenderSelf'] == crossed['GenderSelf'].max()]
genders = pd.merge(min_gender,max_gender, on = ['model','emotion','EthnicitySelf','AgeRated'])
genders['dif'] = genders['predictions_y'] - genders['predictions_x']


aa = crossed[crossed['EthnicitySelf'] == 'B']
ea = crossed[crossed['EthnicitySelf'] == '0W']
aaea = pd.merge(aa, ea, on = ['model','emotion','GenderSelf','AgeRated'])
aaea['dif'] = aaea['predictions_y'] - aaea['predictions_x']


sns.regplot(x=data['AgeRated'], y=full_model.resid,lowess=True, line_kws={'color':'red'})
plt.axhline(0, c='green')
# Non-constant variance: Using HC3 covariance
# Non-linearity: The variance is non-constant, so I don't believe an F-Test for lack of
#                fit is possible. However, plotting the fitted values vs. the residuals does not indicate any
#                significant departures from linearity.
full_diag.residual_plot()





# What interactions exist between age, race, and gender?


# Do perceptions or self reports better explain WEAT scores? Do they result in more


# Do racial backgrounds affect association scores?
