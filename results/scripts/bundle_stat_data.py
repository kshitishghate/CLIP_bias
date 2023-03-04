import os
import logging
import json

import numpy as np
import pandas as pd
from itertools import combinations
import xml.etree.ElementTree as ET

import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

from results.scripts.linear_reg_diagnostics import Linear_Reg_Diagnostic
from scipy.stats import boxcox, levene
import seaborn as sns
import matplotlib.pyplot as pld


global datasets
datasets = {}


def load_seat(test: str):
    with open(os.path.join('data', 'tests', test), 'r') as f:
        stimuli = json.load(f)

    stimuli = {
        'names': [stimuli['targ1']['category'], stimuli['targ2']['category'],
                  stimuli['attr1']['category'], stimuli['attr2']['category']],
        'X': stimuli['targ1']['examples'],
        'Y': stimuli['targ2']['examples'],
        'A': stimuli['attr1']['examples'],
        'B': stimuli['attr2']['examples']
    }
    assert(len(stimuli['X']) == len(stimuli['Y']))
    return stimuli

def load_results():
    if len(datasets) != 0:
        return datasets
    image_sets_and_codebooks = [
        ('CFD', {'':'CFD U.S. Norming Data'}),
        ('CFD-MR', {'': 'CFD-MR U.S. Norming Data'}),
        ('CFD-INDIA', {'united_states_': 'CFD-I U.S. Norming Data', 'india_':'CFD-I INDIA Norming Data'})
    ]

    results_path = os.path.join('results','data','cfd_7b_individual_dist.csv')

    results = pd.read_csv(results_path)

    a_stimuli = load_seat('sent-weat7b.jsonl')['X']
    b_stimuli = load_seat('sent-weat7b.jsonl')['Y']

    key_frame = pd.DataFrame(
        {'text_stimulus': a_stimuli + b_stimuli, 'category': ['Math'] * len(a_stimuli) + ['Arts'] * len(b_stimuli)})

    results = results.merge(key_frame, on='text_stimulus')

    results['image_fp'] = results['image_fp'].str.replace(os.path.join('data', 'CFD Version 3.0', 'Images') + '/', '',
                                                          regex=False)

    results['image_name'] = [os.path.basename(p) for p in results['image_fp']]

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

load_results()['CFD'].to_csv('results/data/processed_individual_distances.csv',index=False)

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

indep_rated_characteristics = [
    'Attractive', 'Babyfaced', 'Feminine', 'Masculine', 'Prototypic', 'Threatening', 'Unusual',
    'Warm', 'Competent', 'SocialStatus', #'Suitability',
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
