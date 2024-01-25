import math
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from CLIP import clip

import open_clip
import pandas as pd
import numpy as np
import os

from measure_bias_and_performance.utils import cherti_et_al_models, profile_model


def load_seat_replication_results(models=None, openai_only=False):
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



def load_ieat_replication_results(models=None, openai_only=False):
    ieats = pd.read_csv(os.path.join('results','data','ieat_replication.csv'))

    openai_model_names = [m.replace('/','') for m in clip.available_models()] + clip.available_models()
    if openai_only:
        ieats = ieats[ieats['model'].isin(openai_model_names)]
    else:
        ieats = ieats[~ieats['model'].isin(openai_model_names)]

    if models is not None:
        ieats = ieats[ieats['model'].isin(models)]

    ieats['Test'] = ieats['Target'] + '/' + ieats['A'] + ' vs. ' + ieats['B']
    ieats['Test'] = ieats['Test'].str.replace('Pleasant vs. Unpleasant', 'Valence',regex=False)
    ieats['Test'] = ieats['Test'].str.replace('Weapon/Tool vs. Weapon', 'Race/Tool vs. Weapon',regex=False)
    ieats['Test'] = ieats['Test'].str.replace('Weapon/Tool-modern vs. Weapon-modern', 'Race/Tool vs. Weapon (Modern)',regex=False)
    ieats = ieats.sort_values([
        'Test'
    ])

    original_results = pd.read_csv(os.path.join('ieat','output','results.csv'))
    original_results = original_results.rename(columns={'d':'effect_size'})
    original_results['Test'] = original_results['Test'].replace(
        {'Disability':'Disabled',
         'Gender-Career':'Gender',
         'Gender-Science':'Gender',
         'Weapon-Race':'Race'
         }
    )
    original_results['Test'] = (original_results['Test'] + '/' + original_results['A'].str.title()
                                + ' vs. ' + original_results['B'].str.title())
    original_results['Test'] = original_results['Test'].replace({
        'Native/Us vs. World':'Native/US vs. World',
        'Weapon-Race (Modern)/Tool-Modern vs. Weapon-Modern':'Race/Tool vs. Weapon (Modern)'
    })
    original_results['Test'] = original_results['Test'].str.replace('Pleasant vs. Unpleasant', 'Valence',regex=False)
    original_results['Test'] = original_results['Test'].str.replace('Weapon/Tool vs. Weapon', 'Race/Tool vs. Weapon',regex=False)
    original_results['Test'] = original_results['Test'].str.replace('Weapon/Tool-modern vs. Weapon-modern', 'Race/Tool vs. Weapon (Modern)',regex=False)

    original_results = original_results[original_results['Test'].isin(ieats['Test'])]
    ieats = ieats[ieats['Test'].isin(original_results['Test'])]

    return ieats, original_results



def load_cross_modal_results(models=None, openai_only=False):
    """Load the cross modal results, potentially filtering by model name"""

    all_results = pd.read_csv(os.path.join('results','data','cross_modal.csv'))
    clip_model_names = [m.replace('/','') for m in clip.available_models()] + clip.available_models()
    if openai_only:
        all_results = all_results[all_results['model'].isin(clip_model_names)]
    else:
        all_results = all_results[~all_results['model'].isin(clip_model_names)]

    if models is not None:
        all_results = all_results[all_results['model'].isin(models)]

    all_results = all_results[['Image Test','Text Test','image_target_1',
           'image_target_2', 'image_attribute_1',
           'image_attribute_2', 'text_target_1', 'text_target_2',
           'text_attribute_1', 'text_attribute_2', 'order','effect_size','model']]

    averaged_results = all_results.groupby(['Image Test','Text Test','image_target_1',
           'image_target_2', 'image_attribute_1',
           'image_attribute_2', 'text_target_1', 'text_target_2',
           'text_attribute_1', 'text_attribute_2', 'order'])['effect_size'].mean().reset_index()

    averaged_results = averaged_results.pivot(
        index=[c for c in averaged_results.columns if c not in ['order','effect_size']],
        columns='order',
        values='effect_size').reset_index()


    all_results = all_results.pivot(
        index=[c for c in all_results.columns if c not in ['order','effect_size']],
        columns='order',
        values='effect_size').reset_index()


    ieat_results = pd.read_csv(os.path.join('results','data','ieat_replication.csv'))
    if openai_only:
        ieat_results = ieat_results[ieat_results['model'].isin(clip_model_names)]
    else:
        ieat_results = ieat_results[~ieat_results['model'].isin(clip_model_names)]

    if models is not None:
        ieat_results = ieat_results[ieat_results['model'].isin(models)]

    ieat_results['Test'] = ieat_results['Target'] + '/' + ieat_results['A'] + ' vs. ' + ieat_results['B']
    ieat_results['Test'] = ieat_results['Test'].str.replace('Pleasant vs. Unpleasant', 'Valence',regex=False)
    ieat_results['Test'] = ieat_results['Test'].str.replace('Weapon/Tool vs. Weapon', 'Race/Tool vs. Weapon',regex=False)
    ieat_results['Test'] = ieat_results['Test'].str.replace('Weapon/Tool-modern vs. Weapon-modern', 'Race/Tool vs. Weapon (Modern)',regex=False)
    ieat_results['Test'] = ieat_results['Test'].str.replace('Gender/Science vs. Liberal-Arts', 'Gender/Science vs. Arts',regex=False)
    ieat_results = ieat_results.rename(columns={'effect_size':'all_images'})

    all_results = all_results.merge(ieat_results, left_on=['Image Test', 'model'], right_on=['Test','model'])
    ieat_results = ieat_results.groupby(['Test','Target', 'X', 'Y', 'A', 'B', 'nt', 'na', 'Attribute'])['all_images'].mean().reset_index()

    averaged_results = averaged_results.merge(ieat_results, left_on='Image Test', right_on='Test')



    seat_results = pd.read_csv(os.path.join('results','data','seat_replication.csv'))
    if openai_only:
        seat_results = seat_results[seat_results['model'].isin(clip_model_names)]
    else:
        seat_results = seat_results[~seat_results['model'].isin(clip_model_names)]
    if models is not None:
        seat_results = seat_results[seat_results['model'].isin(models)]
    seat_results['Test'] = seat_results["test_name"].str.replace('.jsonl','',regex=False).str.replace('_',' ',regex=False).str.title()
    seat_results['Test'] = seat_results['Test'].str.strip()
    seat_results = seat_results[seat_results['Test'].str.contains('Weat')]
    seat_results = seat_results.rename(columns={'effect_size':'all_text'})
    seat_results['test_name'] = seat_results['test_name'].str.upper().str.replace('.JSONL','',regex=False)

    all_results = all_results.merge(seat_results, left_on=['Text Test','model'], right_on=['test_name','model'])
    seat_results = seat_results.groupby(['test_name', 'X', 'Y', 'A', 'B'])['all_text'].mean().reset_index()


    averaged_results = averaged_results.merge(seat_results, left_on='Text Test', right_on='test_name')

    all_results = all_results[['Text Test', 'Image Test', 'model', 'X_x','Y_x','A_x','B_x','all_images', 'text as target','image as target', 'all_text']].rename(columns= lambda x: x.replace(' ','_'))
    all_results['sign_flip'] = ((all_results['all_images'] >= 0) & (all_results['all_text'] < 0)) | ((all_results['all_text'] >= 0) & (all_results['all_images'] < 0))
    all_results['text_target_outside_range'] = (
        ((all_results['text_as_target'] > all_results['all_text']) & (all_results['text_as_target'] > all_results['all_images']))
        | ((all_results['text_as_target'] < all_results['all_text']) & (all_results['text_as_target'] < all_results['all_images']))
    )

    all_results['image_target_outside_range'] = (
        ((all_results['image_as_target'] > all_results['all_text']) & (all_results['image_as_target'] > all_results['all_images']))
        | ((all_results['image_as_target'] < all_results['all_text']) & (all_results['image_as_target'] < all_results['all_images']))
    )
    all_results['either_cross_outside_range'] = all_results['image_target_outside_range'] | all_results['text_target_outside_range']
    all_results['image_greater'] = all_results['all_images'] > all_results['all_text']
    all_results['cross_modal_dif'] = np.abs(all_results['image_as_target'] - all_results['text_as_target'])
    all_results['image_text_dif'] = np.abs(all_results['all_images'] - all_results['all_text'])

    return averaged_results, all_results
