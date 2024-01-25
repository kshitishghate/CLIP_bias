import os

import xml.etree.ElementTree as ET

import torch
import nltk
import numpy as np
import pandas as pd

from scipy.special import comb
from references import open_clip
from tqdm import tqdm
from nltk.corpus import wordnet
from pattern.text.en import pluralize

import os


from measure_bias_and_performance.extract_clip import extract_images, extract_text, load_words_greenwald
from measure_bias_and_performance.sc_weat import SCWEAT, WEAT
from measure_bias_and_performance.utils import save_test_results


def test_already_run(test, file_name):
    if not os.path.exists(file_name):
        return False
    previous_results = pd.read_csv(file_name)
    relevant_results = previous_results[
        (previous_results['model'] == test['model'])
        & (previous_results['weat_num'] == int(test['weat_num']))
    ]
    return len(relevant_results) > 0


def get_all_image_paths():
    all_images = []
    for root, dir, files in os.walk(os.path.join('data','CFD Version 3.0', 'Images')):
        for f in files:
            if f.split('.')[-1] == 'jpg':
                all_images.append(os.path.join(root, f))
    return all_images


def perform_test(device):

    npermutations = 100000

    results_fp = os.path.join('results', 'data', 'weat_replication.csv')
    models = open_clip.list_pretrained()
    # Not using convnext_xxlarge because it is not supported by timm 0.6.12
    models = [m for m in models if m[0] != 'convnext_xxlarge']

    # models.reverse()
    total = len(models) * 10


    if os.path.exists(results_fp):
        model_name_strs = ['_'.join(m).replace('/','') for m in models]
        completed = pd.read_csv(results_fp)
        completed = completed['model'].isin(model_name_strs).astype(int)
        completed = completed.sum()
    else:
        completed = 0

    remaining = total - completed

    with tqdm(total=remaining) as pbar:
        for model_name in models:
            model, _, preprocess = open_clip.create_model_and_transforms(model_name[0], pretrained=model_name[1],
                                                                         device=device)
            tokenizer = open_clip.get_tokenizer(model_name[0])

            for weat_num in range(1, 11):
                stimuli = load_weat(weat_num)
                test = {'weat_num':str(weat_num),
                        'X': stimuli['names'][0],
                        'Y': stimuli['names'][1],
                        'A': stimuli['names'][2],
                        'B': stimuli['names'][3],
                        'nt': len(stimuli['X']),
                        'na': len(stimuli['A']),
                        'model': '_'.join(model_name).replace('/','')}

                if not test_already_run(test, results_fp):
                    np.random.seed(5718980)


                    embeddings = {
                        'X': extract_text(model, tokenizer, stimuli['X'], device, model_name),
                        'Y': extract_text(model, tokenizer, stimuli['Y'], device, model_name),
                        'A': extract_text(model, tokenizer, stimuli['A'], device, model_name),
                        'B': extract_text(model, tokenizer, stimuli['B'], device, model_name),
                    }


                    test_result = WEAT(embeddings['X'], embeddings['Y'], embeddings['A'], embeddings['B'])
                    test_result = pd.Series({
                        'effect_size':test_result.effect_size(),
                        'p_value': test_result.p(npermutations),
                        'npermutations':npermutations,
                    })

                    save_test_results(pd.concat([pd.Series(test), test_result]), results_fp)
                    pbar.update()


def load_weat(weat_num: int):
    with open(os.path.join('data','weat_words.txt'), 'r') as f:
        lines = f.readlines()
    l_start = [i for i, l in enumerate(lines) if l == f'WEAT{weat_num}\n'][0]
    lines = [l.replace('\n','') for l in lines[l_start+1:l_start+5]]

    stimuli = {
        'names': [l.split(':')[0] for l in lines],
        'X': lines[0].split(':')[1].split(','),
        'Y': lines[1].split(':')[1].split(','),
        'A': lines[2].split(':')[1].split(','),
        'B': lines[3].split(':')[1].split(',')
    }
    assert(len(stimuli['X']) == len(stimuli['Y']))
    assert(len(stimuli['A']) == len(stimuli['B']))
    return stimuli


if __name__ == '__main__':
    should_continue=True
    while should_continue:
        if os.path.exists(os.path.join('results','data','weat_replication.csv')):
            num_results_so_far = len(pd.read_csv(os.path.join('results','data','weat_replication.csv')))
        else:
            num_results_so_far = 0
        try:
            perform_test('cuda' if torch.cuda.is_available() else 'cpu')
        except:
            perform_test('cpu')
        if num_results_so_far == len(pd.read_csv(os.path.join('results','data','weat_replication.csv'))):
            should_continue = False
