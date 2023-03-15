import os

import xml.etree.ElementTree as ET

import torch
import nltk
import numpy as np
import pandas as pd
import json

from scipy.special import comb
import open_clip
from tqdm import tqdm
from nltk.corpus import wordnet
from pattern.text.en import pluralize

import os




from eats.extract_clip import extract_images, extract_text, load_words_greenwald
from eats.sc_weat import SCWEAT, WEAT
from eats.result_saving import save_test_results


def test_already_run(test, file_name):
    if not os.path.exists(file_name):
        return False
    previous_results = pd.read_csv(file_name)
    relevant_results = previous_results[
        (previous_results['model'] == test['model'])
        & (previous_results['test_name'] == test['test_name'])
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

    results_fp = os.path.join('results', 'data', 'seat_replication.csv')
    models = open_clip.list_pretrained()
    # Not using convnext_xxlarge because it is not supported by timm 0.6.12
    models = [m for m in models if m[0] != 'convnext_xxlarge']
    tests = [f for f in os.listdir(os.path.join('data','tests')) if f.split('.')[1] == 'jsonl']

    # models.reverse()
    total = len(models) * len(tests)


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

            for test_name in tests:
                try:
                    stimuli = load_seat(test_name)
                    test = {'test_name': str(test_name),
                            'X': stimuli['names'][0],
                            'Y': stimuli['names'][1],
                            'A': stimuli['names'][2],
                            'B': stimuli['names'][3],
                            'nt': len(stimuli['X']),
                            'na': len(stimuli['A']),
                            'model': '_'.join(model_name).replace('/','')}
                except AssertionError as e:
                    test = {'test_name': str(test_name),
                            'X': np.NaN,
                            'Y': np.NaN,
                            'A': np.NaN,
                            'B': np.NaN,
                            'nt': np.NaN,
                            'na': np.NaN,
                            'model': '_'.join(model_name).replace('/','')}


                if not test_already_run(test, results_fp):
                    if not test['X'] is np.NaN:
                        np.random.seed(5718980)


                        try:
                            embeddings = {
                                'X': extract_text(model, tokenizer, stimuli['X'], device, model_name),
                                'Y': extract_text(model, tokenizer, stimuli['Y'], device, model_name),
                                'A': extract_text(model, tokenizer, stimuli['A'], device, model_name),
                                'B': extract_text(model, tokenizer, stimuli['B'], device, model_name),
                            }
                            test_result = WEAT(embeddings['X'], embeddings['Y'], embeddings['A'], embeddings['B'])
                            test_result = pd.Series({
                                'effect_size': test_result.effect_size(),
                                'p_value': test_result.p(npermutations),
                                'npermutations': npermutations,
                            })
                        except RuntimeError as e:
                            test_result = pd.Series({}, dtype=str)

                    else:
                        test_result = pd.Series({}, dtype=str)
                    save_test_results(pd.concat([pd.Series(test), test_result]), results_fp)
                    pbar.update()


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


if __name__ == '__main__':
    should_continue=True
    while should_continue:
        if os.path.exists(os.path.join('results','data','seat_replication.csv')):
            num_results_so_far = len(pd.read_csv(os.path.join('results','data','seat_replication.csv')))
        else:
            num_results_so_far = 0
        try:
            perform_test('cuda' if torch.cuda.is_available() else 'cpu')
        except:
            perform_test('cpu')
        if num_results_so_far == len(pd.read_csv(os.path.join('results','data','seat_replication.csv'))):
            should_continue = False
