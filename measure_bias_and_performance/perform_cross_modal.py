import json
import os
import torch
import nltk
import numpy as np
import pandas as pd
import random
import sys
import argparse

from scipy.special import comb
import open_clip
from tqdm import tqdm
import logging as log
log.basicConfig(format='%(asctime)s: %(message)s', datefmt='%m/%d %I:%M:%S %p', level=log.INFO)

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from measure_bias_and_performance.extract_clip import load_images, extract_images, extract_text
from measure_bias_and_performance.sc_weat import WEAT
from measure_bias_and_performance.utils import test_already_run, save_test_results, cherti_et_al_ckpts, cherti_et_al_models

def load_seat(test: str, name: str):
    log.info('tests_nrc')
    with open(os.path.join('data', 'tests_nrc', test), 'r') as f:
        stimuli = json.load(f)

    accessible_stimuli = {
        stimuli['targ1']['category']: stimuli['targ1']['examples'],
        stimuli['targ2']['category']: stimuli['targ2']['examples'],
        stimuli['attr1']['category']: stimuli['attr1']['examples'],
        stimuli['attr2']['category']: stimuli['attr2']['examples']
    }
    assert(len(accessible_stimuli[stimuli['targ1']['category']]) == len(accessible_stimuli[stimuli['targ2']['category']]))
    return accessible_stimuli[name]

def perform_test(start_index, end_index, results_path):
    all_tests = pd.read_csv(os.path.join('data', 'cross_modal_controlled_no_weat_all_tests.csv'))
    models = open_clip.list_pretrained()
    # models = cherti_et_al_models()
    print(models)
    models = [m for m in models if m[0] != 'convnext_xxlarge']
    models = models[start_index:end_index]
    total = len(models) * len(all_tests) * 4

    results_fp = os.path.join('results', 'data_replication',results_path)
    if os.path.exists(results_fp):
        model_name_strs = ['_'.join(m).replace('/', '') for m in models]
        completed = pd.read_csv(results_fp)
        completed = completed['model'].isin(model_name_strs).astype(int)
        completed = completed.sum()
    else:
        completed = 0
    remaining = total - completed

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with tqdm(total=remaining, smoothing=0) as pbar:
        for model_name in models:
            model, preprocess, tokenizer = None, None, None

            for i, test in all_tests.iterrows():
                for order in ['image as target', 'text as target', 'all image', 'all text']:
                    test['order'] = order
                    if not test_already_run('_'.join(model_name).replace('/', ''), test, results_fp):
                        if model is None:
                        # If the model is not in open_clip, download it
                        # if 'epoch_' in model_name[1]:
                        #     download_intermediate_ckpt(model_name[1].replace('references/scaling-laws-openclip/', ''))

                        # device =  "mps"
                            try:
                                try:
                                    cache_dir = '/data/user_data/kghate/clip_bias'
                                    model_path = os.path.join(cache_dir, model_name[1])
                                    # if os.path.exists(model_path):
                                    if model is None:
                                        ## For cherti et al
                                        # model, _, preprocess = open_clip.create_model_and_transforms(model_name[0],
                                        #                                                              pretrained=model_path,
                                        #                                                              device=device,
                                        #                                                              cache_dir=cache_dir)
                                        # For 
                                        model, _, preprocess = open_clip.create_model_and_transforms(model_name[0],
                                                                                     pretrained=model_name[1],
                                                                                     device=device, cache_dir = '/data/user_data/kghate/clip_bias')
                                    else:
                                        model, _, preprocess = open_clip.create_model_and_transforms(model_name[0],
                                                                                                     pretrained=None,
                                                                                                     device=device)
                                        state_dict = torch.load(model_path, map_location=device)
                                        model.load_state_dict(state_dict)
                                except Exception as e:
                                    print(f"Error loading model: {e}")
                                    raise
                            except RuntimeError as e:
                                if "Model config" in str(e):
                                    print(f"Skipping model {model_name[0]} due to config not found.")
                                    continue
                                else:
                                    raise e
                            tokenizer = open_clip.get_tokenizer(model_name[0])

                        np.random.seed(82804230)

                        if order == 'image as target':
                            target_loader = load_images
                            target_extractor = extract_images
                            target_location = test['image_target_dir']
                            attribute_loader = load_seat
                            attribute_extractor = extract_text
                            attribute_location = test['text_file']
                            x = test['image_target_1']
                            y = test['image_target_2']
                            a = test['text_attribute_1']
                            b = test['text_attribute_2']
                        elif order == 'text as target':
                            target_loader = load_seat
                            target_extractor = extract_text
                            target_location = test['text_file']
                            attribute_loader = load_images
                            attribute_extractor = extract_images
                            attribute_location = test['image_attribute_dir']
                            x = test['text_target_1']
                            y = test['text_target_2']
                            a = test['image_attribute_1']
                            b = test['image_attribute_2']
                        elif order == 'all image':
                            target_loader = load_images
                            target_extractor = extract_images
                            target_location = test['image_target_dir']
                            attribute_loader = load_images
                            attribute_extractor = extract_images
                            attribute_location = test['image_attribute_dir']
                            x = test['image_target_1']
                            y = test['image_target_2']
                            a = test['image_attribute_1']
                            b = test['image_attribute_2']
                        elif order == 'all text':
                            target_loader = load_seat
                            target_extractor = extract_text
                            target_location = test['text_file']
                            attribute_loader = load_seat
                            attribute_extractor = extract_text
                            attribute_location = test['text_file']
                            x = test['text_target_1']
                            y = test['text_target_2']
                            a = test['text_attribute_1']
                            b = test['text_attribute_2']

                        stimuli = {
                            'X': target_loader(target_location, x),
                            'Y': target_loader(target_location, y),
                            'A': attribute_loader(attribute_location, a),
                            'B': attribute_loader(attribute_location, b)
                        }
                        test['nt'] = len(stimuli['X'])
                        test['naa'] = len(stimuli['A'])
                        test['nab'] = len(stimuli['B'])

                        embeddings = {
                            'X': target_extractor(model, preprocess,tokenizer, stimuli['X'], device, model_name),
                            'Y': target_extractor(model, preprocess,tokenizer, stimuli['Y'], device, model_name),
                            'A': attribute_extractor(model, preprocess,tokenizer, stimuli['A'], device, model_name),
                            'B': attribute_extractor(model, preprocess,tokenizer, stimuli['B'], device, model_name),
                        }

                        npermutations = min(
                            1000000,
                            int(comb(len(embeddings['X']) + len(embeddings['Y']), len(embeddings['X']))),
                        )

                        test_result = WEAT(embeddings['X'], embeddings['Y'], embeddings['A'], embeddings['B'])
                        test_result = pd.Series({
                            'model': model_name,
                            'pvalue': test_result.p(npermutations),
                            'effect_size': test_result.effect_size(),
                            'npermutations': npermutations,
                        })

                        save_test_results(pd.concat([test, test_result]), results_fp)
                        pbar.update()

            test_already_run('_'.join(model_name).replace('/', ''), test, results_fp, hard_reload=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run cross-modal tests on a range of models.')
    parser.add_argument('start_index', type=int, help='Starting index for the range of models')
    parser.add_argument('end_index', type=int, help='Ending index for the range of models')
    parser.add_argument('results_path', type=str, help='Path to store results')
    args = parser.parse_args()

    perform_test(args.start_index, args.end_index, args.results_path)
