import os
import json

import torch
import nltk
import numpy as np
import pandas as pd

from scipy.special import comb
import open_clip
from tqdm import tqdm



from eats.extract_clip import load_images, extract_images, extract_text
from ieat.weat.weat.test import Test
from eats.result_saving import test_already_run, save_test_results


def load_seat(test: str, name: str):
    with open(os.path.join('data', 'tests', test), 'r') as f:
        stimuli = json.load(f)

    accessible_stimuli = {
        stimuli['targ1']['category']: stimuli['targ1']['examples'],
        stimuli['targ2']['category']: stimuli['targ2']['examples'],
        stimuli['attr1']['category']: stimuli['attr1']['examples'],
        stimuli['attr2']['category']: stimuli['attr2']['examples']
    }
    assert(len(accessible_stimuli[stimuli['targ1']['category']]) == len(accessible_stimuli[stimuli['targ2']['category']]))
    return accessible_stimuli[name]

def perform_test():
    all_tests = pd.read_csv(os.path.join('data', 'cross_modal_tests.csv'))

    models = open_clip.list_pretrained()
    # Not using convnext_xxlarge because it is not supported by timm 0.6.12
    models = [m for m in models if m[0] != 'convnext_xxlarge']

    total = len(models) * len(all_tests) * 2
    results_fp = os.path.join('results', 'data', 'cross_modal.csv')
    if os.path.exists(results_fp):
        model_name_strs = ['_'.join(m).replace('/', '') for m in models]
        completed = pd.read_csv(results_fp)
        completed = completed['model'].isin(model_name_strs).astype(int)
        completed = completed.sum()
    else:
        completed = 0
    remaining = total - completed

    with tqdm(total=remaining) as pbar:
        for model_name in models:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # device =  "mps"
            model, _, preprocess = open_clip.create_model_and_transforms(model_name[0], pretrained=model_name[1],
                                                                         device=device)
            tokenizer = open_clip.get_tokenizer(model_name[0])

            for i, test in all_tests.iterrows():
                for order in ['image as target','text as target']:
                    test['order'] = order
                    if not test_already_run('_'.join(model_name).replace('/', ''), test, results_fp):
                        np.random.seed(82804230)

                        if order == 'image as target':
                            target_loader = load_images
                            target_extractor = extract_images
                            target_location = test['image_target_dir']
                            target_preprocessor = preprocess
                            attribute_loader = load_seat
                            attribute_extractor = extract_text
                            attribute_location = test['text_file']
                            attribute_preprocessor = tokenizer
                            x = test['image_target_1']
                            y = test['image_target_2']
                            a = test['text_attribute_1']
                            b = test['text_attribute_2']
                        elif order == 'text as target':
                            target_loader = load_seat
                            target_extractor = extract_text
                            target_location = test['text_file']
                            target_preprocessor = tokenizer
                            attribute_loader = load_images
                            attribute_extractor = extract_images
                            attribute_location = test['image_attribute_dir']
                            attribute_preprocessor = preprocess
                            x = test['text_target_1']
                            y = test['text_target_2']
                            a = test['image_attribute_1']
                            b = test['image_attribute_2']

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
                            'X': target_extractor(model, target_preprocessor, stimuli['X'], device, model_name),
                            'Y': target_extractor(model, target_preprocessor, stimuli['Y'], device, model_name),
                            'A': attribute_extractor(model, attribute_preprocessor, stimuli['A'], device, model_name),
                            'B': attribute_extractor(model, attribute_preprocessor, stimuli['B'], device, model_name),
                        }

                        npermutations = min(
                            1000000,
                            int(comb(len(embeddings['X']) + len(embeddings['Y']), len(embeddings['X']))),
                        )

                        test_result = Test(embeddings['X'], embeddings['Y'], embeddings['A'], embeddings['B'])
                        test_result = pd.Series({
                            'model':'_'.join(model_name).replace('/', ''),
                            'pvalue':test_result.p(npermutations),
                            'effect_size':test_result.effect_size(),
                            'npermutations':npermutations,
                        })

                        save_test_results(pd.concat([test, test_result]), results_fp)
                        pbar.update()

if __name__ == '__main__':
    perform_test()