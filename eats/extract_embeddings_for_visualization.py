import os
import json

import torch
import nltk
import numpy as np
import pandas as pd

from scipy.special import comb
from CLIP import clip
from tqdm import tqdm
from PIL import Image
import open_clip



from eats.extract_clip import load_images, extract_images, extract_text


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


def save_embeddings(test: pd.Series, model_name: str, modality: str, embeddings: dict, stimuli_names: dict):
    """Save the embeddings for a given test and modality to a csv file."""

    if modality == 'image':
        stimuli_names = (
                [os.path.basename(f) for f in stimuli_names['X']]
                + [os.path.basename(f) for f in stimuli_names['Y']]
                + [os.path.basename(f) for f in stimuli_names['A']]
                + [os.path.basename(f) for f in stimuli_names['B']]
        )
    elif modality == 'text':
        stimuli_names = (
            stimuli_names['X'] + stimuli_names['Y'] + stimuli_names['A'] + stimuli_names['B']
        )
    else:
        raise ValueError(f'Unknown modality: {modality}')

    metadata = pd.DataFrame({'image_test': test['Image Test'],
                             'text_test': test['Text Test'],
                             'model': model_name,
                             'modality': modality,
                             'stimuli_category': (
                                     [test['image_target_1']] * len(embeddings['X'])
                                     + [test['image_target_2']] * len(embeddings['Y'])
                                     + [test['image_attribute_1']] * len(embeddings['A'])
                                     + [test['image_attribute_2']] * len(embeddings['B'])
                             ),
                             'stimuli_name': stimuli_names
                             })
    embeddings = pd.concat([pd.DataFrame(e) for e in embeddings.values()])

    embeddings_fp = os.path.join('results', 'data', 'raw_embeddings', f'{model_name}.csv')
    metadata_fp = os.path.join('results', 'data',  'raw_embeddings', f'{model_name}_metadata.csv')

    if not os.path.exists(embeddings_fp):
        embeddings.to_csv(embeddings_fp, index=False, header=False, sep='\t')
        metadata.to_csv(metadata_fp, index=False, sep='\t')
    else:
        embeddings.to_csv(embeddings_fp, index=False, mode='a', header=False, sep='\t')
        metadata.to_csv(metadata_fp, index=False, mode='a', header=False, sep='\t')



def perform_test():
    all_tests = pd.read_csv(os.path.join('data', 'cross_modal_tests.csv'))

    models = open_clip.list_pretrained()[48:]
    # Not using convnext_xxlarge because it is not supported by timm 0.6.12
    models = [m for m in models if m[0] != 'convnext_xxlarge']
    total = len(models) * len(all_tests)

    # # clear embeddings directory
    # embeddings_dir = os.path.join('results', 'data', 'raw_embeddings')
    # if os.path.exists(embeddings_dir):
    #     for f in os.listdir(embeddings_dir):
    #         os.remove(os.path.join(embeddings_dir, f))
    # else:
    #     os.makedirs(embeddings_dir)

    with tqdm(total=total) as pbar:
        for model_name in models:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # device =  "mps"
            model, _, preprocess = open_clip.create_model_and_transforms(model_name[0], pretrained=model_name[1], device=device)
            tokenizer = open_clip.get_tokenizer(model_name[0])

            for i, test in all_tests.iterrows():

                np.random.seed(82804230)

                image_stimuli = {
                    'X': load_images(test['image_target_dir'], test['image_target_1']),
                    'Y': load_images(test['image_target_dir'], test['image_target_2']),
                    'A': load_images(test['image_attribute_dir'], test['image_attribute_1']),
                    'B': load_images(test['image_attribute_dir'], test['image_attribute_2'])
                }
                image_embeddings = {
                    'X': extract_images(model, preprocess, image_stimuli['X'], device, model_name),
                    'Y': extract_images(model, preprocess, image_stimuli['Y'], device, model_name),
                    'A': extract_images(model, preprocess, image_stimuli['A'], device, model_name),
                    'B': extract_images(model, preprocess, image_stimuli['B'], device, model_name),
                }
                save_embeddings(test, '_'.join(model_name).replace('/',''), 'image', image_embeddings, image_stimuli)

                text_stimuli = {
                    'X': load_seat(test['text_file'], test['text_target_1']),
                    'Y': load_seat(test['text_file'], test['text_target_2']),
                    'A': load_seat(test['text_file'], test['text_attribute_1']),
                    'B': load_seat(test['text_file'], test['text_attribute_2'])
                }
                text_embeddings = {
                    'X': extract_text(model, tokenizer, text_stimuli['X'], device, model_name),
                    'Y': extract_text(model, tokenizer, text_stimuli['Y'], device, model_name),
                    'A': extract_text(model, tokenizer, text_stimuli['A'], device, model_name),
                    'B': extract_text(model, tokenizer, text_stimuli['B'], device, model_name),
                }
                save_embeddings(test, '_'.join(model_name).replace('/',''), 'text', text_embeddings, text_stimuli)

                pbar.update()

if __name__ == '__main__':
    perform_test()

