import os
import json

import torch
import nltk
import numpy as np
import pandas as pd

from scipy.special import comb
from references.CLIP import clip
from tqdm import tqdm
from PIL import Image
from references import open_clip

from measure_bias_and_performance.utils import cherti_et_al_models
from results.analysis_scripts.bundle_eat_data import load_model_info

global completed_tests
completed_tests = None
global attr_completed_tests
attr_completed_tests = None


from measure_bias_and_performance.extract_clip import load_images, extract_images, extract_text


def load_seat(test: str):
    with open(os.path.join('data', 'tests', test), 'r') as f:
        stimuli = json.load(f)

    stimuli = {
        'names': [
            stimuli['targ1']['category'],
            stimuli['targ2']['category'],
            stimuli['attr1']['category'],
            stimuli['attr2']['category']
        ],
        'X': stimuli['targ1']['examples'],
        'Y': stimuli['targ2']['examples'],
        'A': stimuli['attr1']['examples'],
        'B': stimuli['attr2']['examples']
    }
    assert(len(stimuli['X']) == len(stimuli['Y']))
    return stimuli

def load_attributes():
    all_attributes = []
    for dataset in ['Flickr30k', 'VisualGenome']:
        cat_dir = os.path.join('data', 'LabelingPeople', dataset, 'resources', 'categories')
        categories = os.listdir(cat_dir)
        for cat in categories:
            save_path = os.path.join(cat_dir, cat)
            attributes = pd.read_csv(save_path, header=None, comment='#')
            attributes['category'] = cat.replace('.txt','')
            all_attributes.append(attributes)
    all_attributes = pd.concat(all_attributes)
    all_attributes.columns = ['attribute', 'category']
    all_attributes = all_attributes.drop_duplicates()

    all_attributes['root_word'] = all_attributes['attribute']
    sentence_attributes = all_attributes.copy()
    sentence_attributes['attribute'] = sentence_attributes['attribute'].apply(lambda x:  ('A person who is ' + x + '.').replace('..','.'))

    all_attributes = pd.concat([all_attributes, sentence_attributes]).reset_index(drop=True)
    return all_attributes


def extract_image_embd(model, preprocess, test, device, model_name):
    np.random.seed(82804230)

    attr_folder = test['Attribute'] if test['Attribute'] == 'Valence' else test['Target']

    stimuli = {
        'X': load_images(test['Target'], test['X']),
        'Y': load_images(test['Target'], test['Y']),
        'A': load_images(attr_folder, test['A']),
        'B': load_images(attr_folder, test['B'])
    }
    assert len(stimuli['X']) == test['nt'] and len(stimuli['Y']) == test['nt']
    assert len(stimuli['A']) == test['na'] and len(stimuli['B']) == test['na']

    embeddings = {
        'X': extract_images(model, preprocess, stimuli['X'], device, model_name),
        'Y': extract_images(model, preprocess, stimuli['Y'], device, model_name),
        'A': extract_images(model, preprocess, stimuli['A'], device, model_name),
        'B': extract_images(model, preprocess, stimuli['B'], device, model_name),
    }
    save_embeddings(test, model_name, 'image', embeddings, stimuli['X'] + stimuli['Y'] + stimuli['A'] + stimuli['B'])


def save_embeddings(test: pd.Series, model_name: str, modality: str, embeddings: dict, stimuli_names: dict):
    """Save the embeddings for a given test and modality to a csv file."""
    model_name = '_'.join(model_name).replace('/', '')

    if modality == 'image':
        test_name = test['X'] + '_' + test['Y'] + '_' + test['A'] + '_' + test['B']
        metadata = pd.DataFrame({'test': test_name,
                                 'model': model_name,
                                 'modality': modality,
                                 'stimuli_category': (
                                         [test['X']] * len(embeddings['X'])
                                         + [test['Y']] * len(embeddings['Y'])
                                         + [test['A']] * len(embeddings['A'])
                                         + [test['B']] * len(embeddings['B'])
                                 ),
                                 'stimuli_name': stimuli_names
                                 })
    elif modality == 'text':
        test_name = test['test_name']
        metadata = pd.DataFrame({'test': test_name,
                                 'model': model_name,
                                 'modality': modality,
                                 'stimuli_category': (
                                         [test['X']] * len(embeddings['X'])
                                         + [test['Y']] * len(embeddings['Y'])
                                         + [test['A']] * len(embeddings['A'])
                                         + [test['B']] * len(embeddings['B'])
                                 ),
                                 'stimuli_name': stimuli_names
                                 })
    else:
        raise ValueError(f'Unknown modality: {modality}')


    embeddings = pd.concat([pd.DataFrame(e) for e in embeddings.values()])

    embeddings_fp = os.path.join('results', 'data', 'raw_embeddings', f'{model_name}.csv')
    metadata_fp = os.path.join('results', 'data',  'raw_embeddings', f'{model_name}_metadata.csv')

    global completed_tests

    completed_tests.update((model_name, test_name, modality))

    if not os.path.exists(embeddings_fp):
        os.makedirs(os.path.dirname(embeddings_fp), exist_ok=True)
        embeddings.to_csv(embeddings_fp, index=False, header=False, sep='\t')
        metadata.to_csv(metadata_fp, index=False, sep='\t')
    else:
        embeddings.to_csv(embeddings_fp, index=False, mode='a', header=False, sep='\t')
        metadata.to_csv(metadata_fp, index=False, mode='a', header=False, sep='\t')


def save_attr_embeddings(attr, embeddings, model_name):
    attr = attr.copy()
    model_name = '_'.join(model_name).replace('/', '')
    attr['model'] = model_name


    embeddings_fp = os.path.join('results', 'data', 'raw_embeddings', f'{model_name}_attr.csv')
    metadata_fp = os.path.join('results', 'data',  'raw_embeddings', f'{model_name}_attr_metadata.csv')

    global attr_completed_tests
    if attr_completed_tests is None:
        attr_completed_tests = set()
    attr_completed_tests.update([model_name])
    if not os.path.exists(embeddings_fp):
        os.makedirs(os.path.dirname(embeddings_fp), exist_ok=True)
        embeddings.to_csv(embeddings_fp, index=False, header=False, sep='\t')
        attr.to_csv(metadata_fp, index=False, sep='\t')
    else:
        embeddings.to_csv(embeddings_fp, index=False, mode='a', header=False, sep='\t')
        attr.to_csv(metadata_fp, index=False, mode='a', header=False, sep='\t')



def extract_text_embd(model, tokenizer, test_name, model_name, device):
    try:
        stimuli = load_seat(test_name)
        test = {'test_name': str(test_name),
                'X': stimuli['names'][0],
                'Y': stimuli['names'][1],
                'A': stimuli['names'][2],
                'B': stimuli['names'][3],
                'nt': len(stimuli['X']),
                'na': len(stimuli['A']),
                'model': '_'.join(model_name).replace('/', '')}
    except AssertionError as e:
        test = {'test_name': str(test_name),
                'X': np.NaN,
                'Y': np.NaN,
                'A': np.NaN,
                'B': np.NaN,
                'nt': np.NaN,
                'na': np.NaN,
                'model': '_'.join(model_name).replace('/', '')}

    if not test['X'] is np.NaN:
        np.random.seed(5718980)

        embeddings = {
            'X': extract_text(model, tokenizer, stimuli['X'], device, model_name),
            'Y': extract_text(model, tokenizer, stimuli['Y'], device, model_name),
            'A': extract_text(model, tokenizer, stimuli['A'], device, model_name),
            'B': extract_text(model, tokenizer, stimuli['B'], device, model_name),
        }
    save_embeddings(test, model_name, 'text', embeddings, stimuli['X'] + stimuli['Y'] + stimuli['A'] + stimuli['B'])


def needs_completion(model_name, test, modality):
    if modality == 'image':
        test = test['X'] + '_' + test['Y'] + '_' + test['A'] + '_' + test['B']
    global completed_tests
    model_name = '_'.join(model_name).replace('/', '')
    if completed_tests is None:
        save_dir = os.path.join('results', 'data', 'raw_embeddings')
        os.makedirs(save_dir, exist_ok=True)
        embedding_paths = [f for f in os.listdir(save_dir)
                           if f.split('.')[-1] == 'csv' and 'metadata' in f and not 'attr' in f]
        completed_embeddings = [pd.read_csv(os.path.join('results', 'data', 'raw_embeddings', f), sep='\t') for f in embedding_paths]
        if len(completed_embeddings) == 0:
            completed_embeddings = pd.DataFrame(columns=['test', 'model', 'modality'])
        else:
            completed_embeddings = pd.concat(completed_embeddings)
        completed_embeddings = completed_embeddings[['test', 'model', 'modality']].drop_duplicates()
        completed_tests = set([(t[0], t[1], t[2]) for t in completed_embeddings.values.tolist()])
    return (test, model_name, modality) not in completed_tests

def attr_needs_completion(model_name):
    if type(model_name) is tuple:
        model_name = '_'.join(model_name).replace('/', '')
    global attr_completed_tests
    if attr_completed_tests is None:
        save_dir = os.path.join('results', 'data', 'raw_embeddings')
        os.makedirs(save_dir, exist_ok=True)
        embedding_paths = [f for f in os.listdir(save_dir)
                           if f.split('.')[-1] == 'csv' and 'metadata' in f and 'attr' in f]
        completed_embeddings = [pd.read_csv(os.path.join('results', 'data', 'raw_embeddings', f), sep='\t') for f in embedding_paths]
        if len(completed_embeddings) == 0:
            completed_embeddings = pd.DataFrame(columns=['test', 'model'])
        else:
            completed_embeddings = pd.concat(completed_embeddings)
        completed_embeddings = completed_embeddings['model'].unique().tolist()
        attr_completed_tests = set(completed_embeddings)
    return model_name not in attr_completed_tests

def extract_attr_embd(model, tokenizer, model_name, device):
    attr = load_attributes()

    batches = [attr.iloc[i:i + 100] for i in range(0, len(attr['attribute'].tolist()), 100)]
    embeddings = []
    for batch in batches:
        embeddings.append(extract_text(model, tokenizer, batch['attribute'].tolist(), device, model_name))
    embeddings = pd.concat([pd.DataFrame(e) for e in embeddings])
    save_attr_embeddings(attr, embeddings, model_name)

def perform_test():
    results_fp = os.path.join('results', 'data', 'seat_replication.csv')
    models = cherti_et_al_models() + open_clip.list_pretrained()
    # Not using convnext_xxlarge because it is not supported by timm 0.6.12
    models = [m for m in models if m[0] != 'convnext_xxlarge']

    models = [m for m in models if m[0] != 'convnext_xxlarge']

    model_info = load_model_info(False)

    open_clip_models = pd.DataFrame(models, columns=['architecture', 'other_info']).merge(
        model_info[['architecture', 'other_info']], how='inner',
        on=['architecture', 'other_info'])
    open_clip_models = [o for o in zip(open_clip_models['architecture'].tolist(), open_clip_models['other_info'].tolist())]

    scaling_laws_models = model_info[model_info['model_name'].str.contains('scaling-laws-openclip')]['model_name'].str.split('_scaling-laws-openclip')
    scaling_laws_models = [o for o in zip(scaling_laws_models.str[0].tolist(), scaling_laws_models.str[1].tolist())]

    models = (
            open_clip_models +
            scaling_laws_models
    )

    text_tests = [f for f in os.listdir(os.path.join('data','tests')) if f.split('.')[-1] == 'jsonl']
    image_tests = pd.read_csv(os.path.join('data', 'ieat_tests.csv'))

    with tqdm(total=len(models) * (len(text_tests) + len(image_tests) + 1)) as pbar:
        for model_name in models:
            device = "cuda" if torch.cuda.is_available() else "cpu"

            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name[0],
                pretrained=os.path.join('references','scaling-laws-openclip', model_name[1]) if '.pt' in model_name[1] else model_name[1],
                device=device,
                cache_dir='references/scaling-laws-openclip' if '.pt' in model_name[1] else None)
            tokenizer = open_clip.get_tokenizer(model_name[0])

            for test_name in text_tests:
                # if needs_completion(model_name, test_name, 'text'):
                    extract_text_embd(model, tokenizer, test_name, model_name, device)
                    pbar.update()
            for _, test in image_tests.iterrows():
                if needs_completion(model_name, test, 'image'):
                    extract_image_embd(model, preprocess, test, device, model_name)
                pbar.update()
            if attr_needs_completion(model_name):
                extract_attr_embd(model, tokenizer, model_name, device)
            pbar.update()


if __name__ == '__main__':
    perform_test()

