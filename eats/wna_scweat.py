import os

import xml.etree.ElementTree as ET

import torch
import nltk
import numpy as np
import pandas as pd

from scipy.special import comb
from CLIP import clip
from tqdm import tqdm
from nltk.corpus import wordnet
from pattern.text.en import pluralize

from eats.extract_clip import load_images, extract_images, extract_text
from eats.sc_weat import SCWEAT
from eats.result_saving import save_test_results


def retrieve_word_name(obj, hierarchy):
    word = obj.attrib['name']
    if '-' in word:
        word_classes = ['positive', 'negative', 'neutral', 'ambiguous']
        for word_class in word_classes:
            word = word.replace(f'{word_class}-', '')
        if 'isa' in obj.attrib.keys():
            parent_obj = [p for p in hierarchy.getroot() if p.attrib['name'] == obj.attrib['isa']][0]
            parent = retrieve_word_name(parent_obj, hierarchy)
            word = word.replace(f'{parent}-','')
        word = word.replace('-', ' ')
    return word

def find_all_tests(models = None):
    if models is None:
        models = clip.available_models()
    all_models = pd.DataFrame({'model': models, 'I': 1})

    all_demographics = pd.read_csv(os.path.join('data', 'sc_weat_tests.csv'))
    all_demographics['I'] = 1

    hierarchy = ET.parse(os.path.join('data', 'wn-domains-3.2', 'wn-affect-1.1', 'a-hierarchy.xml'))
    cleaned_words = [retrieve_word_name(w, hierarchy) for w in hierarchy.getroot()]
    uncleaned_words = [w.attrib['name'].replace('-', ' ') for w in hierarchy.getroot()]
    all_words = np.unique(cleaned_words + uncleaned_words)
    all_plurals = [pluralize(w) for w in all_words]
    all_words = pd.DataFrame({'singular': all_words, 'plural': all_plurals, 'I': 1})

    bleached = pd.read_csv(os.path.join('data', 'emotion_templates', 'bleached.txt'), header=0, comment='#')
    feeling = pd.read_csv(os.path.join('data', 'emotion_templates', 'feeling.txt'), header=0, comment='#')
    simple = pd.read_csv(os.path.join('data', 'emotion_templates', 'simple.txt'), header=0, comment='#')
    all_templates = pd.concat([bleached, feeling, simple], axis=0)
    all_templates['I'] = 1

    # place all words into templates
    all_tests = all_templates.merge(all_words, on='I')
    templated = []
    for i, (_, requires_plural, for_words_that_start_with_vowel, template, _, singular, plural) in all_tests.iterrows():
        vowels = ['a', 'e', 'i', 'o', 'u']
        # if there aren't any grammer requirements for the template, just place the singular version of the word in
        if np.isnan(for_words_that_start_with_vowel):
            templated.append(template.replace('<EMOTION>', singular))
        # if the template is for vowels, but this word doesn't start with a vowel, then throw this template out
        elif for_words_that_start_with_vowel and singular[0] not in vowels:
            templated.append(np.nan)
        # if the template is not for vowels, but this word starts with a vowel, then throw this template out
        elif not for_words_that_start_with_vowel and singular[0] in vowels:
            templated.append(np.nan)
        elif requires_plural:
            templated.append(template.replace('<EMOTION>', plural))
        elif not requires_plural:
            templated.append(template.replace('<EMOTION>', singular))
        else:
            raise ValueError
    all_tests['formatted'] = templated
    all_tests = all_tests[~all_tests['formatted'].isna()]

    all_tests = all_tests.merge(all_models, on='I')
    all_tests = all_tests.merge(all_demographics, on='I')

    all_tests = all_tests.drop(columns='I')

    all_tests = all_tests.sort_values(['model', 'Title']).reset_index(drop=True)

    return all_tests


def find_remaining_tests_to_run(models=None):
    all_tests = find_all_tests(models=models)
    if not os.path.exists(os.path.join('results','data','wna_scweat.csv')):
        return all_tests
    else:
        already_completed = pd.read_csv(os.path.join('results','data','wna_scweat.csv'))
        outer_join = all_tests.merge(already_completed, on=['model','Title','formatted'], suffixes = ('','_DELETE'), how='outer', indicator=True)
        remaining_tests = outer_join[outer_join['_merge'] == 'left_only'].drop(['_merge'] + [c for c in outer_join.columns if '_DELETE' in c], axis=1)
        remaining_tests = remaining_tests.sort_values(['model','Title']).reset_index(drop=True)
        return remaining_tests



def perform_test(device):
    results_fp = os.path.join('results', 'data', 'wna_scweat.csv')
    models = clip.available_models()
    models.reverse()
    models = models[:4]

    tests_to_run = find_remaining_tests_to_run(models=models)

    current_model_name = None


    new_results = []
    with tqdm(total=len(tests_to_run)) as pbar:
        for i, test in tests_to_run.iterrows():
            if current_model_name != test['model']:
                current_model_name = test['model']
                model, preprocess = clip.load(current_model_name, device)

            test['na'] = len(load_images(test['Title'], test['A'], 'cfd'))

            np.random.seed(82804230)

            stimuli = {
                'w': test['formatted'],
                'A': load_images(test['Title'], test['A'], 'cfd'),
                'B': load_images(test['Title'], test['B'], 'cfd'),
            }

            embeddings = {
                'w': extract_text(model, preprocess, stimuli['w'], device, current_model_name),
                'A': extract_images(model, preprocess, stimuli['A'], device, current_model_name),
                'B': extract_images(model, preprocess, stimuli['B'], device, current_model_name),
            }


            test_result = SCWEAT(embeddings['w'], embeddings['A'], embeddings['B'])
            test['association_score'] = test_result.association_score()[0]
            test['dist_to_pleasantness'] = test_result.mean_similarity('A')[0]
            test['dist_to_unpleasantness'] = test_result.mean_similarity('B')[0]
            test['dist_std'] = test_result.AB_std()

            new_results.append(test)

            if i % 250 == 0:
                new_results = pd.concat(new_results, axis=1).T
                save_test_results(new_results, results_fp)
                new_results = []
            pbar.update()
    new_results = pd.concat(new_results, axis=1).T
    save_test_results(new_results, results_fp)


if __name__ == '__main__':
    should_continue=True
    while should_continue:
        if os.path.exists(os.path.join('results','data','wna_scweat.csv')):
            num_results_so_far = len(pd.read_csv(os.path.join('results','data','wna_scweat.csv')))
        else:
            num_results_so_far = 0
        try:
            perform_test('cuda' if torch.cuda.is_available() else 'cpu')
        except:
            perform_test('cpu')
        if num_results_so_far == len(pd.read_csv(os.path.join('results','data','wna_scweat.csv'))):
            should_continue = False
