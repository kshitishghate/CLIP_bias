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


def test_already_run(model_name, test, file_name):
    if not os.path.exists(file_name):
        return False
    previous_results = pd.read_csv(file_name)
    relevant_results = previous_results[
        (previous_results['model'] == model_name)
        & (previous_results['w'] == test['w'])
        & (previous_results['A'] == test['A'])
        & (previous_results['B'] == test['B'])
        & (previous_results['Title'] == test['Title'])
        & (True if 'context' not in test.index else (previous_results['context'] == test['context']))
    ]
    return len(relevant_results) > 0


def place_in_templates(word, negate=False):
    vowel_list = ['a','e','i','o','u']
    # Check if first letter is vowel, which would mean using "an" instead of "a"
    if negate:
        if word[0] in vowel_list:
            singular_templates = ['This is not an {}.', 'That is not an {}.', 'There is not an {}.', 'Here is not an {}.',
                                  'The {} is not here.',
                                  'The {} is not there.', 'An {} is not a thing.', 'It is not an {}.']
        else:
            singular_templates = ['This is not a {}.', 'That is not a {}.', 'There is not a {}.', 'Here is not a {}.',
                                  'The {} is not here.',
                                  'The {} is not there.', 'A {} is not a thing.', 'It is not a {}.']

        plural_templates = ['These are not {}.', 'Those are not {}.', 'They are not {}.', 'The {} are not here.',
                            'The {} are not there.', '{} are not things.']
    else:
        if word[0] in vowel_list:
            singular_templates = ['This is an {}.', 'That is an {}.', 'There is an {}.', 'Here is an {}.', 'The {} is here.',
                                 'The {} is there.', 'An {} is a thing.', 'It is an {}.']
        else:
            singular_templates = ['This is a {}.', 'That is a {}.', 'There is a {}.', 'Here is a {}.', 'The {} is here.',
                         'The {} is there.', 'A {} is a thing.', 'It is a {}.']

        plural_templates = ['These are {}.', 'Those are {}.', 'They are {}.', 'The {} are here.',
                            'The {} are there.', '{} are things.']

    templated = [t.format(word).capitalize() for t in singular_templates] + [t.format(pluralize(word)).capitalize() for t in plural_templates]
    return templated, singular_templates + plural_templates


def perform_test():
    nouns = [
        'anger', 'disgust', 'fear', 'joy',
        'sadness', 'surprise', 'anticipation', 'trust'
    ]

    npermutations = 10000
    all_tests = pd.read_csv(os.path.join('data', 'sc_weat_tests.csv'))

    hierarchy = ET.parse(os.path.join('data','wn-domains-3.2','wn-affect-1.1','a-hierarchy.xml'))
    all_words = [w.attrib['name'].replace('positive-','').replace('negative-','').replace('neutral-','').replace('-',' ') for w in hierarchy.getroot()]

    results_fp = os.path.join('results', 'data', 'wna_scweat_fairface.csv')
    models = clip.available_models()
    # models.reverse()
    models = models[:4]
    total = len(models) * 18 * 298 * 3
    if os.path.exists(results_fp):
        completed = pd.read_csv(results_fp)
        completed = [completed['model'].str.contains(a) for a in models]
        completed = pd.concat(completed, axis=1).any(axis=1).sum()
    else:
        completed = 0
    remaining = total - completed

    with tqdm(total=remaining) as pbar:
        for model_name in models:
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            device = 'cpu'
            model, preprocess = clip.load(model_name, device)


            for word in all_words:
                semantically_bleached, sb_templates = place_in_templates(word, False)
                # semantically_bleached_and_negated, sbn_templates = place_in_templates(word, True)

                other_templates = ['{}', 'A person who is feeling {}.', 'A person who makes me feel {}.',
                                   'A person who is conveying {}.',
                                   # 'not {}.','A person who is not feeling {}.', 'A person who does not make me feel {}.',
                                   # 'A person who is not conveying {}.'
                                   ]

                all_to_test = (semantically_bleached
                               # + semantically_bleached_and_negated
                               + [t.format(word) for t in other_templates])
                all_templates = (sb_templates
                                 # + sbn_templates
                                 + other_templates)

                for phrase, template in zip(all_to_test, all_templates):

                    for i, test in all_tests.iterrows():

                        test['w'] = phrase
                        test['word'] = word
                        test['template'] = template
                        test.name = None


                        if not test_already_run(model_name, test, results_fp):
                            test['na'] = len(load_images(test['Title'], test['A'], 'fairface'))

                            np.random.seed(82804230)

                            stimuli = {
                                'w': phrase,
                                'A': load_images(test['Title'], test['A'], 'fairface'),
                                'B': load_images(test['Title'], test['B'], 'fairface'),

                            }

                            embeddings = {
                                'w': extract_text(model, preprocess, stimuli['w'], device, model_name),
                                'A': extract_images(model, preprocess, stimuli['A'], device, model_name),
                                'B': extract_images(model, preprocess, stimuli['B'], device, model_name),
                            }


                            test_result = SCWEAT(embeddings['w'], embeddings['A'], embeddings['B'])
                            test_result = pd.Series({
                                'model':model_name,
                                'association_score':test_result.association_score()[0],
                                'p':test_result.p(n_samples=npermutations),
                                'npermutations':npermutations,
                            })

                            save_test_results(pd.concat([test, test_result]), results_fp)
                            pbar.update()


if __name__ == '__main__':
    perform_test()
