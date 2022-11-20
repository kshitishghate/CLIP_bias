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
from ieat.weat.weat.test import Test
from eats.result_saving import test_already_run, save_test_results



def get_word_list(word):
    # frequency data taken from: https://anc.org/data/anc-second-release/frequency-data/
    # https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

    hierarchy = ET.parse(os.path.join('data','wn-domains-3.2','wn-affect-1.1','a-hierarchy.xml'))
    prev_parent_len = -1
    parents = [word, 'negative-'+word,'positive-'+word, 'neutral-'+word]
    while len(parents) > prev_parent_len:
        prev_parent_len = len(parents)
        parents = [word, 'negative-'+word,'positive-'+word, 'neutral-'+word] + [w.attrib['name'] for w in hierarchy.getroot() if 'isa' in w.attrib.keys() and w.attrib['isa'] in parents]

    freq = pd.read_csv(os.path.join('data', 'ANC-written-count.txt'), sep='\t', encoding='ISO-8859-1', header=None)
    word_freq = freq[(freq[1] == word)][3].values[0]

    words = freq[(freq[0].isin(parents)
                  | freq[1].isin(parents)
                  | ('positive-' + freq[0]).isin(parents)
                  | ('negative-' + freq[0]).isin(parents)
                  | ('neutral-' + freq[1]).isin(parents)
                  )
                     & (freq[2].isin(['NN', 'NNP']))
                     & (freq[3] >= word_freq / 20)]

    words = np.unique(words[0].tolist())
    np.random.shuffle(words)
    return words


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

    templates = [t.format(word).capitalize() for t in singular_templates] + [t.format(pluralize(word)).capitalize() for t in plural_templates]
    return templates


def perform_test():
    nouns = [
        'anger', 'disgust', 'fear', 'joy',
        # 'sadness', 'surprise', 'anticipation', 'trust'
    ]

    all_tests = pd.read_csv(os.path.join('data', 'tests.csv'))
    all_tests = all_tests[all_tests['A'] == 'Pleasant']
    all_tests = all_tests[~all_tests['Title'].isin(['Disabled','Insect-Flower','Religion',
                                                    'Sexuality','Skin-Tone','Weapon','Weight','Age','Arab-Muslim',
                                                    'Native'])].reset_index(drop=True)


    total = len(clip.available_models()) * len(nouns) * 2 * len(all_tests)
    results_fp = os.path.join('results', 'data', 'categorical_wna.csv')
    if os.path.exists(results_fp):
        completed = pd.read_csv(results_fp)
        completed = [completed['A'].str.contains(a) for a in nouns]
        completed = pd.concat(completed, axis=1).any(axis=1).sum()
    else:
        completed = 0
    remaining = total - completed

    with tqdm(total=remaining) as pbar:
        models = clip.available_models()
        models.reverse()
        for model_name in models:
            # device = 'cuda' if torch.cuda.is_available() else 'cpu'
            device = 'cpu'
            model, preprocess = clip.load(model_name, device)

            for emotion in nouns:
                for comparison_name in ['negation', 'apathy']:
                    seed = ord(model_name[0]) + ord(model_name[-1])*10 + ord(model_name[-2])*100 + ord(model_name[-1])*1000 + ord(emotion[0])*10000 + ord(emotion[0])*100000 + ord(comparison_name[0])*10000
                    np.random.seed(seed)
                    words = get_word_list(emotion)
                    if comparison_name == 'negation':
                        templated =[place_in_templates(w, False) for w in words]
                        phrases = [item for sublist in templated for item in sublist]

                        opposite_templated =[place_in_templates(w, True) for w in words]
                        opposite_phrases = [item for sublist in opposite_templated for item in sublist]

                    elif comparison_name == 'apathy':
                        opposite_words = get_word_list('apathy')
                        words = words[:min(len(opposite_words), len(words))]
                        opposite_words = opposite_words[:min(len(opposite_words), len(words))]

                        templated =[place_in_templates(w, False) for w in words]
                        phrases = [item for sublist in templated for item in sublist]

                        opposite_templated = [place_in_templates(w, False) for w in opposite_words]
                        opposite_phrases = [item for sublist in opposite_templated for item in sublist]


                    for i, test in all_tests.iterrows():
                        test['A'] = emotion
                        test['B'] = f'not {emotion}' if comparison_name == 'negation' else comparison_name
                        test['na'] = len(phrases)
                        test['nt'] = len(load_images(test['Title'], test['X'], 'cfd'))
                        test.name = None

                        if not test_already_run(model_name, test, results_fp):
                            np.random.seed(82804230)

                            stimuli = {
                                'X': load_images(test['Title'], test['X'], 'cfd'),
                                'Y': load_images(test['Title'], test['Y'], 'cfd'),
                                emotion: phrases,
                                comparison_name: opposite_phrases
                            }

                            embeddings = {
                                'X': extract_images(model, preprocess, stimuli['X'], device, model_name),
                                'Y': extract_images(model, preprocess, stimuli['Y'], device, model_name),
                                emotion: extract_text(model, preprocess, stimuli[emotion], device, model_name),
                                comparison_name: extract_text(model, preprocess, stimuli[comparison_name], device,
                                                              model_name)
                            }

                            npermutations = min(
                                1000000,
                                int(comb(len(embeddings['X']) + len(embeddings['Y']), len(embeddings['X']))),

                            )

                            test_result = Test(embeddings['X'], embeddings['Y'], embeddings[emotion], embeddings[comparison_name])
                            test_result = pd.Series({
                                'model':model_name,
                                'pvalue':test_result.p(npermutations),
                                'effect_size':test_result.effect_size(),
                                'npermutations':npermutations,
                            })

                            save_test_results(pd.concat([test, test_result]), results_fp)
                            pbar.update()


if __name__ == '__main__':
    perform_test()
