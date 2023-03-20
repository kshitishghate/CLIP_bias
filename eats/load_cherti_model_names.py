import os

import pandas as pd

import open_clip


def cherti_et_al_models():
    """Get model names from Cherti et al. Adapted from scaling-laws-openclip/download_models.py"""

    # get model checkpoint names
    trained_models_info = pd.read_csv('scaling-laws-openclip/trained_models_info.csv')

    # Full info for models in paper
    all_samples_seen = ["3B", "13B", "34B"]
    all_dataset = ["80M", "400M", "2B"]
    all_model = ["ViT-B-32", "ViT-B-16", "ViT-L-14", "ViT-H-14", "ViT-g-14"]

    full_model_list = []
    for samples_seen in all_samples_seen:
        for dataset in all_dataset:
            for model in all_model:
                res = trained_models_info[
                    (trained_models_info.arch==model) & (trained_models_info.samples_seen_pretty==samples_seen) & (trained_models_info.data==dataset)
                ]
                if len(res) == 1:
                    if os.path.exists(os.path.join('scaling-laws-openclip', res['name'].iloc[0])):
                        full_model_list.append((model, os.path.join('scaling-laws-openclip', res['name'].iloc[0])))
                    else:
                        print(
                            f'ERROR: model {samples_seen, dataset, model} not found in scaling-laws-clip folder. Please '
                            f'download it using scaling-laws-openclip/download_models.py')
                elif len(res) > 1:
                    print('ERROR: more than one model found')

    # Remove models that are already in open_clip


    return full_model_list




if __name__ == '__main__':
    print(cherti_et_al_models())



