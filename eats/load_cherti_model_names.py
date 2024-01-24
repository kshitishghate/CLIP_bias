import os
import re

import pandas as pd

import open_clip

from huggingface_hub import hf_hub_url, get_hf_file_metadata, list_files_info
from huggingface_hub.utils._errors import EntryNotFoundError


def cherti_et_al_ckpts():
    all_epoch_files = [f.path for f in list_files_info("laion/scaling-laws-openclip")
                 if f.path.endswith('.pt') and 'full_checkpoints' in f.path
                 and 'epoch_' in f.path and 'latest' not in f.path]
    base_models = cherti_et_al_models()
    ckpts = []
    for f in all_epoch_files:
        model_name = (
            'scaling-laws-openclip/'
            + re.sub('/epoch_\d+\.pt', '', f.replace('full_checkpoints/', ''))
            + '.pt'
        )
        base_model = [b for b in base_models if b[1] == model_name]
        if len(base_model) != 1:
            print(f'ERROR: {model_name} not found in base models')
        else:
            local_path = os.path.join('scaling-laws-openclip', os.path.normpath(f))
            ckpts.append((base_model[0][0], local_path))
    return ckpts


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
                    if not os.path.exists(os.path.join('scaling-laws-openclip', res['name'].iloc[0])):
                        print(
                            f'ERROR: model {samples_seen, dataset, model} not found in scaling-laws-clip folder. Please '
                            f'download it using scaling-laws-openclip/download_models.py')
                    full_model_list.append((model, os.path.join('scaling-laws-openclip', res['name'].iloc[0])))

                elif len(res) > 1:
                    print('ERROR: more than one model found')

    # Remove models that are already in open_clip


    return full_model_list




if __name__ == '__main__':
    print(cherti_et_al_ckpts())



