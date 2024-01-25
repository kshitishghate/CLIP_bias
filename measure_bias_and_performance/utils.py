import os
import re

from references import open_clip
import pandas as pd
import torch
from fvcore.nn import FlopCountAnalysis, ActivationCountAnalysis, flop_count_str
from huggingface_hub import list_files_info

global prev
prev = None

def test_already_run(model_name, test, file_name, hard_reload=False):
    if not os.path.exists(file_name):
        return False
    global prev
    if prev == None or hard_reload:
        previous_results = pd.read_csv(file_name)
    else:
        previous_results = prev
    relevant_results = previous_results[
        (previous_results['model'] == model_name)
        & (True if 'X' not in test.index else (previous_results['X'] == test['X']))
        & (True if 'Y' not in test.index else (previous_results['Y'] == test['Y']))
        & (True if 'A' not in test.index else (previous_results['A'] == test['A']))
        & (True if 'B' not in test.index else (previous_results['B'] == test['B']))
        & (True if 'Image Test' not in test.index else (previous_results['Image Test'] == test['Image Test']))
        & (True if 'Text Test' not in test.index else (previous_results['Text Test'] == test['Text Test']))
        & (True if 'Target' not in test.index else (previous_results['Target'] == test['Target']))
        & (True if 'na' not in test.index else (previous_results['na'] == test['na']))
        & (True if 'nt' not in test.index else (previous_results['nt'] == test['nt']))
        & (True if 'naa' not in test.index else (previous_results['naa'] == test['naa']))
        & (True if 'nab' not in test.index else (previous_results['nab'] == test['nab']))
        & (True if 'context' not in test.index else (previous_results['context'] == test['context']))
        & (True if 'order' not in test.index else (previous_results['order'] == test['order']))
        & (True if 'category' not in test.index else (previous_results['category'] == test['category']))
    ]
    return len(relevant_results) > 0


def save_test_results(result, file_name):
    if type(result) == pd.Series:
        result = pd.DataFrame(result).T
    if os.path.exists(file_name):
        # Read file header to make sure we can get the same column order
        header = pd.read_csv(file_name, nrows=0).columns.tolist()

        # If there are new columns, add them to the previous results
        new_columns = [c for c in result.columns if c not in header]
        if len(new_columns) > 0:
            previous_results = pd.read_csv(file_name)
            all_results = pd.concat([previous_results, result]).reset_index(drop=True)
            all_results.to_csv(file_name, index=False)
        else:
            result[header].to_csv(file_name, index=False, mode='a', header=False)
    else:
        result.to_csv(file_name, index=False)


def cherti_et_al_ckpts():
    all_epoch_files = [f.path for f in list_files_info("laion/scaling-laws-openclip")
                 if f.path.endswith('.pt') and 'full_checkpoints' in f.path
                 and 'epoch_' in f.path and 'latest' not in f.path]
    base_models = cherti_et_al_models()
    ckpts = []
    for f in all_epoch_files:
        model_name = (
            'references/scaling-laws-openclip/'
            + re.sub('/epoch_\d+\.pt', '', f.replace('full_checkpoints/', ''))
            + '.pt'
        )
        base_model = [b for b in base_models if b[1] == model_name]
        if len(base_model) != 1:
            print(f'ERROR: {model_name} not found in base models')
        else:
            local_path = os.path.join('references/scaling-laws-openclip', os.path.normpath(f))
            ckpts.append((base_model[0][0], local_path))
    return ckpts


def cherti_et_al_models():
    """Get model names from Cherti et al. Adapted from scaling-laws-openclip/download_models.py"""

    # get model checkpoint names
    trained_models_info = pd.read_csv('references/scaling-laws-openclip/trained_models_info.csv')

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
                    if not os.path.exists(os.path.join('references/scaling-laws-openclip', res['name'].iloc[0])):
                        print(
                            f'ERROR: model {samples_seen, dataset, model} not found in scaling-laws-clip folder. Please '
                            f'download it using scaling-laws-openclip/download_models.py')
                    full_model_list.append((model, os.path.join('references','scaling-laws-openclip', res['name'].iloc[0])))

                elif len(res) > 1:
                    print('ERROR: more than one model found')

    # Remove models that are already in open_clip


    return full_model_list


def profile_fvcore(
        model,
        image_input_size=(3, 224, 224),
        text_input_size=(77,),
        batch_size=1,
        detailed=False,
        force_cpu=False
):
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_image_input = torch.ones((batch_size,) + image_input_size, device=device, dtype=dtype)
    example_text_input = torch.ones((batch_size,) + text_input_size, device=device, dtype=torch.int64)
    fca = FlopCountAnalysis(model, (example_image_input, example_text_input))
    aca = ActivationCountAnalysis(model, (example_image_input, example_text_input))
    if detailed:
        fcs = flop_count_str(fca)
        print(fcs)
    return fca.total(), aca.total()


def profile_fvcore_text(
        model,
        text_input_size=(77,),
        batch_size=1,
        detailed=False,
        force_cpu=False
):
    # Adapted from:
    # https://github.com/mlfoundations/open_clip/blob/37b729bc69068daa7e860fb7dbcf1ef1d03a4185/src/training/profile.py
    #

    if force_cpu:
        model = model.to('cpu')
    device = next(model.parameters()).device
    example_input = torch.ones((batch_size,) + text_input_size, device=device, dtype=torch.int64)
    fca = FlopCountAnalysis(model, example_input)
    aca = ActivationCountAnalysis(model, example_input)
    if detailed:
        fcs = flop_count_str(fca)
        print(fcs)
    return fca.total(), aca.total()


def profile_fvcore_image(
        model,
        image_input_size=(3, 224, 224),
        batch_size=1,
        detailed=False,
        force_cpu=False
):
    # Adapted from:
    # https://github.com/mlfoundations/open_clip/blob/37b729bc69068daa7e860fb7dbcf1ef1d03a4185/src/training/profile.py
    #

    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.ones((batch_size,) + image_input_size, device=device, dtype=dtype)
    fca = FlopCountAnalysis(model, example_input)
    aca = ActivationCountAnalysis(model, example_input)
    if detailed:
        fcs = flop_count_str(fca)
        print(fcs)
    return fca.total(), aca.total()


def count_params(model):
    # Adapted from:
    # https://github.com/mlfoundations/open_clip/blob/37b729bc69068daa7e860fb7dbcf1ef1d03a4185/src/training/profile.py
    #

    return sum([m.numel() for m in model.parameters()])


def profile_model(model_name):
    # Adapted from:
    # https://github.com/mlfoundations/open_clip/blob/37b729bc69068daa7e860fb7dbcf1ef1d03a4185/src/training/profile.py
    #

    model = open_clip.create_model(model_name, force_custom_text=True, pretrained_hf=False)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    if isinstance(model.visual.image_size, (tuple, list)):
        image_input_size = (3,) + tuple(model.visual.image_size[-2:])
    else:
        image_input_size = (3, model.visual.image_size, model.visual.image_size)
    text_input_size = (77,)

    results = {}
    results['model'] = model_name
    results['image_size'] = image_input_size[1]

    model_cfg = open_clip.get_model_config(model_name)
    if model_cfg:
        vision_cfg = open_clip.CLIPVisionCfg(**model_cfg['vision_cfg'])
        text_cfg = open_clip.CLIPTextCfg(**model_cfg['text_cfg'])
        results['image_width'] = int(vision_cfg.width)
        results['text_width'] = int(text_cfg.width)
        results['embed_dim'] = int(model_cfg['embed_dim'])
        results['params'] = count_params(model)
        results['image_params'] = count_params(model.visual)
        results['text_params'] = count_params(model.text)
    else:
        results['image_width'] = 0
        results['text_width'] = 0
        results['embed_dim'] = 0
        results['params'] = count_params(model)
        results['image_params'] = count_params(model.visual)
        results['text_params'] = count_params(model.text)

    retries = 2
    while retries:
        retries -= 1
        try:
            results['macs'], results['acts'] = profile_fvcore(
                model, image_input_size=image_input_size, text_input_size=text_input_size, force_cpu=not retries)

            results['image_macs'], results['image_acts'] = profile_fvcore_image(
                model.visual, image_input_size=image_input_size, force_cpu=not retries)

            results['text_macs'], results['text_acts'] = profile_fvcore_text(
                model.text, text_input_size=text_input_size, force_cpu=not retries)
            break
        except RuntimeError as e:
            pass
    return results
