import open_clip
import pandas as pd
import numpy as np
from results.analysis_scripts.result_loading import load_seat_replication_results
from results.analysis_scripts.result_loading import load_ieat_replication_results
from results.analysis_scripts.result_loading import load_cross_modal_results
import os

from measure_bias_and_performance.utils import cherti_et_al_models, profile_model


# may want to download other models from:
# https://github.com/LAION-AI/scaling-laws-openclip/blob/master/download_models.py
# https://huggingface.co/laion/scaling-laws-openclip/tree/main


def load_cross_modal_bias():
    _, cross_modal_results = load_cross_modal_results(openai_only=False)
    cross_modal_results = cross_modal_results.rename(columns={'X_x': 'X', 'Y_x': 'Y', 'A_x': 'A', 'B_x': 'B'})

    cross_modal_results['test_category'] = cross_modal_results['Image_Test'].replace(regex=True, to_replace={
        '.*EA ?- ?AA (Names|Terms) ?/ ?Valences? ?': 'EA-AA/Valence',
        'Insect-Flower/Valence': 'Flowers-Insects/Valence',
        'Gender/Science vs. Liberal-Arts': 'Science-Liberal Arts/Gender',
        'Math/Gender': 'Math-Arts/Gender',
        'Gender/Career vs. Family': 'Gender/Career',
        '.*Age/Valence': 'Age/Valence',
        'Native/US vs. World': 'Native/US-World',
        'Asian/American vs. Foreign': 'Asian/US-Foreign',
        'Arab-Muslim/Valence':'Arab-Non Arab/Valence',
        'Disabled/Valence': 'Disability/Valence',
        'Race/Tool vs. Weapon \(Modern\)': 'EA-AA/Modern Tool-Modern Weapon',
        'Race/Tool vs. Weapon$': 'EA-AA/Tool-Weapon',
        'Skin-Tone/Valence':'Skin Tone/Valence',
        'Race/Valence': 'EA-AA/Valence',
    })
    cross_modal_results['overall_test_category'] = cross_modal_results['test_category'].replace(regex=True, to_replace={
        'Age/Valence': 'Age',
        'Arab-Non Arab/Valence': 'Race/Skin Tone',
        'Asian/US-Foreign':'Race/Skin Tone',
        'Disability/Valence': 'Health',
        'Gender/Career': 'Gender',
        'Gender/Science vs. Arts':'Gender',
        'Science-Liberal Arts/Gender':'Gender',
        'Flowers-Insects/Valence': 'Non-Human',
        'Native/US-World':'Race/Skin Tone',
        'EA-AA/Tool-Weapon':'Race/Skin Tone',
        'EA-AA/Modern Tool-Modern Weapon':'Race/Skin Tone',
        'EA-AA/Valence':'Race/Skin Tone',
        'Religion/Valence':'Religion',
        'Sexuality/Valence':'Sexuality',
        'Skin Tone/Valence':'Race/Skin Tone',
        'Weight/Valence': 'Health'
    })
    return cross_modal_results

def load_image_bias_results():
    image_biases, _ = load_ieat_replication_results(openai_only=False)
    image_biases['test_category'] = image_biases['Test'].replace(regex=True, to_replace={
        '.*EA ?- ?AA (Names|Terms) ?/ ?Valences? ?': 'EA-AA/Valence',
        'Insect-Flower/Valence': 'Flowers-Insects/Valence',
        'Gender/Science vs. Liberal-Arts': 'Science-Liberal Arts/Gender',
        'Math/Gender': 'Math-Arts/Gender',
        'Gender/Career vs. Family': 'Gender/Career',
        '.*Age/Valence': 'Age/Valence',
        'Native/US vs. World': 'Native/US-World',
        'Asian/American vs. Foreign': 'Asian/US-Foreign',
        'Arab-Muslim/Valence':'Arab-Non Arab/Valence',
        'Disabled/Valence': 'Disability/Valence',
        'Race/Tool vs. Weapon \(Modern\)': 'EA-AA/Modern Tool-Modern Weapon',
        'Race/Tool vs. Weapon$': 'EA-AA/Tool-Weapon',
        'Skin-Tone/Valence':'Skin Tone/Valence',
        'Race/Valence': 'EA-AA/Valence',
    })
    image_biases['overall_test_category'] = image_biases['test_category'].replace(regex=True, to_replace={
        'Age/Valence': 'Age',
        'Arab-Non Arab/Valence': 'Race/Skin Tone',
        'Asian/US-Foreign':'Race/Skin Tone',
        'Disability/Valence': 'Health',
        'Gender/Career': 'Gender',
        'Science-Liberal Arts/Gender':'Gender',
        'Flowers-Insects/Valence': 'Non-Human',
        'Native/US-World':'Race/Skin Tone',
        'EA-AA/Tool-Weapon':'Race/Skin Tone',
        'EA-AA/Modern Tool-Modern Weapon':'Race/Skin Tone',
        'EA-AA/Valence':'Race/Skin Tone',
        'Religion/Valence':'Religion',
        'Sexuality/Valence':'Sexuality',
        'Skin Tone/Valence':'Race/Skin Tone',
        'Weight/Valence': 'Health'
    })
    image_biases['Test'] = image_biases['Test'].replace(regex=True, to_replace={
        'Age/Valence': 'Age/Valence (Images)',}
    )
    image_biases['modality'] = 'image'
    image_biases['stimuli_type'] = 'images'
    image_biases['word_category'] = 'images'
    return  image_biases

def load_text_bias_results():
    text_biases, _ = load_seat_replication_results(openai_only=False)
    text_biases['test_category'] = text_biases['Test'].replace(regex=True, to_replace={
        '.*EA ?- ?AA (Names|Terms) ?/ ?Valences? ?': 'EA-AA/Valence',
        '.*Flower ?/ ?Valences? ?': 'Flowers-Insects/Valence',
        '.*Science/Gender.*': 'Science-Liberal Arts/Gender',
        '.*Math/Gender.*': 'Math-Arts/Gender',
        '.*Gendered.*Career*': 'Gender/Career',
        '.*Age/Valence': 'Age/Valence',
        '.*Instruments/Valence': 'Instruments/Valence',
        '.*Physical Disease/Permanent': 'Physical-Mental Disease/Permanence',
    })

    text_biases['overall_test_category'] = text_biases['test_category'].replace(regex=True, to_replace={
        '.*Gender.*': 'Gender',
        '.*EA-AA.*': 'Race/Skin Tone',
        '.*Age.*': 'Age',
        'Instrument.*': 'Non-Human',
        'Flowers.*': 'Non-Human',
        'Physical-Mental Disease.*': 'Health'
    })

    text_biases = text_biases.rename(columns={'p_value':'pvalue'})

    text_biases['stimuli_type'] = np.where(text_biases['Test'].str.contains('Sentences'), 'sentences', 'words')
    text_biases['word_category'] = text_biases['Test'].replace(regex=True, to_replace={
        '.*Names.*': 'Names',
        '.*Terms.*': 'Terms',
        '.*Age.*': 'Names',
        '.*Instrument.*': 'Terms',
        '.*Flower.*': 'Terms',
        '.*Disease.*': 'Terms'
    })
    text_biases['modality'] = 'text'
    text_biases['Test'] = text_biases['Test'].replace(regex=True, to_replace={
        'Age/Valence': 'Age/Valence (Text)',}
    )
    text_biases['Test'] = np.where(
        text_biases['Test'].str.contains('EA-AA Names/Valence'),
        text_biases['Test'] + ' (' + text_biases['test_name'] + ')',
        text_biases['Test']
    )
    return text_biases





def load_model_info(include_num_params=True):
    # Want to fix ordering on Math/Gender tests and Science/Gender tests - make sure it's the same in both modalities
    model_list = open_clip.list_pretrained()
    model_list = [m for m in model_list if m[0] != 'convnext_xxlarge']

    model_info = pd.DataFrame({
        'architecture': [m[0] for m in model_list],
        'other_info': [m[1] for m in model_list]
    })

    # Remove models that are trained for 31 epochs, as we are using the 32 epoch models, which are likely non-independent
    model_info = model_info[~model_info['other_info'].str.contains('_e31$', regex=True)]

    # Remove frozen roberta model, as it was initialized with weights from another model we are using ('ViT-H-14', 'laion2b_s32b_b79k')
    # https://huggingface.co/laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k
    model_info = model_info[~model_info['other_info'].str.contains('frozen', regex=True)]

    # Remove quickgelu models that are repeats of non-quickgelu models
    quickgelu_repeats = [
        'RN101-quickgelu_yfcc15m',
        'RN50-quickgelu_yfcc15m',
        'RN50-quickgelu_cc12m',
        'RN50-quickgelu_openai',
        'RN101-quickgelu_openai',
        'ViT-B-32-quickgelu_openai',
        'ViT-B-32-quickgelu_laion400m_e32'
    ]
    model_info = model_info[~(model_info['architecture'] + '_' + model_info['other_info']).isin(quickgelu_repeats)]

    # remove models that are repeats (based on  either 1. architecture, training dataset, and num epochs/num samples or
    # 2. correlation with bias results from another model)
    # between cherti et al and openclip
    repeated_between_modelsets = [
        'ViT-H-14_laion2b_s32b_b79k',
        # repeat of 'ViT-H-14_scaling-laws-openclipModel-H-14_Data-2B_Samples-34B_lr-5e-4_bs-79k.pt' (correlation match)
        'ViT-g-14_laion2b_s12b_b42k',
        # repeat of 'ViT-g-14_scaling-laws-openclipModel-g-14_Data-2B_Samples-13B_lr-5e-4_bs-64k.pt' (correlation match)
        'ViT-B-16_laion2b_s34b_b88k',
        # repeat of 'ViT-B-16_scaling-laws-openclipModel-B-16_Data-2B_Samples-34B_lr-1e-3_bs-88k.pt' (correlation match)
        'ViT-B-16_laion400m_e32',
        # repeat of 'ViT-B-16_scaling-laws-openclipModel-B-16_Data-400M_Samples-13B_lr-5e-4_bs-33k.pt' (correlation match)
        'ViT-H-14_laion2b_s32b_b79k',
        # repeat of 'ViT-H-14_scaling-laws-openclipModel-H-14_Data-2B_Samples-34B_lr-5e-4_bs-79k.pt' (correlation match)
        'ViT-L-14_laion2b_s32b_b82k',
        # repeat of 'ViT-L-14_scaling-laws-openclipModel-L-14_Data-2B_Samples-34B_lr-1e-3_bs-86k.pt' (correlation match)
        'ViT-B-32_laion2b_s34b_b79k',
        # repeat of 'ViT-B-32_scaling-laws-openclipModel-B-32_Data-2B_Samples-34B_lr-1e-3_bs-79k.pt' (correlation match)
        'ViT-B-32_laion400m_e32',
        # POSSIBLE repeat 'ViT-B-32_scaling-laws-openclipModel-B-32_Data-400M_Samples-13B_lr-1e-3_bs-86k.pt' (num epochs match)
        'ViT-L-14_laion400m_e32'
        # POSSIBLE repeat of 'ViT-L-14_scaling-laws-openclipModel-L-14_Data-400M_Samples-13B_lr-1e-3_bs-86k.pt' (num epochs match)
    ]
    model_info = model_info[
        ~(model_info['architecture'] + '_' + model_info['other_info']).isin(repeated_between_modelsets)]

    model_info['model_family'] = model_info['architecture'].replace(regex=True, to_replace={
        'ViT.+': 'ViT',
        'RN.+': 'RN',
        'convnext.+': 'convnext',
        'xlm-roberta-.+-ViT': 'xlm-roberta-ViT'
    })

    model_info['dataset'] = model_info['other_info'].replace(regex=True, to_replace={
        'openai': 'OpenAI WebImageText',
        'laion2b.+': 'laion2b',
        'laion400m.+': 'laion400m',
        'laion_aesthetic.+': 'laion_aesthetic',
        'laion5b.+': 'laion5b',
        'frozen_laion5b.+': 'laion5b'
    })

    billion = 1000000000
    model_info['samples_seen'] = model_info['other_info'].replace(regex=True, to_replace={
        '.+s3b.+': 3 * billion,
        '.+s12b.+': 12 * billion,
        '.+s13b.+': 13 * billion,
        '.+s26b.+': 26 * billion,
        '.+s29b.+': 29 * billion,
        '.+s32b.+': 32 * billion,
        '.+s34b.+': 34 * billion,
        '.+s39b.+': 39 * billion,
        'openai': 13 * billion,
        'yfcc15m': 32 * 0.015 * billion, #https://github.com/mlfoundations/open_clip/discussions/472
        'cc12m': 32 * 0.0108 * billion, # https://github.com/mlfoundations/open_clip/discussions/472
        'laion400m_e32': 32 * 0.4 * billion, # https://github.com/mlfoundations/open_clip/discussions/472
        'laion2b_e16': 16 * 2 * billion, # https://github.com/mlfoundations/open_clip/discussions/472
    })

    model_info['model_name'] = model_info['architecture'] + '_' + model_info['other_info']
    model_info['fine_tuned'] = model_info['other_info'].str.contains('finetuned')

    cmodel_info = pd.read_csv('scaling-laws-openclip/trained_models_info.csv').rename(columns={
        'name': 'model_name',
        'samples_seen_pretty': 'samples_seen',
        'data': 'dataset',
        'arch': 'architecture'
    })
    cmodel_info['fine_tuned'] = False
    cmodel_info['samples_seen'] = cmodel_info['samples_seen'].str.slice(0, -1).astype(int) * billion
    cmodel_info['model_family'] = 'ViT'
    cmodel_info['dataset'] = 'laion' + cmodel_info['dataset'].str.lower()

    model_info = pd.concat([model_info, cmodel_info]).sort_values(
        ['architecture', 'dataset', 'samples_seen']).reset_index(drop=True)

    # Get num params/num gmacs
    if include_num_params:
        model_info_dict = {}
        already_computed = {}
        for i, model in model_info.iterrows():
            if len(model_info_dict) == 0:
                results = profile_model(model['architecture'])
                for key, value in results.items():
                    model_info_dict[key] = [value]
                already_computed[model['architecture']] = results
            else:
                if model['architecture'] not in already_computed.keys():
                    already_computed[model['architecture']] = profile_model(model['architecture'])
                for key in model_info_dict.keys():
                    try:
                        model_info_dict[key].append(already_computed[model['architecture']][key])
                    except KeyError:
                        model_info_dict[key].append(np.nan)

        model_info_dict.pop('model')
        for key in model_info_dict.keys():
            model_info[key] = model_info_dict[key]

    model_info['dataset_size'] = model_info['dataset'].replace({
        'OpenAI WebImageText':0.4*billion,
        'yfcc15m':0.015*billion,
        'cc12m':0.012*billion,
        'laion2b':2*billion,
        'laion400m':0.4*billion,
        'laion80m':0.08*billion,
        'mscoco_finetuned_laion2b':2*billion,
        'laion_aesthetic':0.9*billion,
        'laion5b':5*billion
    })
    model_info['dataset_family'] = model_info['dataset'].replace({
        'laion2b':'english-only-laion2b',
        'laion400m':'english-only-laion2b',
        'laion80m':'english-only-laion2b',
    })
    model_info['model_source'] = model_info['model_name'].replace(to_replace={
        '.+openai': 'openai',
        '.+\.pt': 'cherti',
    },regex=True)
    model_info['model_source'] = np.where(
        model_info['model_source'].isin(['openai','cherti']),
        model_info['model_source'],
        'open_clip'
    )
    model_info['epochs'] = np.where(
        model_info['other_info'] == 'openai',
        32, # Models were trained for 32 epochs (according to original clip paper, appendix F, and used 400m samples per epoch)
        model_info['epochs']
    )

    model_info['samples_per_epoch'] = np.where(
        model_info['other_info'] == 'openai',
        400000000, # Models were trained for 32 epochs (according to original clip paper, appendix F, and used 400m samples per epoch)
        model_info['samples_per_epoch']
    )

    # cherti et al. performance info
    cherti_performance = pd.read_csv(os.path.join('scaling-laws-openclip','zeroshot_results.csv'))
    cherti_performance = cherti_performance.pivot(index=['arch','name'],
                            values=['acc1','acc5', 'mean_per_class_recall', 'image_retrieval_recall@5',
                                    'text_retrieval_recall@5','mean_average_precision'],
                            columns=['downstream_dataset']) # Pivot to have one row per model
    cherti_performance = cherti_performance.dropna(axis=1, how='all').reset_index() # drop empty columns
    cherti_performance = cherti_performance[[('arch', ''), ('name', ''), ('acc1', 'vtab'), ('acc1', 'vtab+')]]  # rename
    cherti_performance.columns = ['architecture', 'model_name', 'vtab', 'vtab+'] # rename columns
    cherti_performance = cherti_performance[cherti_performance['model_name'] != 'openai']


    openclip_local_performance = pd.read_csv('results/performance_evaluation/benchmark.csv').drop_duplicates()
    openclip_local_performance = openclip_local_performance.pivot(index=['model','model_fullname', 'pretrained'],
                                                  values=['acc1', 'acc5', 'mean_per_class_recall',
                                                          'image_retrieval_recall@5',
                                                          'text_retrieval_recall@5', 'mean_average_precision'],
                                                  columns=['dataset'])  # Pivot to have one row per model
    openclip_local_performance = openclip_local_performance.dropna(axis=1, how='all').reset_index()  # drop empty columns
    openclip_local_performance = openclip_local_performance.rename(
        columns={'model': 'architecture', 'pretrained': 'model_name'})  # rename columns

    # open_clip performance info
    openclip_performance = pd.read_csv('https://raw.githubusercontent.com/mlfoundations/open_clip/ebe135b23161e375892acf72a1ee884e03976ab8/docs/openclip_results.csv')
    openclip_performance = openclip_performance.rename(columns={'name': 'architecture', 'pretrained': 'model_name'})
    openclip_performance = openclip_local_performance.merge(openclip_performance, on=['architecture', 'model_name'])

    vtab_tasks = [
        'Caltech-101',
        'CIFAR-100',
        'CLEVR Counts',
        'CLEVR Distance',
        'Describable Textures',
        'EuroSAT',
        'KITTI Vehicle Distance',
        'Oxford-IIIT Pet',
        'Oxford Flowers-102',
        'PatchCamelyon',
        'RESISC45',
        'SUN397',
        'SVHN',
        ('acc1', 'wds/vtab/diabetic_retinopathy'),
        ('acc1', 'wds/vtab/dmlab'),
        ('acc1', 'wds/vtab/dsprites_label_orientation'),
        ('acc1', 'wds/vtab/dsprites_label_x_position'),
        ('acc1', 'wds/vtab/dsprites_label_x_position'),
        ('acc1', 'wds/vtab/smallnorb_label_azimuth'),
    ]


    vtab_plus_tasks = [
        'Caltech-101',
        'CIFAR-10',
        'CIFAR-100',
        'CLEVR Counts',
        'CLEVR Distance',
        'Describable Textures',
        'EuroSAT',
        'KITTI Vehicle Distance',
        'Oxford-IIIT Pet',
        'Oxford Flowers-102',
        'PatchCamelyon',
        'RESISC45',
        'SUN397',
        'SVHN',
        'ImageNet 1k',
        'ImageNet v2',
        'ImageNet Sketch',
        'ImageNet-A',
        'ImageNet-R',
        'ObjectNet',
        'Pascal VOC 2007',
        'Stanford Cars',
        'FGVC Aircraft',
        'MNIST',
        'STL-10',
        'GTSRB',
        'Country211',
        'Rendered SST2',

        ('acc1', 'wds/vtab/diabetic_retinopathy'),
        ('acc1', 'wds/vtab/dmlab'),
        ('acc1', 'wds/vtab/dsprites_label_orientation'),
        ('acc1', 'wds/vtab/dsprites_label_x_position'),
        ('acc1', 'wds/vtab/dsprites_label_x_position'),
        ('acc1', 'wds/vtab/smallnorb_label_azimuth'),

        ('acc1', 'wds/fer2013'),
    ]

    openclip_performance['vtab'] = openclip_performance[vtab_tasks].mean(axis=1)
    openclip_performance['vtab+'] = openclip_performance[vtab_plus_tasks].mean(axis=1)
    openclip_performance['model_fullname'] = openclip_performance[('model_fullname', '')]
    openclip_performance = openclip_performance[['architecture', 'model_fullname', 'vtab', 'vtab+']].rename(columns={'model_fullname':'model_name'})
    openclip_performance['model_name'] = openclip_performance['model_name'].str.replace(' ', '_')


    all_performance = pd.concat([cherti_performance,
                                 openclip_performance]).reset_index(drop=True)
    all_performance = all_performance[~(all_performance['model_name']).isin(repeated_between_modelsets)]
    all_performance = all_performance[~(all_performance['model_name']).isin(quickgelu_repeats)]

    model_info = model_info.merge(all_performance, on=['architecture', 'model_name'], how='outer')

    model_info['model_name'] = np.where(
        model_info['model_name'].str.slice(-3) != '.pt',
        model_info['model_name'],
        model_info['architecture'] +'_scaling-laws-openclip'+ model_info['model_name']
    )

    model_info = model_info.drop(columns=['lr','warmup','gpus','samples_per_sec','local_bs', 'detailed_training_info','url'])


    # condense multi-layer headings
    model_info.columns = model_info.columns.map(''.join)

    return model_info


def find_correlations(models_to_withold):
    sorted_images = image_biases.sort_values(['X', 'Y', 'A', 'B', 'Attribute', 'Target', 'Test', 'effect_size'])
    # sorted_images = sorted_images[~sorted_images['model'].isin(models_to_withold)]
    sorted_text = text_biases.sort_values(['X', 'Y', 'A', 'B', 'Test', 'test_name', 'effect_size'])
    # sorted_text = sorted_text[~sorted_text['model'].isin(models_to_withold)]
    all_cors = []
    names = []
    for m1 in sorted_images['model'].unique():
        if 'frozen' in m1 or '_e31' in m1:
            continue
        m1_cors = []
        names.append(m1)
        for m2 in sorted_images['model'].unique():
            if 'frozen' in m2 or '_e31' in m2:
                continue
            d1 = sorted_images[sorted_images['model'] == m1]['effect_size']
            d2 = sorted_images[sorted_images['model'] == m2]['effect_size']
            d3 = sorted_text[sorted_text['model'] == m1]['effect_size']
            d4 = sorted_text[sorted_text['model'] == m2]['effect_size']
            m1_cors.append(max(np.corrcoef(d1, d2)[0, 1], np.corrcoef(d3, d4)[0, 1]))
        all_cors.append(m1_cors, )
    c = pd.DataFrame(np.array(all_cors), columns=names, index=names)
    c = pd.melt(c.reset_index(), id_vars='index')
    c = c[c['index'] != c['variable']].sort_values('value', ascending=False)
    print(c)

if __name__ == '__main__':
    # image_biases = load_image_bias_results()
    # text_biases = load_text_bias_results()
    # full_biases = pd.concat([image_biases, text_biases], axis=0)
    # drop na columns
    # full_biases = full_biases.dropna(axis=1)
    # full_biases = full_biases.drop(columns=['X','Y','A','B','nt','na', 'npermutations'])

    # cross_modal_results = load_cross_modal_bias()
    model_info = load_model_info(False)
    full_biases = full_biases.merge(model_info, left_on='model', right_on='model_name')
    # cross_modal_results = cross_modal_results.merge(model_info, left_on='model', right_on='model_name')

    full_biases.to_csv(os.path.join('results', 'data', 'unimodal_data_for_modeling.csv'), index=False)
    # cross_modal_results.to_csv(os.path.join('results', 'data', 'bimodal_data_for_modeling.csv'), index=False)
