import os
import pickle

import numpy as np

import pandas as pd

from results.analysis_scripts.bundle_eat_data import load_model_info

from slur_distance_measurement import get_image_demos

def load_results():
    """Load results from results/slurs"""
    all_images, genders, races, indices_of_interest = get_image_demos()

    # Filter out any models we are not using
    model_info = load_model_info(include_num_params=False)
    all_results_fnames = [p for p in os.listdir(os.path.join('results', 'slurs'))
                          if p.endswith('.pkl')
                          and (
                                  p.replace(' ','_').replace('.pkl','') in model_info['model_name'].values
                                  or p.replace(' ','_').replace('.pkl','').replace('openclip_', 'openclip') in model_info['model_name'].values
                          )]

    # Get model names for joining to model_info
    all_model_names = []
    for fname in all_results_fnames:
        if '.pt' in fname:
            model_name = fname.replace(' ','_').replace('.pkl','').replace('openclip_', 'openclip')
        else:
            model_name = fname.replace(' ','_').replace('.pkl','')
        all_model_names.append(model_name)

    # load all results
    all_results = []
    for fname, mname in zip(all_results_fnames, all_model_names):
        with open(os.path.join('results', 'slurs', fname), 'rb') as f:
            result = pickle.load(f)
        processed_result = {'model_name': mname}

        for k, v in result.items():

            if k == 'gender':
                assert v == genders
                processed_result[k] = v
            elif k == 'race':
                assert v == races
                processed_result[k] = v
            elif k == 'image_fp':
                if len(v) != len(indices_of_interest):
                    processed_result[k] = [v for i, v in enumerate(v) if i in indices_of_interest]
                else:
                    processed_result[k] = v
            elif 'slur_' in k and ('_ethnicity' in k or '_gender' in k):
                pass
            elif len(v) == len(all_images):
                if '_result' in k:
                    col = result[k.replace('_result','')]
                else:
                    col = k
                processed_result[col] = v[indices_of_interest]
            elif len(v) == len(indices_of_interest):
                if '_result' in k:
                    col = result[k.replace('_result','')]
                else:
                    col = k
                processed_result[col] = v
        all_results.append(pd.DataFrame(processed_result))
    joined = pd.concat(all_results, axis=0)
    return joined


def load_slur_metadata():
    """Load slur metadata"""
    slur_metadata = pd.read_csv(os.path.join('results', 'data', 'filtered_slurs_with_metadata.csv'))

    # drop unnamed columns
    slur_metadata = slur_metadata.loc[:, ~slur_metadata.columns.str.contains('^Unnamed')]

    # Rename columns
    slur_metadata.columns=['slur','language','sighting_count','offensiveness','target']

    # Remove non digit characters from sighting_count
    slur_metadata['sighting_count'] = slur_metadata['sighting_count'].str.replace(r'\D', '', regex=True).str.replace('^$', '0', regex=True).astype(int)

    return slur_metadata






if __name__ == '__main__':
    all_distances = load_results()
    slur_metadata = load_slur_metadata()


    controls = ['Person', 'Human']

    all_distances['similarity_with_controls'] = (all_distances[controls].sum(axis=1)) / 2


    slices = ['similarity_with_controls']
    for i in [5, 10, 20, 50, 100]:
        most_common = slur_metadata.sort_values('sighting_count').iloc[-i:]['slur'].values
        all_distances[f'similarity_with_{i}_most_common'] = all_distances[most_common].mean(axis=1)
        all_distances[f'closer_to_controls_than_{i}_most_common'] = all_distances['similarity_with_controls'] > all_distances[f'similarity_with_{i}_most_common']
        slices += [f'similarity_with_{i}_most_common', f'closer_to_controls_than_{i}_most_common']


    agg = all_distances.groupby(['model_name','gender','race'])[slices].mean().reset_index()

    model_info = load_model_info()

    agg = agg.merge(model_info, on='model_name')
    agg.to_csv('results/data/aggregated_slur_distance_results.csv', index=False)

    all_distances = all_distances[['model_name', 'image_fp', 'race','gender'] + slices]
    all_distances = all_distances.merge(model_info, on='model_name')
    all_distances.to_csv('results/data/complete_slur_distance_results.csv', index=False)





    print('here')