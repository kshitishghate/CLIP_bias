import math
import os
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns




all_results = pd.read_csv(os.path.join('results','data','cross_modal.csv'))

all_results = all_results[['Image Test','Text Test','image_target_1',
       'image_target_2', 'image_attribute_1',
       'image_attribute_2', 'text_target_1', 'text_target_2',
       'text_attribute_1', 'text_attribute_2', 'order','effect_size','model']]

averaged_results = all_results.groupby(['Image Test','Text Test','image_target_1',
       'image_target_2', 'image_attribute_1',
       'image_attribute_2', 'text_target_1', 'text_target_2',
       'text_attribute_1', 'text_attribute_2', 'order'])['effect_size'].mean().reset_index()

averaged_results = averaged_results.pivot(
    index=[c for c in averaged_results.columns if c not in ['order','effect_size']],
    columns='order',
    values='effect_size').reset_index()


all_results = all_results.pivot(
    index=[c for c in all_results.columns if c not in ['order','effect_size']],
    columns='order',
    values='effect_size').reset_index()


ieat_results = pd.read_csv(os.path.join('results','data','ieat_replication.csv'))
ieat_results['Test'] = ieat_results['Target'] + '/' + ieat_results['A'] + ' vs. ' + ieat_results['B']
ieat_results['Test'] = ieat_results['Test'].str.replace('Pleasant vs. Unpleasant', 'Valence',regex=False)
ieat_results['Test'] = ieat_results['Test'].str.replace('Weapon/Tool vs. Weapon', 'Race/Tool vs. Weapon',regex=False)
ieat_results['Test'] = ieat_results['Test'].str.replace('Weapon/Tool-modern vs. Weapon-modern', 'Race/Tool vs. Weapon (Modern)',regex=False)
ieat_results['Test'] = ieat_results['Test'].str.replace('Gender/Science vs. Liberal-Arts', 'Gender/Science vs. Arts',regex=False)
ieat_results = ieat_results.rename(columns={'effect_size':'all_images'})

all_results = all_results.merge(ieat_results, left_on=['Image Test', 'model'], right_on=['Test','model'])
ieat_results = ieat_results.groupby(['Test','Target', 'X', 'Y', 'A', 'B', 'nt', 'na', 'Attribute'])['all_images'].mean().reset_index()

averaged_results = averaged_results.merge(ieat_results, left_on='Image Test', right_on='Test')



seat_results = pd.read_csv(os.path.join('results','data','seat_replication.csv'))
seat_results['Test'] = seat_results["test_name"].str.replace('.jsonl','',regex=False).str.replace('_',' ',regex=False).str.title()
seat_results['Test'] = seat_results['Test'].str.strip()
seat_results = seat_results[seat_results['Test'].str.contains('Weat')]
seat_results = seat_results.rename(columns={'effect_size':'all_text'})
seat_results['test_name'] = seat_results['test_name'].str.upper().str.replace('.JSONL','',regex=False)

all_results = all_results.merge(seat_results, left_on=['Text Test','model'], right_on=['test_name','model'])
seat_results = seat_results.groupby(['test_name', 'X', 'Y', 'A', 'B'])['all_text'].mean().reset_index()


averaged_results = averaged_results.merge(seat_results, left_on='Text Test', right_on='test_name')

all_results = all_results[['Text Test', 'Image Test', 'X_x','Y_x','A_x','B_x','all_images', 'text as target','image as target', 'all_text']].rename(columns= lambda x: x.replace(' ','_'))
all_results['sign_flip'] = ((all_results['all_images'] >= 0) & (all_results['all_text'] < 0)) | ((all_results['all_text'] >= 0) & (all_results['all_images'] < 0))
all_results['text_target_outside_range'] = (
    ((all_results['text_as_target'] > all_results['all_text']) & (all_results['text_as_target'] > all_results['all_images']))
    | ((all_results['text_as_target'] < all_results['all_text']) & (all_results['text_as_target'] < all_results['all_images']))
)

all_results['image_target_outside_range'] = (
    ((all_results['image_as_target'] > all_results['all_text']) & (all_results['image_as_target'] > all_results['all_images']))
    | ((all_results['image_as_target'] < all_results['all_text']) & (all_results['image_as_target'] < all_results['all_images']))
)
all_results['either_cross_outside_range'] = all_results['image_target_outside_range'] | all_results['text_target_outside_range']
all_results['image_greater'] = all_results['all_images'] > all_results['all_text']
all_results['cross_modal_dif'] = np.abs(all_results['image_as_target'] - all_results['text_as_target'])
all_results['image_text_dif'] = np.abs(all_results['all_images'] - all_results['all_text'])

def plot_overall(df, y_axis_col):
    plt.clf()
    df = df.copy()

    def jitter(y, jitter_size=0.25):
        return y + random.uniform(0, jitter_size * 2) - jitter_size




    df[y_axis_col]  = range(len(df[y_axis_col]))




    fig, (ax) = plt.subplots(1, 1, figsize=(10, 6))


    plt.scatter(y=df[y_axis_col], x=df['text as target'], c='#66CCEE', marker='^')
    plt.scatter(y=df[y_axis_col], x=df['image as target'], c='#4477AA', marker='v')
    plt.scatter(y=df[y_axis_col], x=df['all_text'], c='#EE6677', marker='s')
    plt.scatter(y=df[y_axis_col], x=df['all_images'], c='#AA3377', marker='D')


    plt.axvline(x=0, color='black')

    # label y axis
    plt.yticks(range(len(df)))
    labs = list(df['Image Test'] + ' (' + df['Text Test'].str.lower() + ')')
    plt.gca().set_yticklabels(labs)



    # add legend
    handles = [
        plt.scatter([-10], [-100], c='#EE6677', marker='s', label='Both Text'),
        plt.scatter([-10], [-100],  c='#66CCEE', marker='^', label='Text Target, Image Attribute'),
        plt.scatter([-10], [-100], c='#4477AA', marker='v', label='Image Target, Text Attribute'),
        plt.scatter([-10], [-100], c='#AA3377', marker='D', label='Both Images'),
    ]
    ax.legend(handles=handles,
              # bbox_to_anchor=(0,0.85),
              loc='upper left')

    plt.xlim(-1.99, 1.99)
    plt.ylim(-0.5, len(df)-0.5)

    plt.tight_layout()
    plt.savefig(os.path.join('results','plots','cross_modal.pdf'))


if __name__ == '__main__':
    plot_overall(averaged_results, 'Test')









