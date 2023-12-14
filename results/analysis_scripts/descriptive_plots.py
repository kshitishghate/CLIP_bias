import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def jitter(values,j):
    return values + np.random.normal(j,0.5,values.shape)

plt.clf()
plt.style.use(os.path.join('results','plots', 'default.mplstyle'))

full_data = pd.read_csv('results/data/unimodal_data_for_modeling.csv')

model_data = full_data[['model','architecture','other_info','model_family','dataset','samples_seen','model_name',
                        'fine_tuned','epochs','samples_per_epoch','embed_dim','params','image_params','text_params',
                        'macs','acts','image_macs','image_acts','dataset_size','dataset_family',
                        'model_source']].drop_duplicates()

model_data['Samples Seen (Billions)'] = model_data['samples_seen'] / 1e9
model_data['GMACs per Sample'] = model_data['macs'] / 1e9

plt.clf()
scatter = sns.scatterplot(jitter(model_data['GMACs per Sample'], 0), jitter(model_data['Samples Seen (Billions)'],0),s=100,color='black',alpha=0.5)
plt.xscale('log')
scatter.xaxis.set_major_formatter(ScalarFormatter())
scatter.set_xticks([10, 50, 100, 500])
plt.tight_layout()
plt.savefig('results/plots/gmacs_and_samples.pdf')
