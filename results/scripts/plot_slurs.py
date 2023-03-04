import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import seaborn as sns


plt.clf()
a = pd.read_csv('/Users/is28/Documents/Code/language_vision_bias/results/data/raw_distances.csv')
a['Slur > Man/Woman'] = np.where(a['gender'] == 'Female', a['slur'] > a['Woman'], a['slur'] > a['Man'])
a['Slur > Black Person'] = a['slur'] > a['Black Person']
a['Slur > African-American Person'] = a['slur'] > a['African-American Person']
xname = 'Average Percent of Images Closer to Slur Than to Word "Person" \n(With 95% CI)'
a[xname] = a['slur'] > a['Person']


b = a.groupby(['race','gender','model_name']).mean().reset_index()
b['Intersectional Group'] = b['race'] + ', ' + b['gender']
b[xname] = b[xname] * 100
b = b.rename(columns={'race':'Ethnicity','gender':'Gender'})

sns.catplot(
    data=b, x=xname, y='Ethnicity', hue="Gender",
    kind="bar",
)


plt.xlim(0,100)
plt.savefig(os.path.join('results/plots/slur_pcts.pdf'))