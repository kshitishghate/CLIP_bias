import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results = pd.read_csv('results_backedup.csv')
results = results.rename(columns={'effect_size':'Effect Size','model':'Model','Title':'Test'})
fig = sns.catplot(data=results, x='Test', y='Effect Size', hue='Model', kind="swarm")

fig.ax.set_xticklabels(results['Test'].unique(), rotation=20, ha='right')
# plt.tight_layout()
plt.savefig('effect_sizes.png')



results = pd.read_csv('results_backedup.csv')
results = results.rename(columns={'pvalue':'P-Value','model':'Model','Title':'Test'})
fig = sns.catplot(data=results, x='Test', y='P-Value', hue='Model', kind="swarm")

fig.ax.set_xticklabels(results['Test'].unique(), rotation=20, ha='right')
# plt.tight_layout()
plt.savefig('pvals.png')

