import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./data/summary_scale_25_03_17.csv')


df_melted = df.melt(id_vars=['topic_idx'], value_vars=[col for col in df.columns if col.endswith('_pr')],
                    var_name='model', value_name='prauc')
mapping_dict = {'./model_trained/os_covid_contrastive_e6_24_07_22_pr':'COVID-19 (Similarity: 0.80)', '/user/home/zx16649/study_screening/retrieval/model_trained/os_covid_3_24_11_08_pr':'selected COVID-19 topics (Similarity: 1)', 
                '/user/home/zx16649/study_screening/retrieval/model_trained/os_rct_24_10_17_pr':'COVID-19 + ES (Similarity: 0.55)', '/user/home/zx16649/study_screening/retrieval/model_trained/os_with_philippa_24_10_23_pr':'COVID-19 + ES + SYNERGY (Similarity: 0.48)'}
model_order = [
    'selected COVID-19 topics (Similarity: 1)',
    'COVID-19 (Similarity: 0.80)',
    'COVID-19 + ES (Similarity: 0.55)',
    'COVID-19 + ES + SYNERGY (Similarity: 0.48)',
]
df_melted['model'] = df_melted['model'].map(mapping_dict)

idx2topic = {11:'NMABS',13:'CSR',14:'IVM',15:'SSRI',16:'CQHCQ',17:'LPVR',26:'PMH'}
df_melted['topic_idx'] = df_melted['topic_idx'].map(idx2topic)
df_melted['model'] = pd.Categorical(df_melted['model'], ordered=True)
df_melted.at[0, 'prauc'] = np.nan
df_melted.at[1, 'prauc'] = np.nan
df_melted.at[4, 'prauc'] = np.nan
df_melted.at[6, 'prauc'] = np.nan
# Print the melted DataFrame
print("\nMelted DataFrame:")
print(df_melted)
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df_melted, x='topic_idx', y='prauc', hue='model')
for container in ax.containers:  # Iterate over each container (one per hue category)
    ax.bar_label(container, fmt="%.2f")  # Add labels with 2 decimal points

# Add labels and title
plt.xlabel('Topic')
plt.ylabel('PRAUC')
plt.title('Model Performance on different COVID-19 Topics')
plt.legend(title='Model')
plt.savefig('performance_scaling.png')

df_criteria = pd.read_csv('./data/summary_criteria_no_25_03_20.csv')
df_melted = df_criteria.melt(id_vars=['topic_idx'], value_vars=[col for col in df_criteria.columns if col.endswith('_pr') or col.endswith('ratio')],
                    var_name='model', value_name='prauc')
mapping_dict = {'./model_trained/os_covid_contrastive_e6_24_07_22_pr':'with criteria (free text)', '/user/work/zx16649/model_trained/os_covid_criteria/os_covid_no_criteria_24_10_15_pr':'no criteria', 
                '/user/work/zx16649/model_trained/os_covid_criteria/os_covid_structured_criteria_25_01_17_pr':'structured criteria', '/user/work/zx16649/model_trained/os_covid_criteria/os_covid_include_exclude_criteria_25_01_17_pr':'inclusion/exclusion criteria', 
                'pos_neg_ratio':'baseline'}
# df_melted['model'] = df_melted['model'].str.replace('_pr', '')

model_order = [
    'baseline',
    'no criteria',
    'with criteria (free text)',
    # 'structured criteria',
    # 'inclusion/exclusion criteria'
]

df_melted['model'] = df_melted['model'].map(mapping_dict)

idx2topic = {11:'NMABS',13:'CSR',14:'IVM',15:'SSRI',16:'CQHCQ',17:'LPVR',26:'PMH'}
df_melted['topic_idx'] = df_melted['topic_idx'].map(idx2topic)
# df_melted['model'] = pd.Categorical(df_melted['model'], categories=model_order, ordered=True)
df_melted['model'] = pd.Categorical(df_melted['model'], ordered=True)

# Print the melted DataFrame
print("\nMelted DataFrame:")
print(df_melted)
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df_melted, x='topic_idx', y='prauc', hue='model')
for container in ax.containers:  # Iterate over each container (one per hue category)
    ax.bar_label(container, fmt="%.2f")  # Add labels with 2 decimal points

# Add labels and title
plt.xlabel('Topic')
plt.ylabel('PRAUC')
plt.title('Model Performance on different COVID-19 Topics')
plt.legend(title='Model')

plt.savefig('performance_criteria.png')
# plt.show()