from model_functions import plot_roc, plot_precision_recall, roc_for_classification, predict_with_sbert
import pandas as pd
import os
import torch
from utils import read_criteria, read_data_os, merge_prediction_results, calculate_pos_ratio
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

hyperparam={'top_k':None,
            'max_length':512,
            'models':['<PATH_TO_TRAINED_MODEL>'],
            'data_path':'<PATH_TO_DATA>',
            'index_path':None,
            'prediction_path':'<PATH_TO_PREDICTION_RESULTS>', 
            'fig_path':None, # add a path to save the figures
            'criteria':True, # True to add criteria in the systematic review topic inputs
            'summary':True}

train_file = 'tv_train.csv'
test_file = 'test.csv'
criteria_file = 'criteria.csv'
structured_criteria_file = 'criteria_structured.csv'
include_exclude_criteria_file = 'criteria_inclusion_exclusion.csv'

train_query, train_corpus, train_label = read_data_os(hyperparam['data_path']+train_file, label_key='label_included')
test_query, test_corpus, test_label = read_data_os(hyperparam['data_path']+test_file, label_key='label_included')
df = pd.DataFrame(zip(test_query, test_corpus, test_label), columns=['q','d','l'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if hyperparam['prediction_path']:
    os.makedirs(hyperparam['prediction_path'], exist_ok=True)
if hyperparam['fig_path']:
    os.makedirs(hyperparam['fig_path'], exist_ok=True)


grouped = df.groupby('q')
topics = list(grouped.groups.keys())


train_df = pd.DataFrame(zip(train_query, train_corpus, train_label), columns=['q','d','l'])
train_grouped = train_df.groupby('q')

# select_topics = ['Diferential efcacy and_afety of_nti_SARS_CoV_2 antibody therapies for_he_anagement of_OVID_19: a_ystematic review and_etwork meta_analysis','Efficacy and safety of corticosteroid regimens for the treatment of hospitalized COVID-19 patients: a meta-analysis','Efficacy and safety of ivermectin for the treatment of COVID-19: a systematic review and meta-analysis','Efficacy and safety of selective serotonin reuptake inhibitors in COVID-19 management: a systematic review and meta-analysis','Efficacy of chloroquine and hydroxychloroquine for the treatment of hospitalized COVID-19 patients: a meta-analysis','Efficacy of lopinavir_itonavir combination therapy for the treatment of hospitalized COVID-19 patients: a meta-analysis','Prevalence of mental health symptoms in children and adolescents during the COVID-19 pandemic: A meta-analysis'] # covid
skip_topics = ['Efficacy of therapeutic plasma exchange for treatment of autoimmune hemolytic anemia: A systematic review and meta-analysis of randomized controlled trials']
select_topics = ['Efficacy and safety of ivermectin for the treatment of COVID-19: a systematic review and meta-analysis','Efficacy and safety of selective serotonin reuptake inhibitors in COVID-19 management: a systematic review and meta-analysis','Efficacy of lopinavir_itonavir combination therapy for the treatment of hospitalized COVID-19 patients: a meta-analysis'] # covid 3
summary = []
column_name = ['topic_idx', 'topic', 'size', 'positive_size','negative_size','pos_neg_ratio']
initial_column_number = len(column_name)
count = 0

for i, query in enumerate(topics):
    performance = []
    topic_nature = []
    topic_nature.append(i)
    topic_nature.append(query)
    if query not in select_topics:
        continue
    if query in skip_topics:
        continue
    print(query)

    plot_fpr, plot_tpr, plot_model = [],[],[]
    plot_p, plot_r = [], []

        
    train_data_group = train_grouped.get_group(query)
    train_data = train_data_group['d'].tolist()
    train_label = train_data_group['l'].tolist()

    topic_group = grouped.get_group(query)
    documents = topic_group['d'].tolist()
    labels = topic_group['l'].tolist()
    
    p, r, _, prauc, fpr, tpr, _, predictions_df, auroc = roc_for_classification(train_data, train_label, documents, labels, device, pr=True, save_prediction=hyperparam['prediction_path'])
    
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    
    for dir in hyperparam['models']:
        print(dir)
        
        if hyperparam['criteria']:
            if 'structured' in dir:
                criteria = read_criteria(hyperparam['data_path']+structured_criteria_file)
            elif 'include_exclude' in dir:
                criteria = read_criteria(hyperparam['data_path']+include_exclude_criteria_file)
            else:
                criteria = read_criteria(hyperparam['data_path']+criteria_file)

            if 'no_criteria' in dir:
                q = 'Query: ' + query
            else:
                q = 'Query: ' + query+'. Criteria: '+criteria[query]
        else:
            q = 'Query: ' + query
        print(q)
    
        p,r, _, prauc, fpr, tpr, _, predictions, auroc = predict_with_sbert(model_name, q,documents, hyperparam['top_k'], labels=labels, device=device, pr=True, binary_threshold=0.5)
        
        plot_fpr.append(fpr)
        plot_tpr.append(tpr)
        plot_model.append(dir)
        plot_p.append(p)
        plot_r.append(r)
        print('auroc: {}, prauc: {}'.format(auroc,prauc))
        if count == 0:
            column_name.extend([f"{dir}_roc",f"{dir}_pr"])
        performance.extend([auroc,prauc])
        if hyperparam['prediction_path']:
            predictions_df = merge_prediction_results(predictions_df, predictions, model_name=dir)

        # del model
        # del tokenizer
        torch.cuda.empty_cache() 
        

    test_size = len(documents)
    pos_count, neg_count, positive_ratio = calculate_pos_ratio(labels)
    topic_nature.extend([pos_count+neg_count, pos_count, neg_count, positive_ratio])
    data_to_summary = topic_nature+performance 
    summary.append(data_to_summary)
    print(summary)
    print(column_name)
    
    t = f"Topic {i}"
    fig_title = t + ', size = ' + str(pos_count+neg_count) + ' (' + str(pos_count) + ':' + str(neg_count) + ')'
    print(t)
    if hyperparam['fig_path']:
        
        fig_name_roc = hyperparam['fig_path']+ t +'_roc.png'
        fig_name_pr = hyperparam['fig_path']+ t +'_pr.png'
        
        plot_roc(fpr=plot_fpr, tpr=plot_tpr, model_name=plot_model, topic=fig_title, fig_name=fig_name_roc)

        plot_precision_recall(precision=plot_p, recall=plot_r, model_name=plot_model, topic=fig_title, positive_ratio= positive_ratio, fig_name=fig_name_pr)
    if hyperparam['prediction_path']:
        prediction_name=hyperparam['prediction_path']+t+'.csv'
        predictions_df.to_csv(prediction_name, index=False)
        # predictions_df = predictions_df.sort_values(by='general_model', ascending=False, ignore_index=True)
        # predictions_df = predictions_df.drop('idx', axis=1)
        # predictions_df.to_csv(prediction_name, index=True, index_label='rank')
    count += 1

        
if hyperparam['prediction_path'] and hyperparam['summary']:
    summary_df = pd.DataFrame(summary, columns=column_name)
    summary_df.to_csv(hyperparam['prediction_path']+'summary.csv', index=False)
        
        