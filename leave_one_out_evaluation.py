from model_functions import plot_roc, plot_precision_recall, predict_with_sbert
import pandas as pd
import os
import torch
from utils import read_criteria, read_data_os
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

hyperparam={'top_k':None,
            'max_length':512,
            'data_path':'./data/',
            'model_path':'<PATH_TO_TRAINED_MODELS>',
            'index_path':None,
            'criteria':True,
            'fig_path':'<PATH_TO_SAVE_FIGURES>',
            'prediction_path':'<PATH_TO_SAVE_PREDICTIONS>',
            'test': True}

train_file = 'train.csv'
test_file = 'test.csv'
criteria_file = 'criteria.csv'

train_query, train_corpus, train_label = read_data_os(hyperparam['data_path']+train_file, label_key='label_included')
test_query, test_corpus, test_label = read_data_os(hyperparam['data_path']+test_file, label_key='label_included')
df = pd.DataFrame(zip(test_query, test_corpus, test_label), columns=['q','d','l'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(hyperparam['prediction_path'], exist_ok=True)
os.makedirs(hyperparam['fig_path'], exist_ok=True)

grouped = df.groupby('q')
topics = list(grouped.groups.keys())


train_df = pd.DataFrame(zip(train_query, train_corpus, train_label), columns=['q','d','l'])
train_grouped = train_df.groupby('q')

selected_topics = ['Diferential efcacy and_afety of_nti_SARS_CoV_2 antibody therapies for_he_anagement of_OVID_19: a_ystematic review and_etwork meta_analysis','Efficacy and safety of corticosteroid regimens for the treatment of hospitalized COVID-19 patients: a meta-analysis','Efficacy and safety of ivermectin for the treatment of COVID-19: a systematic review and meta-analysis','Efficacy and safety of selective serotonin reuptake inhibitors in COVID-19 management: a systematic review and meta-analysis','Efficacy of chloroquine and hydroxychloroquine for the treatment of hospitalized COVID-19 patients: a meta-analysis','Efficacy of lopinavir_itonavir combination therapy for the treatment of hospitalized COVID-19 patients: a meta-analysis','Prevalence of mental health symptoms in children and adolescents during the COVID-19 pandemic: A meta-analysis'] # covid

plot_fpr, plot_tpr, plot_model = [],[],[]
plot_p, plot_r, performance_prauc = [], [], []
performance = []
# for query in topics:
idx2topic = {0:'NMABS',1:'CSR',2:'IVM',3:'SSRI',4:'CQHCQ',5:'LPVR',6:'PMH'}
# for idx, query in enumerate(selected_topics):
for idx, query in enumerate(topics):
    print(query)

    if hyperparam['test']:
        # for evaluating test set
        train_topic_group = train_grouped.get_group(query)
        topic_group = grouped.get_group(query)
    else:
        # for evaluating train set
        train_topic_group = train_grouped.filter(lambda x: x.name != query)
        topic_group = grouped.filter(lambda x: x.name != query)


    train_documents = train_topic_group['d'].tolist()
    train_labels = train_topic_group['l'].tolist()
    documents = topic_group['d'].tolist()
    labels = topic_group['l'].tolist()

    # combine data from train.csv and test.csv
    documents.extend(train_documents)
    labels.extend(train_labels)
    
    target_topic = query.split()[0]
    for dir in os.listdir(hyperparam['model_path']):
        if '_'+target_topic not in dir:
            continue
        print(dir)
        
        if hyperparam['criteria']:
            criteria = read_criteria(hyperparam['data_path']+criteria_file)
            if hyperparam['test']:
                # for test set
                q = 'Query: ' + query+'. Criteria: '+criteria[query]
            else:
                q = []
                for t in list(filter(lambda x: x != query, selected_topics)):
                    q.append(f'Query: {t}. Criteria: {criteria[t]}')
            
            model_name = hyperparam['model_path'] + dir
            
            p,r, _, prauc, fpr, tpr, _, predictions, auroc = predict_with_sbert(model_name,q,documents, hyperparam['top_k'], labels=labels, device=device, pr=True)
        
        plot_fpr.append(fpr)
        plot_tpr.append(tpr)
        # plot_model.append("{}, PRAUC: {:.2f}".format(idx2topic[idx],prauc))
        plot_model.append(target_topic)

        print('auroc: {}'.format(auroc))
        performance.append(auroc)

        plot_p.append(p)
        plot_r.append(r)
        # plot_model.append(target_topic)

        print('prauc: {}'.format(prauc))
        performance_prauc.append(prauc)

        if hyperparam['prediction_path']:
            prediction_name=hyperparam['prediction_path']+target_topic+'.csv'
            predictions.to_csv(prediction_name, index=False)
        torch.cuda.empty_cache() 

if hyperparam['prediction_path']:
    summary = pd.DataFrame(zip(selected_topics, performance, performance_prauc), columns=['query','auroc_leave_one_out','prauc_leave_one_out'])
    summary.to_csv(hyperparam['prediction_path']+'summary.csv',index=False)                                                                                  
if hyperparam['fig_path']:
    average_performance = sum(performance)/len(performance)
    # t = 'Leave One Out (average auroc: ' + f"{average_performance:.2f}" + ')'
    t = 'Leave One Out AUROC'
    fig_name_roc = hyperparam['fig_path']+ hyperparam['dataset'] +'_leave_one_out_roc.png'
    plot_roc(fpr=plot_fpr, tpr=plot_tpr, model_name=plot_model, topic=t, fig_name=fig_name_roc)
    
    average_prauc = sum(performance_prauc)/len(performance_prauc)
    # t = 'Leave One Out (average prauc: ' + f"{average_prauc:.2f}" + ')'
    t = 'Leave One Out PRAUC'
    fig_name_pr = hyperparam['fig_path']+ hyperparam['dataset'] +'_leave_one_out_pr.png'
    plot_precision_recall(precision=plot_p, recall=plot_r, model_name=plot_model, topic=t, fig_name=fig_name_pr)


print('average auroc: {}'.format(average_performance))
    