import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

seed=42

def read_criteria(file):
    if '.txt' in file:
        with open(file, 'r') as f:
            lines = f.readlines()
        criteria = ''.join(line.strip() for line in lines)
    else:
        df = pd.read_csv(file)
        c = df['criteria']
        t = df['topic']
        criteria = {k:v for k,v in zip(t,c)}
     
    return criteria


def combine_title_abstract(title, abstract):
    corpus = ['Title: ' + str(t) + '. Abstract: ' + str(a) for t,a in zip(title,abstract)]
    return corpus


def read_data_os(file, query_key='query', title_key='title', abstract_key='abstract', label_key=None):
    df = pd.read_csv(file)
    query = df[query_key].tolist()
    # formated_query = ['Query: ' + q for q in query]
    title = df[title_key].tolist()
    abstract = df[abstract_key].tolist()
    corpus = combine_title_abstract(title, abstract)
    if label_key:
        label = df[label_key].tolist()
        return query, corpus, label
    return query, corpus


def merge_prediction_results(df,new_results,model_name):
    merged_df = pd.merge(df, new_results, on=['idx', 'study', 'ground_truth'], how='left')
    merged_df.rename(columns={'similarity': model_name}, inplace=True)
    return merged_df


def calculate_pos_ratio(true_labels):
    pos_count = len([x for x in true_labels if x==1])
    neg_count = len(true_labels) - pos_count
    ratio =  pos_count / len(true_labels)
    return pos_count, neg_count, ratio

    df= pd.read_csv(data_path)
    
    if criteria_path:
        criteria = read_criteria(criteria_path)
        df['topic'] = 'Topic: ' + df['query'] + '. Criteria: ' +  df['query'].map(criteria).fillna('')
    else:
        df['topic'] = 'Topic: ' + df['query']
    df['study'] = 'Title: ' + df['title'].astype(str) + '. Abstract: ' + df['abstract'].astype(str)
    
    if skip_topic:
        train_df = df[~df['query'].isin(skip_topic)]
    else:
        train_df = df
    
    anchors = train_df['topic'].unique().tolist()
    
    df_label_0 = train_df[train_df['label_included'] == 0]
    df_label_1 = train_df[train_df['label_included'] == 1]
    
    # Group by 'group_column' and aggregate 'value_column' into lists for label 0
    grouped_label_0 = df_label_0.groupby('query')['study'].apply(list).reset_index()
    negatives = grouped_label_0['study'].tolist()
    
    # Group by 'group_column' and aggregate 'value_column' into lists for label 1
    grouped_label_1 = df_label_1.groupby('query')['study'].apply(list).reset_index()
    positives = grouped_label_1['study'].tolist()
    
    return anchors, positives, negatives

def recall_by_class(tn, fp, fn, tp):
    recall_pos = tp/(tp+fn)
    recall_neg = tn/(tn+fp)
    return recall_pos, recall_neg
    

def evaluate_classification(gt, prediction):
    tn, fp, fn, tp = confusion_matrix(gt, prediction).ravel()
    accuracy = accuracy_score(gt, prediction)
    precision = precision_score(gt, prediction)
    recall = recall_score(gt, prediction)
    recall_pos, recall_neg = recall_by_class(tn, fp, fn, tp)
    f1 = f1_score(gt, prediction)
    specificity = tn / (tn + fp)
    return tn, fp, fn, tp, accuracy, precision, recall, recall_pos, recall_neg, f1, specificity

