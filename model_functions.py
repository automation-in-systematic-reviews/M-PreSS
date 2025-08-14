import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import TypeVar
import csv
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import faiss

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
T = TypeVar('T')


def get_label_from_dataloader(dataloader):
    label = []
    for batch in dataloader:
        l = batch['label']
        label.extend(l.tolist())
    
    return label


def embed_paired_text(model, dataloader, device='cpu'):
    embeddings_1, embeddings_2 = [], []
    with torch.no_grad():
        for batch in dataloader:
            # print("speed test")
            # e_1 = model(batch['input_ids1'].to(device), batch['attention_mask1'].to(device))
            # e_2 = model(batch['input_ids2'].to(device), batch['attention_mask2'].to(device))
            e_1,e_2 = model(batch['input_ids1'].to(device), batch['attention_mask1'].to(device), batch['input_ids2'].to(device), batch['attention_mask2'].to(device))
            embeddings_1.append(e_1)
            embeddings_2.append(e_2)
        
    return torch.cat(embeddings_1, dim=0).cpu().numpy(), torch.cat(embeddings_2, dim=0).cpu().numpy()


def find_best_threshold(value, label, strategy='best_acc', similarity=True):
    if similarity:
        sorted_value, sorted_label = zip(*sorted(zip(value, label),reverse=True))
        bad_first_sorted_value, bad_first_sorted_label = zip(*sorted(zip(value, label),reverse=False))
    else:
        sorted_value, sorted_label = zip(*sorted(zip(value, label),reverse=False))
        bad_first_sorted_value, bad_first_sorted_label = zip(*sorted(zip(value, label),reverse=True))
    threshold = -1
    max_performance = 0
    
    if strategy == 'best_acc':
        pos=0
        remain_neg = label.count(0)
        for i in range(len(sorted_label)-1):
            if sorted_label[i] == 1:
                pos += 1
            else:
                remain_neg -= 1
            acc = (pos+remain_neg)/len(sorted_label)
            if acc > max_performance:
                max_performance = acc
                threshold = (sorted_value[i]+sorted_value[i+1])/2
    elif strategy == 'best_recall':
        ncorrect = 0
        positive_true = sum(label)
        for i in range(len(sorted_label)-1):
            if sorted_label[i] == 1:
                ncorrect += 1
            if ncorrect > 0:
                recall = ncorrect / positive_true
                if recall > max_performance:
                    max_performance = recall
                    threshold = (sorted_value[i]+sorted_value[i+1])/2
    elif strategy == 'best_precision':
        nextract = 0
        ncorrect = 0
        for i in range(len(sorted_label)-1):
            nextract += 1
            if sorted_label[i] == 1:
                ncorrect += 1
            if ncorrect > 0:
                precision = ncorrect / nextract
                if precision >= max_performance:
                    max_performance = precision
                    threshold = (sorted_value[i]+sorted_value[i+1])/2
    elif strategy == 'best_npv':
        nfalse_negatives = 0  # Count of false negatives (FN)
        ntrue_negatives = 0  # Count of true negatives (TN)
        total_negatives = sum([1 for x in bad_first_sorted_label if x == 0])  # Total negatives in the dataset
        total_positives = len(bad_first_sorted_label) - total_negatives  # Total positives in the dataset
        
        for i in range(len(bad_first_sorted_label)-1):
            # Evaluate the current instance as either FN or TN based on the label
            if bad_first_sorted_label[i] == 1:
                nfalse_negatives += 1  # This is a false negative if predicted as 0 (below the threshold)
            else:
                ntrue_negatives += 1  # This is a true negative if predicted as 0 (below the threshold)
            
            # Compute NPV only if we have non-zero True Negatives
            if ntrue_negatives + nfalse_negatives > 0:
                npv = ntrue_negatives / (ntrue_negatives + nfalse_negatives)
                if npv >= max_performance:
                    max_performance = npv
                    threshold = (bad_first_sorted_value[i] + bad_first_sorted_value[i+1]) / 2
    elif strategy == 'best_f1':
        nextract = 0
        ncorrect = 0
        positive_true = sum(label)

        for i in range(len(sorted_label) - 1):
            nextract += 1
            if sorted_label[i] == 1:
                ncorrect += 1
            if ncorrect > 0:
                precision = ncorrect / nextract
                recall = ncorrect / positive_true
                f1 = 2 * precision * recall / (precision + recall)
                if f1 > max_performance:
                    max_performance = f1
                    threshold = (sorted_label[i] + sorted_label[i + 1]) / 2
        
            
    return threshold, max_performance


def save_retrieval_results(file: str, results:list):
    if not os.path.exists(file_path):
        # Create the CSV file
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Test_name', 'k', 'dcg@k', 'threshold', 'Accuracy@k', 'Precision@k', 'Recall@k', 'F1@k'])
        print(f"CSV file '{file_path}' created.")
    with oepn(file_path, mode='a',newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(results)
        print(f"Results written to '{file_path}'")
    


def plot_size_acc(size, acc, label, save_path=None):
    for x, y, label in zip(size, acc, label):
        plt.scatter(x, y, label=label)

    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.grid(True)
    
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    plt.title('Training Size vs. Accuracy (Incidence)')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()


def calculate_roc(true_labels: list, y_scores: list):
    fpr, tpr, thresholds = roc_curve(true_labels, y_scores)
    return fpr, tpr, thresholds

def calculate_auroc(true_labels, y_scores):
    auroc = roc_auc_score(true_labels, y_scores)
    return auroc


def calculate_pr_auc(recall, precision):
    return auc(recall, precision)


def plot_roc(fpr: list, tpr: list, model_name: list, topic: str, fig_name='roc.png'):
    plt.figure()
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Reference Line')
    
    for f, t, name in zip(fpr, tpr, model_name):
        plt.plot(f, t, label=name)
    plt.legend(loc='upper right')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = 'ROC Curve: ' + topic
    plt.title(title)
    plt.grid(True)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.savefig(fig_name)
    plt.close()

def calculate_precision_recall(true_labels, y_scores):
    precision, recall, thresholds = precision_recall_curve(true_labels, y_scores)
    return precision, recall, thresholds
    
def plot_precision_recall(precision, recall, model_name, topic, positive_ratio=None, fig_name='roc.png'):
    plt.figure()
    if positive_ratio:
        plt.plot([0, 1], [positive_ratio, positive_ratio], color='navy', lw=2, linestyle='--', label='Reference Line')
    for p, r, name in zip(precision, recall, model_name):
        plt.plot(r, p, label=name)
    
    plt.legend(loc='upper right')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    title = 'Precision-Recall Curve: ' + topic
    plt.title(title)
    plt.grid(True)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.savefig(fig_name)
    plt.close()


def search_top_k_results(query, corpus, k, index_path=None):
    def normalize_L2(data):
        """ Normalizing the L2 norm of the data to 1 """
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        return data / norms
    
    if not k:
        k=len(corpus)
    faiss.normalize_L2(query)
    faiss.normalize_L2(corpus)
    
    d = corpus.shape[1]
    index = faiss.IndexFlatIP(d) # normalised inner product = cosine similarity
    # index_gpu = faiss.index_cpu_to_all_gpus(index)

    index.add(corpus)
    distance, indices = index.search(query, k)
    
    if index_path:
        faiss.write_index(index, index_path)
    return distance[0], indices[0]
    '''
    reshaped_query = query.reshape(1,-1)
    similarity_scores = cosine_similarity(reshaped_query, corpus)[0]  # [0] to get the similarity scores as a 1D array

    # Get the indices of the top-k scores in descending order
    top_k_indices = np.argsort(similarity_scores)[::-1][:k]
    top_k_scores = similarity_scores[top_k_indices]
    return top_k_scores, top_k_indices
    '''


def threshold_to_binary_labels(predicted_value, threshold=0.5, similarity=True):
    binary_labels = []
    if similarity:
        smaller_than_threshold = 0
        larger_than_threshold = 1
    else:
        smaller_than_threshold = 1
        larger_than_threshold = 0
    for v in predicted_value:
        if v <= threshold:
            binary_labels.append(smaller_than_threshold)
        else:
            binary_labels.append(larger_than_threshold)
    return binary_labels

    '''
    actual prediction for each topic
    :param query: tokenized query
    :param documents: documents Dataset
    :param labels: list of true label
    '''
    model.eval()
    
    document_dataloader = DataLoader(documents, batch_size=128, shuffle=False)
    with torch.no_grad():
        q_e = model(**query)
        d_e = []
        for batch in document_dataloader:
            e = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            d_e.append(e)
        d_e = torch.cat(d_e, dim=0)

    distance,indices = search_top_k_results(q_e.cpu().numpy(),d_e.cpu().numpy(),top_k,index_path)
    idx_to_label = {i:l for i, l in enumerate(labels)}
    sorted_labels = [idx_to_label[i] for i in indices]
    fpr, tpr, thresholds = calculate_roc(sorted_labels, distance)
    auroc = calculate_auroc(sorted_labels, distance)    
    
    
    if save_prediction:
        idx_to_document = {i:doc for i, doc in enumerate(documents)}
        data_to_save = []
        for idx, d in zip(indices, distance):
            data_to_save.append([idx, idx_to_document[idx], d])
        predictions = pd.DataFrame(data_to_save,columns=['idx', 'study', 'similarity'])
        if labels:
            predictions['gt']=sorted_labels
    else:
        predictions = []

    if pr:
        p,r,thresholds_pr = calculate_precision_recall(sorted_labels, distance)
        prauc = calculate_pr_auc(r,p)
        return p,r,thresholds_pr, prauc, fpr, tpr, thresholds, predictions, auroc
            
    return fpr, tpr, thresholds, predictions, auroc

def predict_with_sbert(model_name, query,documents,top_k, device= 'cpu', labels=None, pr=False, index_path=None, binary_threshold=0.5, model=None):
    if model_name:
        model = SentenceTransformer(model_name, device=device)
    print("Max Sequence Length:", model.max_seq_length)
    q_e = model.encode(query, device=device, convert_to_numpy=True)
    if type(query) == str:
        q_e = q_e.reshape(1,768)
    d_e = model.encode(documents)

    distance,indices = search_top_k_results(q_e,d_e,top_k,index_path)
    idx_to_label = {i:l for i, l in enumerate(labels)}
    sorted_labels = [idx_to_label[i] for i in indices]
    fpr, tpr, thresholds = calculate_roc(sorted_labels, distance)
    auroc = calculate_auroc(sorted_labels, distance)    

    
    idx_to_document = {i:doc for i, doc in enumerate(documents)}
    data_to_save = []
    for idx, d in zip(indices, distance):
        data_to_save.append([idx, idx_to_document[idx], d])
    predictions = pd.DataFrame(data_to_save,columns=['idx', 'study', 'similarity'])
    similarity = predictions['similarity'].tolist()
    # binary_results = threshold_to_binary_labels(similarity, binary_threshold)
    # key_name = f'general_model(threshold={binary_threshold})'
    # predictions[key_name] = binary_results
    if labels:
        predictions['ground_truth']=sorted_labels

    if pr:
        p,r,thresholds_pr = calculate_precision_recall(sorted_labels, distance)
        prauc = calculate_pr_auc(r,p)
        return p,r,thresholds_pr, prauc, fpr, tpr, thresholds, predictions, auroc

    
    return fpr, tpr, thresholds, predictions, auroc

 
def get_bert_embeddings_with_sbert(texts, model_name, seq_length, device='cpu'):
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = seq_length
    embeddings = model.encode(texts, device=device, convert_to_tensor=True)

    return embeddings

def bert_classifier(train_data, train_label, model_name, batch_size, seq_length, epoch, learning_rate, device='cpu'):
    
    # train_embeddings = get_bert_embeddings(train_data, model_name, batch_size, seq_length, device)
    train_embeddings = get_bert_embeddings_with_sbert(train_data, model_name, seq_length, device)
    # print(train_embeddings[:5])
    train_dataset = TensorDataset(train_embeddings.cpu(), torch.tensor(train_label).float().unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # classifier
    classifier = nn.Sequential(
        nn.Linear(train_embeddings.size(1), 1),
        nn.Sigmoid()
    )
    # classifier.apply(weights_init)
    
    if torch.cuda.device_count() > 1:
        classifier = torch.nn.parallel.DataParallel(classifier)
    classifier.to(device)
    # for name, param in classifier.named_parameters():
    #     print(f"{name}: {param.data}")

    criterion = nn.BCELoss() # Binary Cross Entropy
    optimizer = optim.AdamW(classifier.parameters(), lr=learning_rate)

    result = []
    # train loop
    classifier.train()
    for e in range(epoch):
        train_loss = 0.0
        for batch in train_loader:
            inputs, targets = (b.to(device) for b in batch)
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    # for name, param in classifier.named_parameters():
    #     print(f"{name}: {param.data}")
    return classifier


def roc_for_classification(train_data, train_label, test_data, test_label, device, test_type=None, test_pmid=None, pr=False, save_prediction=None):
    model_name = 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12'
    learning_rate = 5e-02
    seq_length = 512
    epoch = 6
    batch_size = 32
    print("classifier learning_rate: {}, epoch: {}".format(learning_rate,epoch))
    
    classifier = bert_classifier(train_data, train_label, model_name, batch_size=batch_size, seq_length=seq_length, epoch=epoch, learning_rate=learning_rate,device=device)
    
    # test_embeddings = get_bert_embeddings(test_data, model_name, batch_size, seq_length, device)
    test_embeddings = get_bert_embeddings_with_sbert(test_data, model_name, seq_length, device)
    test_dataset = TensorDataset(test_embeddings.cpu(), torch.tensor(test_label).float().unsqueeze(1))
    # test_embeddings = get_bert_embeddings(train_data, model_name, batch_size, seq_length, device)
    # test_dataset = TensorDataset(test_embeddings.cpu(), torch.tensor(train_label).float().unsqueeze(1))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    classifier.eval()
    results = []
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = (b.to(device) for b in batch)
            outputs = classifier(inputs)
            outputs = torch.flatten(outputs)
            results.extend(outputs.cpu().tolist())
    fpr,tpr, thresholds = calculate_roc(test_label, results)
    auroc = calculate_auroc(test_label, results)
    # fpr,tpr, thresholds = calculate_roc(train_label, results)
    del classifier
    torch.cuda.empty_cache()
    binary_results = threshold_to_binary_labels(results, threshold=0.5)
    
    if save_prediction:
        idx = [i for i in range(len(test_data))]
        if test_type:
            predictions = pd.DataFrame(zip(idx, test_data, test_pmid, test_type, test_label, results, binary_results), columns=['idx','study','pmid','type', 'ground_truth','topic_specific_model', 'topic_specific_model(threshold=0.5)'])
        else:
            predictions = pd.DataFrame(zip(idx, test_data, test_label, results, binary_results), columns=['idx','study', 'ground_truth','topic_specific_model', 'topic_specific_model(threshold=0.5)'])
        # idx = [i for i in range(len(train_data))]
        # predictions = pd.DataFrame(zip(idx, train_data, train_label, results), columns=['idx','study','gt','classifier_prediction'])
    else:
        predictions=[]
        
    if pr:
        p,r,thresholds_pr = calculate_precision_recall(test_label, results)
        prauc = calculate_pr_auc(r,p)
        return p,r,thresholds_pr, prauc, fpr, tpr, thresholds, predictions, auroc
        
    return fpr, tpr, thresholds, predictions, auroc
            
    
    


    


        
            
        
        

    
    