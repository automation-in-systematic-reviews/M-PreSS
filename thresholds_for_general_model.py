from model_functions import find_best_threshold, threshold_to_binary_labels
import pandas as pd
import os
import glob
from utils import evaluate_classification


train_path = '/user/work/zx16649/study_screening_predictions/general_paper/os_train/'
test_path = '/user/work/zx16649/study_screening_predictions/general_paper/os_test/'
suffix = '*_inclusion_exclusion.csv'
test_suffix = '*_inclusion_exclusion.csv'
summary_path = '/user/work/zx16649/study_screening_predictions/general_paper/thresholds/summary_covid_3_25_03_24.csv'
prediction_path = '/user/work/zx16649/study_screening_predictions/general_paper/thresholds/'
model_path = '/user/work/zx16649/model_trained/os_covid_criteria/os_covid_3_include_exclude_criteria_25_01_16'
os.makedirs(prediction_path, exist_ok=True)

search_pattern = os.path.join(train_path, suffix)
train_list = glob.glob(search_pattern)
search_pattern = os.path.join(test_path, test_suffix)
test_list = glob.glob(search_pattern)

all_similarity, all_label = [], []
all_similarity_test, all_label_test = [],[]
data_to_summary = []
for file in train_list:
    parts = file.split('/')[-1]
    incidence = parts.split('_')[0]
    print(incidence)

    df = pd.read_csv(file)
    for test in test_list:
        if incidence in test:
            test_file = test
    test_df = pd.read_csv(test_file)
    
    similarity = df[model_path].to_list()
    # similarity = df['similarity'].to_list()
    label = df['ground_truth'].to_list()
    all_similarity.extend(similarity)
    all_label.extend(label)
    test_similarity = test_df[model_path].to_list()
    test_label = test_df['ground_truth'].to_list()
    all_similarity_test.extend(test_similarity)
    all_label_test.extend(test_label)
    
    threshold_acc, _ = find_best_threshold(similarity, label, strategy='best_acc',similarity=True)
    prediction_acc = threshold_to_binary_labels(test_similarity, threshold=threshold_acc, similarity=True)
    
    tn, fp, fn, tp, accuracy, precision, recall, recall_pos, recall_neg, f1, specificity = evaluate_classification(test_label, prediction_acc)
    print(f'incidence: {incidence}')
    print(f'threshold for best accuracy: {threshold_acc}')
    print(f'true negative: {tn}, false positive:{fp}, false negative:{fn}, true positive:{tp}')
    print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, recall_pos: {recall_pos}, recall_neg: {recall_neg}, f1: {f1}')
    name = f"general_model(threshold_acc={threshold_acc})"
    test_df[name]=prediction_acc
    data_to_summary.append([incidence, 'best_acc', threshold_acc, tn, fp, tp, fn, accuracy, specificity, recall, recall_pos, recall_neg,
                            precision, f1])

    # threshold_p, _ = find_best_threshold(similarity, label, strategy='best_precision', similarity=True)
    # prediction_p = threshold_to_binary_labels(test_similarity, threshold=threshold_p, similarity=True)
    # positive_p = prediction_p.count(1)

    # tn, fp, fn, tp, accuracy, precision, recall, recall_pos, recall_neg, f1, specificity = evaluate_classification(test_label, prediction_p)
    # print(f'incidence: {incidence}')
    # print(f'threshold for best precision: {threshold_p}')
    # print(f'true negative: {tn}, false positive:{fp}, false negative:{fn}, true positive:{tp}')
    # print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, recall_pos: {recall_pos}, recall_neg: {recall_neg}, f1: {f1}')
    # name = f"general_model(threshold_precision={threshold_p})"
    # test_df[name] = prediction_p
    # data_to_summary.append([incidence, 'best_precision', threshold_p, tn, fp, tp, fn, accuracy, specificity, recall,
    #                         precision, f1])
    
    threshold_r, _ = find_best_threshold(similarity, label, strategy='best_recall',similarity=True)
    prediction_r = threshold_to_binary_labels(test_similarity, threshold=threshold_r, similarity=True)
    positive_r = prediction_r.count(1)

    tn, fp, fn, tp, accuracy, precision, recall, recall_pos, recall_neg, f1, specificity = evaluate_classification(test_label, prediction_r)
    print(f'incidence: {incidence}')
    print(f'threshold for best recall (include all positive samples): {threshold_r}')
    print(f'true negative: {tn}, false positive:{fp}, false negative:{fn}, true positive:{tp}')
    print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, recall_pos: {recall_pos}, recall_neg: {recall_neg}, f1: {f1}')
    name = f"general_model(threshold_recall={threshold_r})"
    test_df[name] = prediction_r
    data_to_summary.append([incidence, 'best_recall', threshold_r, tn, fp, tp, fn, accuracy, specificity, recall, recall_pos, recall_neg,
                            precision, f1])

    threshold_f1, _ = find_best_threshold(similarity, label, strategy='best_f1',similarity=True)
    prediction_f1 = threshold_to_binary_labels(test_similarity, threshold=threshold_r, similarity=True)
    positive_f1 = prediction_f1.count(1)

    tn, fp, fn, tp, accuracy, precision, recall, recall_pos, recall_neg, f1, specificity = evaluate_classification(test_label, prediction_f1)
    print(f'incidence: {incidence}')
    print(f'threshold for best f1 (include all positive samples): {threshold_f1}')
    print(f'true negative: {tn}, false positive:{fp}, false negative:{fn}, true positive:{tp}')
    print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, recall_pos: {recall_pos}, recall_neg: {recall_neg}, f1: {f1}')
    name = f"general_model(threshold_f1={threshold_f1})"
    test_df[name] = prediction_f1
    data_to_summary.append([incidence, 'best_f1', threshold_f1, tn, fp, tp, fn, accuracy, specificity, recall, recall_pos, recall_neg,
                            precision, f1])

    # threshold_npv, _ = find_best_threshold(similarity, label, strategy='best_npv',similarity=True)
    # prediction_npv = threshold_to_binary_labels(test_similarity, threshold=threshold_npv, similarity=True)
    # positive_npv = prediction_npv.count(1)

    # tn, fp, fn, tp, accuracy, precision, recall, recall_pos, recall_neg, f1, specificity = evaluate_classification(test_label, prediction_npv)
    # print(f'incidence: {incidence}')
    # print(f'threshold for best recall (include all positive samples): {threshold_npv}')
    # print(f'true negative: {tn}, false positive:{fp}, false negative:{fn}, true positive:{tp}')
    # print(f'accuracy: {accuracy}, precision: {precision}, recall: {recall}, recall_pos: {recall_pos}, recall_neg: {recall_neg}, f1: {f1}')
    # name = f"general_model(threshold_npv={threshold_npv})"
    # test_df[name] = prediction_npv
    # data_to_summary.append([incidence, 'best_npv', threshold_npv, tn, fp, tp, fn, accuracy, specificity, recall,
    #                         precision, f1])

    print('=======================================================')
    

    # difference = positive_npv - positive_p
    # filename = f"{incidence}_incidence_{difference}_to_review.csv"
    filename = f"{incidence}.csv"
    test_df.to_csv(prediction_path+filename, index=False)

    
incidence = 'all'
threshold_acc_all, _ = find_best_threshold(all_similarity, all_label, strategy='best_acc',similarity=True)
prediction_acc_all = threshold_to_binary_labels(all_similarity_test, threshold=threshold_acc_all, similarity=True)
tn, fp, fn, tp, accuracy, precision, recall, recall_pos, recall_neg, f1, specificity = evaluate_classification(all_label_test, prediction_acc_all)
data_to_summary.append([incidence, 'best_acc', threshold_acc_all, tn, fp, tp, fn, accuracy, specificity, recall, precision, f1])
# threshold_p_all, _ = find_best_threshold(all_similarity, all_label, strategy='best_precision',similarity=True)
# prediction_p_all = threshold_to_binary_labels(all_similarity_test, threshold=threshold_p_all, similarity=True)
# tn, fp, fn, tp, accuracy, precision, recall, f1, specificity = evaluate_classification(all_label_test, prediction_p_all)
# data_to_summary.append([incidence, 'best_precision', threshold_p_all, tn, fp, tp, fn, accuracy, specificity, recall, precision, f1])
threshold_r_all, _ = find_best_threshold(all_similarity, all_label, strategy='best_recall',similarity=True)
prediction_r_all = threshold_to_binary_labels(all_similarity_test, threshold=threshold_r_all, similarity=True)
tn, fp, fn, tp, accuracy, precision, recall, recall_pos, recall_neg, f1, specificity = evaluate_classification(all_label_test, prediction_r_all)
data_to_summary.append([incidence, 'best_recall', threshold_r_all, tn, fp, tp, fn, accuracy, specificity, recall, precision, f1])
threshold_f1_all, _ = find_best_threshold(all_similarity, all_label, strategy='best_f1',similarity=True)
prediction_f1_all = threshold_to_binary_labels(all_similarity_test, threshold=threshold_f1_all, similarity=True)
tn, fp, fn, tp, accuracy, precision, recall, recall_pos, recall_neg, f1, specificity = evaluate_classification(all_label_test, prediction_f1_all)
data_to_summary.append([incidence, 'best_f1', threshold_f1_all, tn, fp, tp, fn, accuracy, specificity, recall, precision, f1])
summary = pd.DataFrame(data_to_summary, columns=['topic','strategy', 'threshold', 'true_negative','false_positive',
                                                 'true_positive','false_negative','accuracy', 'specificity',
                                                 'sensitivity/recall', 'sensitivity_positive', 'sensitivity_negative', 'precision', 'f1'])
print(summary.head())
summary.to_csv(summary_path,index=False)


