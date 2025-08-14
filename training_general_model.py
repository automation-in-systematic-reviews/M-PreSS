from sentence_transformers import losses
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from datasets import Dataset
from datetime import datetime
import csv
import os
import torch
from utils import read_criteria
import wandb
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["WANDB_PROJECT"] = "<WANDB_PROJECT_NAME>"

model_path = 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12' # BlueBERT
model = SentenceTransformer(model_path)
model.max_seq_length = 512
print("Max Sequence Length:", model.max_seq_length)
num_epochs = 6
train_batch_size = 6
learning_rate = 2e-5

#As distance metric, we use cosine distance (cosine_distance = 1-cosine_similarity)
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

#Negative pairs should have a distance of at least 0.5
margin = 0.5

dataset_path = './data/'
criteria_file = 'new_criteria.csv'

model_save_path = '<PATH_TO_TRAINED_MODEL>'
wandb_run_name = '<WANDB_RUN_NAME>'
print(model_save_path)
os.makedirs(model_save_path, exist_ok=True)


# selected_topics = ['Fragility Analysis of Statistically Significant Outcomes of Randomized Control Trials in Spine Surgery','Comparative efficacy of adjuvant non-opioid analgesia in adult cardiac surgical patients: A network meta-analysis','Diferential efcacy and_afety of_nti_SARS_CoV_2 antibody therapies for_he_anagement of_OVID_19: a_ystematic review and_etwork meta_analysis','Does the Source of Mesenchymal Stem Cell Have an Effect in the Management of Osteoarthritis of the Knee? Meta-Analysis of Randomized Controlled Trials','Specialized psychotherapies for adults with borderline personality disorder: A systematic review and meta-analysis','Long-term Outcomes of Cognitive Behavioral Therapy for Anxiety-Related Disorders', 'Comparative efficacy and safety of skeletal muscle relaxants for spasticity and musculoskeletal conditions: a systematic review', 'The use of acupuncture in patients with Raynaud_ syndrome: A systematic review and meta-analysis of randomized controlled trials'] # effectiveness
selected_topics = ['Diferential efcacy and_afety of_nti_SARS_CoV_2 antibody therapies for_he_anagement of_OVID_19: a_ystematic review and_etwork meta_analysis','Efficacy and safety of corticosteroid regimens for the treatment of hospitalized COVID-19 patients: a meta-analysis','Efficacy and safety of ivermectin for the treatment of COVID-19: a systematic review and meta-analysis','Efficacy and safety of selective serotonin reuptake inhibitors in COVID-19 management: a systematic review and meta-analysis','Efficacy of chloroquine and hydroxychloroquine for the treatment of hospitalized COVID-19 patients: a meta-analysis','Efficacy of lopinavir_itonavir combination therapy for the treatment of hospitalized COVID-19 patients: a meta-analysis','Prevalence of mental health symptoms in children and adolescents during the COVID-19 pandemic: A meta-analysis'] # covid
# selected_topics = ['Efficacy and safety of ivermectin for the treatment of COVID-19: a systematic review and meta-analysis','Efficacy and safety of selective serotonin reuptake inhibitors in COVID-19 management: a systematic review and meta-analysis','Efficacy of lopinavir_itonavir combination therapy for the treatment of hospitalized COVID-19 patients: a meta-analysis'] # covid 3
print(selected_topics)

######### Read train data  ##########
# Read train data
criteria = read_criteria(dataset_path+criteria_file)
topics = list(criteria.keys())
topics2idx = {k:v for k,v in zip(topics, range(0,len(topics)))}
# print(topics2idx)
# train_samples = []
with open(os.path.join(dataset_path, "train.csv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn)
    query, study, label = [], [], []
    for row in reader:
        if row['query'] not in selected_topics:
            continue
        query.append('Query: '+row['query']+'. Criteria: ' + criteria[row['query']])
        # query.append('Query: ' + row['query']) # no criteria
        study.append('Title: ' + row['title'] + '. Abstract: '+ row['abstract'])
        label.append(int(row['label_included']))

train_samples = {'query':query, 'study':study, 'label':label}
dataset = Dataset.from_dict(train_samples)
print('training sample: {}'.format(len(dataset)))

with open(os.path.join(dataset_path, "test.csv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn)
    query, study, label = [], [], []
    for row in reader:
        if row['query'] not in selected_topics:
            continue
        query.append('Query: '+row['query']+'. Criteria: ' + criteria[row['query']])
        # query.append('Query: ' + row['query']) # no criteria
        study.append('Title: ' + row['title'] + '. Abstract: '+ row['abstract'])
        label.append(int(row['label_included']))

val_samples = {'query':query, 'study':study, 'label':label}
val_dataset = Dataset.from_dict(val_samples)
print('validation sample: {}'.format(len(val_dataset)))
print('training sample: {}'.format(len(dataset)))


train_loss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)


training_args = SentenceTransformerTrainingArguments(
    output_dir=model_save_path,
    num_train_epochs=num_epochs,
    seed=seed,
    per_device_train_batch_size=train_batch_size,
    learning_rate=learning_rate,
    warmup_ratio=0.1,
    fp16=True,
    save_strategy = "no",
    save_total_limit=1,
    metric_for_best_model="loss",
    greater_is_better=False,
    report_to='wandb',
    run_name=wandb_run_name,
    logging_steps=1,
    evaluation_strategy="epoch",
    disable_tqdm=True,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=val_dataset,
    eval_dataset=val_dataset,
    loss=train_loss,
)
trainer.train()
model.save(model_save_path)
# model.push_to_hub('')
