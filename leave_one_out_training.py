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
taskid = os.environ["SLURM_ARRAY_TASK_ID"]

model_path = 'bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12' # BlueBERT
model = SentenceTransformer(model_path)

# model = SentenceTransformer('stsb-bert-base', device='cuda:3')
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
criteria_file = 'criteria.csv'
model_save_path = '<PATH_TO_SAVE_TRAINED_MODEL>'


######### Read train data  ##########
# Read train data
criteria = read_criteria(dataset_path+criteria_file)
topics = list(criteria.keys())
print(len(topics))
skip_topics = topics[int(taskid)]
print(f'skip topic: {skip_topics}')
wandb_run_name = f'loo_{skip_topics}'

model_save_path = model_save_path + skip_topics.replace(" ", "_") + '_' + datetime.now().strftime("%Y%m%d") + '/'
print(model_save_path)
os.makedirs(model_save_path, exist_ok=True)

with open(os.path.join(dataset_path, "train.csv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn)
    query, study, label = [], [], []
    for row in reader:
        if row['query'] in skip_topics: # leave target topic out
            continue
        query.append('Query: '+row['query']+'. Criteria: ' + criteria[row['query']])
        study.append('Title: ' + row['title'] + '. Abstract: '+ row['abstract'])
        label.append(int(row['label_included']))

train_samples = {'query':query, 'study':study, 'label':label}
dataset = Dataset.from_dict(train_samples)
print('training sample: {}'.format(len(dataset)))

with open(os.path.join(dataset_path, "test.csv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn)
    for row in reader:
        # try:
        if row['query'] in skip_topics:
            continue
        query.append('Query: '+row['query']+'. Criteria: ' + criteria[row['query']])
        study.append('Title: ' + row['title'] + '. Abstract: '+ row['abstract'])
        label.append(int(row['label_included']))

# append test set from non-target topics to train the general model
val_samples = {'query':query, 'study':study, 'label':label}
val_dataset = Dataset.from_dict(val_samples)
dataset = val_dataset
print('training sample: {}'.format(len(dataset)))

train_loss = losses.OnlineContrastiveLoss(model=model, distance_metric=distance_metric, margin=margin)

# Train the model
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
    disable_tqdm=True,
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    loss=train_loss,
)
trainer.train()
model.save(model_save_path)
# model.push_to_hub('')
