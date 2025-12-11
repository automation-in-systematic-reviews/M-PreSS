from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
from utils import read_data_os

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

data_path = '/user/work/zx16649/dataset/open_source/'
train_file = 'train.csv'
test_file = 'test.csv'
train_query, train_corpus, train_label = read_data_os(data_path+train_file, label_key='label_included')
test_query, test_corpus, train_label = read_data_os(data_path+train_file, label_key='label_included')

texts = train_corpus + test_corpus

lengths = []
for t in tqdm(texts):
    enc = tokenizer(
        t,
        truncation=False,
        padding=False,
        add_special_tokens=True,  # to include [CLS]/[SEP] in length
    )
    lengths.append(len(enc["input_ids"]))

lengths = np.array(lengths)
max_len = 512

num_over = (lengths > max_len).sum()
frac_over = num_over / len(lengths)

print(f"Sequences >512: {num_over} ({frac_over:.2%} of data)")

truncation_amounts = np.clip(lengths - max_len, a_min=0, a_max=None)
total_tokens_lost = truncation_amounts.sum()
avg_tokens_lost_per_truncated_example = (
    truncation_amounts[truncation_amounts > 0].mean()
    if num_over > 0 else 0
)

print(f"Total tokens lost: {total_tokens_lost}")
print(f"Avg tokens lost per truncated example: {avg_tokens_lost_per_truncated_example:.1f}")

for p in [50, 75, 90, 95, 99, 100]:
    print(f"{p}th percentile length: {np.percentile(lengths, p)}")

