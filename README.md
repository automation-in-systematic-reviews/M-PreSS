# M-PreSS
code base for paper "M-PreSS: Model Pre-training Approach for Study Screening in Systematic Reviews"

## Installation

```bash
conda create -n mpress python=3.8.16
conda activate mpress
pip install -r requirements.txt
```

## Data format

The scripts expect CSV files under a data directory, usually `./data/`.

Typical files are:

```text
data/
├── train.csv
├── test.csv
├── val.csv
└── criteria.csv
```

The main training and evaluation CSV files should contain at least the following columns:

| Column | Description |
|---|---|
| `query` | Systematic review topic/question |
| `title` | Candidate study title |
| `abstract` | Candidate study abstract |
| `label_included` | Binary label: `1` for included/relevant, `0` for excluded/not relevant |

The criteria file should contain:

| Column | Description |
|---|---|
| `topic` | Topic name matching the `query` values |
| `criteria` | Eligibility criteria text for that topic |

Each study is converted into text like:

```text
Title: <title>. Abstract: <abstract>
```

Each query is usually converted into text like:

```text
Query: <topic>. Criteria: <criteria>
```

## Citation

If you find this repository useful, please cite:
```
@article {Xu2025.04.08.25325463,
	author = {Xu, Zhaozhen and Davies, Philippa and Millard, Louise AC and Teng, Lam and Markozannes, Georgios and Erola, Pau and Seleiro, Eduardo AP and Higgins, Julian PT and Martin, Richard M and Sobczyk-Barad, Maria and Tsilidis, Konstantinos K and Chan, Doris SM and Gaunt, Tom R and Liu, Yi},
	title = {M-PreSS: A Model Pre-training Approach for Study Screening in Systematic Reviews},
	elocation-id = {2025.04.08.25325463},
	year = {2025},
	doi = {10.1101/2025.04.08.25325463},
	URL = {https://www.medrxiv.org/content/early/2025/06/18/2025.04.08.25325463},
	eprint = {https://www.medrxiv.org/content/early/2025/06/18/2025.04.08.25325463.full.pdf},
	journal = {medRxiv}
}
```
