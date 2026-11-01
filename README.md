# CSE382: City Hotel (H2) Booking Demand ML System

Guide for the team to get the repo running and start contributing.

## Team Members

Fill in your details here so contributions can be tracked:

| Name | Student ID | GitHub | 
| :--- | :--- | :--- | :--- |
| Mostafa |24p0353 | @Top1mo | 
|  | | |
|  | | |  
| | | | 
|  | | |  

## Quick Setup

```bash
git clone https://github.com/Top1mo/hotel-pms-predictive-suite.git
cd hotel-pms-predictive-suite
git checkout dev

python -m venv .venv

# Activate (pick ONE for your shell)
source .venv/bin/activate        # Bash/Zsh (Linux/macOS)
source .venv/bin/activate.fish   # Fish
.venv\Scripts\activate           # Windows (cmd/PowerShell)

pip install -r requirements.txt
python src/data/download.py
```

If the last step prints `Verified H2.csv: ... rows, ... columns.`, the raw dataset is ready.

To generate the cleaned dataset (`data/cleaned/H2_cleaned.csv`), open and run `data/cleaned/Data_cleaning.ipynb`.

## About the data

The dataset is **not committed** (it's in `.gitignore`). `src/data/download.py`
downloads `H2.csv` into the path set in `config.yaml` and checks the shape
against `expected_rows` / `expected_cols`. If the check fails, the file is
corrupted or wrong: delete it and run the script again.

**Never commit CSV files.** Everyone downloads and generates their own copy.

## Config

All paths and data settings live in `config.yaml`. Don't hardcode paths in
your code, read them from the config.

## Project structure

```text
hotel-pms-predictive-suite/
├── README.md
├── Report.md
├── requirements.txt
├── config.yaml
├── main.py
├── app.py
├── data/
│   ├── raw/
│   └── cleaned/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── features/
│   ├── regression/
│   ├── classification/
│   ├── clustering/
│   ├── tuning/
│   ├── evaluation/
│   └── inference/
├── artifacts/
└── outputs/
```

## Workflow

1. Pull the latest `dev`: `git checkout dev && git pull`
2. Make your own branch: `git checkout -b feature/<short-name>`
3. Commit small and often, with clear messages
4. Push and open a PR into `dev`: `git push -u origin feature/<short-name>`
5. Get one teammate to review before merging
6. Don't push directly to `dev`

## Adding dependencies

If you install a new package, add it to `requirements.txt` in the same PR
so everyone's environment stays in sync.

## Troubleshooting

- **`ModuleNotFoundError`**: venv isn't activated, or you forgot `pip install -r requirements.txt`
- **Shape mismatch on download**: delete the CSV and rerun `download.py`
- **Download fails**: check your internet and that the `url` in `config.yaml` is still valid

## Submission info

- **Repository URL:** `https://github.com/Top1mo/hotel-pms-predictive-suite/tree/dev`
- **Submitted Commit SHA:** `<INSERT_SHA_BEFORE_ZIP>`
