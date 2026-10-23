import os
import urllib.request
import pandas as pd
import yaml

with open('config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)['data']

os.makedirs(os.path.dirname(cfg['raw_path']), exist_ok=True)
if not os.path.exists(cfg['raw_path']):
    print(f"Downloading H2.csv from {cfg['url']}...")
    urllib.request.urlretrieve(cfg['url'], cfg['raw_path'])

df = pd.read_csv(cfg['raw_path'])
assert df.shape == (cfg['expected_rows'], cfg['expected_cols']), f'Shape mismatch: {df.shape}'
print(f'Verified H2.csv: {df.shape[0]} rows, {df.shape[1]} columns.')
