"""Run from any working directory; download the raw file without changing it."""
from pathlib import Path
import urllib.request
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[2]

def main():
    cfg=yaml.safe_load((ROOT/'config.yaml').read_text(encoding='utf-8'))['data']
    path=ROOT/cfg['raw_path']
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(cfg['url'],path)
    df=pd.read_csv(path)
    if df.shape != (cfg['expected_rows'],cfg['expected_cols']):
        raise ValueError(f'Shape mismatch: {df.shape}')
    print(f'Raw H2 available: {df.shape}. Run python main.py prepare for schema validation.')
if __name__=='__main__':
    main()
