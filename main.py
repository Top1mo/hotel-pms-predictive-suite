"""Milestone 1 preparation commands; no model training or test evaluation."""
import argparse
import json
from src.data.prepare import prepare, check_preprocessing, ROOT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','check-preprocessing','analyze'])
    args = parser.parse_args()
    raw,parts,report = prepare()
    if args.command == 'check-preprocessing':
        print(json.dumps(check_preprocessing(parts), indent=2))
    elif args.command == 'analyze':
        from src.evaluation.eda import run_eda
        train = parts['train']
        run_eda(raw.loc[train.index], train, report, ROOT/'outputs'/'eda_training')
        print('Training-only EDA saved under outputs/eda_training.')
    else:
        print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
