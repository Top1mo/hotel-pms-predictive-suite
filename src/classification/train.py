"""Train and evaluate the logistic-regression cancellation classifier."""
import json

from src.classification.model import train_and_evaluate


def main():
    report = train_and_evaluate()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
