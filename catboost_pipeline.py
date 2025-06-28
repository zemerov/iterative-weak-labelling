import argparse
import json
from pathlib import Path

from catboost import CatBoostClassifier, Pool

from iterative_pipeline import load_dataset_df, compute_metrics

# AICODE-NOTE Reusing dataset loading and metric functions from the iterative
# pipeline to keep the codebase consistent


def train_catboost(dataset: str, dev_split: str) -> tuple[CatBoostClassifier, dict, dict]:
    """Train CatBoost model on ``dev_split`` and evaluate on dev and test sets."""
    train_df = load_dataset_df(dataset, dev_split)
    dev_df = load_dataset_df(dataset, dev_split)
    test_df = load_dataset_df(dataset, "test")

    train_pool = Pool(train_df["text"], label=train_df["label"], text_features=[0])
    dev_pool = Pool(dev_df["text"], label=dev_df["label"], text_features=[0])
    test_pool = Pool(test_df["text"], label=test_df["label"], text_features=[0])

    model = CatBoostClassifier(loss_function="MultiClass", iterations=500, verbose=True)
    model.fit(train_pool, eval_set=dev_pool, verbose=True)

    dev_preds = model.predict(dev_pool)
    test_preds = model.predict(test_pool)

    dev_metrics = compute_metrics(dev_df["label"], dev_preds)
    test_metrics = compute_metrics(test_df["label"], test_preds)

    return model, dev_metrics, test_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dev_split", default="validation")
    parser.add_argument("--output_dir", default="catboost")
    args = parser.parse_args()

    model, dev_metrics, test_metrics = train_catboost(args.dataset, args.dev_split)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dev_metrics.json").write_text(json.dumps(dev_metrics, indent=2))
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    model.save_model(out_dir / "model.cbm")


if __name__ == "__main__":
    main()
