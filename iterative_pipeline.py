import argparse
import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from loguru import logger
from snorkel.analysis import LFAnalysis
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
)

from src.criteria_generator import CriteriaGenerator
from src.classifier import DialogueCriteriaClassifier
from src.snorkel_trainer import SnorkelTrainer

MODEL_NAME = "gpt-4.1-nano-2025-04-14"
PROMPT_FILE = "prompts/extract_topics_with_reasoning.txt"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_dataset_df(dataset_name: str, split: str) -> pd.DataFrame:
    ds = load_dataset(dataset_name, split=split)
    text_col = next((c for c in ["text", "sentence", "utterance"] if c in ds.column_names), ds.column_names[0])
    label_col = next((c for c in ["label", "labels", "intent"] if c in ds.column_names), ds.column_names[-1])
    df = ds.to_pandas()
    df = df.rename(columns={text_col: "text", label_col: "label"})
    return df


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    logger.info(f"Saved {len(df)} rows to {path}")


def read_criteria_file(path: Path) -> dict[str, str]:
    criteria = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            criteria[obj["criterion"]] = obj["description"]
    return criteria


def filter_lfs(dev_df: pd.DataFrame, trainer: SnorkelTrainer, threshold: float) -> list[str]:
    L = trainer.applier.apply(dev_df)
    scores = LFAnalysis(L, trainer.lfs).lf_summary(dev_df["label"].values)
    good_lfs = scores[scores["Emp. Acc."] > threshold].index.tolist()
    logger.info(f"Filtered to {len(good_lfs)} labeling functions")
    return good_lfs


def compute_metrics(y_true, y_pred, average="macro"):
    return {
        "f1": f1_score(y_true, y_pred, average=average),
        "precision": precision_score(y_true, y_pred, average=average),
        "recall": recall_score(y_true, y_pred, average=average),
        "ap": average_precision_score(y_true, y_pred, average=average),
    }


def run_iteration(args, iteration: int, error_texts: list[str] | None = None):
    iter_dir = Path(args.output_dir) / args.dataset / f"iter_{iteration}"
    ensure_dir(iter_dir / "weak_labels")
    ensure_dir(iter_dir / "models")
    ensure_dir(iter_dir / "metrics")

    train_df = load_dataset_df(args.dataset, "train")
    test_df = load_dataset_df(args.dataset, "test")
    dev_df = load_dataset_df(args.dataset, args.dev_split)

    generator = CriteriaGenerator("prompts/lf_generation.txt", "prompts/lf_deduplication.txt")

    if error_texts:
        texts = error_texts
        labels = ["unknown"] * len(error_texts)
    else:
        texts = dev_df["text"].tolist()
        labels = dev_df["label"].astype(str).tolist()

    criteria_path = iter_dir / "criteria.jsonl"
    new_criteria = generator.get_new_criteria(args.dataset, texts, labels)
    logger.info(f"Writing {len(new_criteria)} criteria to {criteria_path}")
    with open(criteria_path, "w", encoding="utf-8") as f:
        for item in new_criteria:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    criteria = read_criteria_file(criteria_path)
    trainer = SnorkelTrainer(criteria, sorted(dev_df["label"].unique().tolist()))

    dev_output = iter_dir / "classified_criteria.jsonl"
    clf = DialogueCriteriaClassifier(criteria, PROMPT_FILE, MODEL_NAME)
    results = []
    for text in dev_df["text"]:
        results.append(clf.classify_text(text))
    pd.DataFrame(results).to_json(dev_output, orient="records", lines=True)

    good_lfs = filter_lfs(dev_df, trainer, args.accuracy_threshold)
    filtered_criteria = {k: v for k, v in criteria.items() if k in good_lfs}
    with open(iter_dir / "filtered_lfs.json", "w", encoding="utf-8") as f:
        json.dump(filtered_criteria, f, ensure_ascii=False, indent=2)

    trainer = SnorkelTrainer(filtered_criteria, sorted(dev_df["label"].unique().tolist()))

    for split_name, df in {"train": train_df, "test": test_df, "dev": dev_df}.items():
        L = trainer.applier.apply(df)
        wl_path = iter_dir / "weak_labels" / f"{split_name}.jsonl"
        pd.DataFrame(L, columns=[lf.name for lf in trainer.lfs]).to_json(wl_path, orient="records", lines=True)
        logger.info(f"Saved weak labels for {split_name} to {wl_path}")

    trainer.fit(train_df)
    model_path = iter_dir / "models" / "label_model.pkl"
    trainer.label_model.save(model_path)
    logger.info(f"Saved label model to {model_path}")

    preds = trainer.predict(dev_df)
    metrics = compute_metrics(dev_df["label"], preds)
    metrics_path = iter_dir / "metrics" / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    wrong = dev_df[preds != dev_df["label"]]["text"].tolist()
    return wrong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--dev_split", default="validation")
    parser.add_argument("--max_iter", type=int, default=1)
    parser.add_argument("--accuracy_threshold", type=float, default=0.5)
    args = parser.parse_args()

    errors = None
    for i in range(args.max_iter):
        errors = run_iteration(args, i, errors)


if __name__ == "__main__":
    main()
