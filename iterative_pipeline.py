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
from generate_criteria import read_criteria
from classify_criteria import run_parallel_requests

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


def filter_lfs(pred_df: pd.DataFrame, trainer: SnorkelTrainer, threshold: float) -> list[str]:
    L = trainer.applier.apply(pred_df)
    scores = LFAnalysis(L, trainer.lfs).lf_summary(pred_df["label"].values)
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


def load_classified(path: Path) -> pd.DataFrame:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            row = {"index": obj["index"], "text": obj["text"]} | obj["labels"]
            data.append(row)
    return pd.DataFrame(data)


def classify_texts(texts: list[str], criteria: dict[str, str], output: Path, workers: int) -> pd.DataFrame:
    """Classify texts with DialogueCriteriaClassifier in parallel."""
    if output.exists():
        logger.info(f"Using existing classification from {output}")
    classifier = DialogueCriteriaClassifier(criteria, PROMPT_FILE, MODEL_NAME)
    start = 0
    if output.exists():
        with open(output, "r", encoding="utf-8") as f:
            start = sum(1 for _ in f)
        if start >= len(texts):
            return load_classified(output)
        texts = texts[start:]
    run_parallel_requests(texts, classifier, str(output), workers, start_idx=start)
    return load_classified(output)


def run_iteration(args, iteration: int, error_texts: list[str] | None = None):
    iter_dir = Path(args.output_dir) / args.dataset / f"iter_{iteration}"
    ensure_dir(iter_dir / "weak_labels")
    ensure_dir(iter_dir / "models")
    ensure_dir(iter_dir / "metrics")
    ensure_dir(iter_dir / "classified")

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
    prev_path = (Path(args.output_dir) / args.dataset / f"iter_{iteration-1}" / "criteria.jsonl") if iteration > 0 else None
    existing = read_criteria(prev_path) if prev_path and prev_path.exists() else []
    new_criteria = generator.get_new_criteria(
        args.dataset,
        texts,
        labels,
        existing_criteria={c["criterion"]: c["description"] for c in existing} if existing else None,
    )
    if existing:
        final_criteria = generator.deduplicate_new_criteria(existing, new_criteria)
    else:
        final_criteria = new_criteria
    logger.info(f"Writing {len(final_criteria)} criteria to {criteria_path}")
    with open(criteria_path, "w", encoding="utf-8") as f:
        for item in final_criteria:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    criteria = read_criteria_file(criteria_path)
    classes = sorted(dev_df["label"].unique().tolist())
    trainer_full = SnorkelTrainer(criteria, classes)

    dev_output = iter_dir / "classified" / "dev.jsonl"
    dev_pred_df = classify_texts(dev_df["text"].tolist(), criteria, dev_output, args.num_workers)
    dev_pred_df["label"] = dev_df["label"].map(lambda x: trainer_full.class_to_index[x])

    good_lfs = filter_lfs(dev_pred_df, trainer_full, args.accuracy_threshold)
    filtered_criteria = {k: v for k, v in criteria.items() if k in good_lfs}
    with open(iter_dir / "filtered_lfs.json", "w", encoding="utf-8") as f:
        json.dump(filtered_criteria, f, ensure_ascii=False, indent=2)

    trainer = SnorkelTrainer(filtered_criteria, classes)

    # classify train and test with filtered criteria
    train_output = iter_dir / "classified" / "train.jsonl"
    test_output = iter_dir / "classified" / "test.jsonl"
    train_pred_df = classify_texts(train_df["text"].tolist(), filtered_criteria, train_output, args.num_workers)
    test_pred_df = classify_texts(test_df["text"].tolist(), filtered_criteria, test_output, args.num_workers)

    # add labels as indices
    train_pred_df["label"] = train_df["label"].map(lambda x: trainer.class_to_index[x])
    test_pred_df["label"] = test_df["label"].map(lambda x: trainer.class_to_index[x])

    for name, df in {"train": train_pred_df, "test": test_pred_df, "dev": dev_pred_df}.items():
        L = trainer.applier.apply(df)
        wl_path = iter_dir / "weak_labels" / f"{name}.jsonl"
        pd.DataFrame(L, columns=[lf.name for lf in trainer.lfs]).to_json(wl_path, orient="records", lines=True)
        logger.info(f"Saved weak labels for {name} to {wl_path}")

    trainer.fit(train_pred_df)
    model_path = iter_dir / "models" / "label_model.pkl"
    trainer.label_model.save(model_path)
    logger.info(f"Saved label model to {model_path}")

    preds = trainer.predict(dev_pred_df)
    metrics = compute_metrics(dev_pred_df["label"], preds)
    metrics_path = iter_dir / "metrics" / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    wrong = dev_df[preds != dev_pred_df["label"]]["text"].tolist()
    return wrong


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", default="data")
    parser.add_argument("--dev_split", default="validation")
    parser.add_argument("--max_iter", type=int, default=1)
    parser.add_argument("--accuracy_threshold", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    errors = None
    for i in range(args.max_iter):
        errors = run_iteration(args, i, errors)


if __name__ == "__main__":
    main()
