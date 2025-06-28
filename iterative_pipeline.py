import argparse
import json
from pathlib import Path
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm.auto import tqdm
from loguru import logger
from snorkel.labeling import LFAnalysis
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



def load_dataset_df(dataset_name: str, split: str, base_dir: str = "data") -> pd.DataFrame:
    """
    Load a single split (train/valid/test) for a local dataset stored as JSON.

    Expects files at:
        data/<dataset_name>/source/<split>.json

    JSON schema:
        {
          "<id>": {
            "label": int,
            "data": {"text": str},
            "weak_labels": [int, ...]
          },
          ...
        }

    Returns:
        A DataFrame with columns ['text', 'label', 'weak_labels'].
    """
    # Build path to JSON file
    json_path = os.path.join(base_dir, dataset_name, "source", f"{split}.json")

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Split file not found: {json_path}")

    # Load JSON content
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)

    # Parse into list of records
    records = []
    for idx, entry in raw.items():
        text = entry.get("data", {}).get("text")
        label = entry.get("label")
        weak_labels = entry.get("weak_labels", None)
        records.append({"text": text, "label": label, "weak_labels": weak_labels})

    # Create DataFrame
    df = pd.DataFrame(records)
    return df


def save_jsonl(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    logger.info(f"Saved {len(df)} rows to {path}")


def read_criteria_file(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load criterion descriptions and their associated classes from ``path``."""

    descriptions: dict[str, str] = {}
    classes: dict[str, str] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            criterion_name = obj["criterion"]
            descriptions[criterion_name] = obj["description"]
            classes[criterion_name] = obj["class"]

    return descriptions, classes


def filter_lfs(
    pred_df: pd.DataFrame,
    trainer: SnorkelTrainer,
    threshold: float,
    metrics_dir: Path,
) -> list[str]:
    """Filter labeling functions by empirical accuracy and save score table."""
    L = trainer.applier.apply(pred_df)
    scores = LFAnalysis(L, trainer.lfs).lf_summary(pred_df["label"].values)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    scores_path = metrics_dir / "lf_scores.csv"
    scores.to_csv(scores_path)
    logger.info(f"Saved LF scores to {scores_path}")
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


def run_parallel_generation(
    generator: CriteriaGenerator,
    dataset: str,
    label_groups: dict[str, list[str]],
    existing: dict[str, str] | None,
    num_workers: int,
) -> list[dict[str, str]]:
    """Generate criteria for label groups concurrently."""
    logger.info(
        f"Generating criteria for {len(label_groups)} groups with {num_workers} workers..."
    )
    results: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                generator.get_new_criteria,
                dataset,
                texts,
                [label] * len(texts),
                existing_criteria=existing,
            ): label
            for label, texts in label_groups.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            label = futures[future]
            try:
                res = future.result()
            except Exception as e:  # pragma: no cover - logging only
                logger.error(
                    f"Error generating criteria for label {label}: {repr(e)}"
                )
                res = []
            results.extend(res)
    return results


def run_iteration(args, iteration: int, error_texts: list[dict[str, str]] | None = None):
    iter_dir = Path(args.output_dir) / args.dataset / f"iter_{iteration}"
    # Prepare directory structure for this iteration
    ensure_dir(iter_dir / "weak_labels")
    ensure_dir(iter_dir / "models")
    ensure_dir(iter_dir / "metrics")
    ensure_dir(iter_dir / "classified")

    # Load dataset splits
    train_df = load_dataset_df(args.dataset, "train")
    test_df = load_dataset_df(args.dataset, "test")
    dev_df = load_dataset_df(args.dataset, args.dev_split)

    # Initialize criteria generator
    generator = CriteriaGenerator(
        "prompts/lf_generation.txt",
        "prompts/lf_deduplication.txt",
    )

    if error_texts:
        # AICODE-NOTE init labels with real labels, change the error_text param to get not only texts but labels too
        texts = [sample["text"] for sample in error_texts]
        labels = [str(sample["label"]) for sample in error_texts]
    else:
        texts = dev_df["text"].tolist()
        labels = dev_df["label"].astype(str).tolist()

    criteria_path = iter_dir / "criteria.jsonl"
    prev_path = (
        Path(args.output_dir)
        / args.dataset
        / f"iter_{iteration-1}"
        / "criteria.jsonl"
    ) if iteration > 0 else None
    existing = read_criteria(prev_path) if prev_path and prev_path.exists() else []

    if criteria_path.exists():
        # Criteria already calculated in a previous run
        logger.info(f"Loading existing criteria from {criteria_path}")
        criteria_descriptions, criteria_classes = read_criteria_file(criteria_path)
    else:
        # AICODE-TODO create criteria in a loop. Group texts by label. Send texts with single label
        label_groups: dict[str, list[str]] = {}
        for t, l in zip(texts, labels):
            label_groups.setdefault(l, []).append(t)

        existing_dict = (
            {c["criterion"]: c["description"] for c in existing} if existing else None
        )
        new_criteria = run_parallel_generation(
            generator,
            args.dataset,
            label_groups,
            existing_dict,
            args.num_workers,
        )

        logger.info(f"Generated {len(new_criteria)} new criteria!")
        # AICODE-NOTE deduplicate all criteria after generation in the loop. Concatenate all generated criteria before deduplication
        if existing:
            final_criteria = generator.deduplicate_new_criteria(existing, new_criteria)
        else:
            final_criteria = generator.deduplicate_new_criteria([], new_criteria)
        logger.info(f"Writing {len(final_criteria)} criteria to {criteria_path}")
        with open(criteria_path, "w", encoding="utf-8") as f:
            for item in final_criteria:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        criteria_descriptions, criteria_classes = read_criteria_file(criteria_path)
    classes = sorted(dev_df["label"].unique().tolist())
    trainer_full = SnorkelTrainer(criteria_descriptions, criteria_classes, classes)

    dev_output = iter_dir / "classified" / "dev.jsonl"
    # Use cached classification if available
    if dev_output.exists():
        logger.info(f"Loading existing classification from {dev_output}")
        dev_pred_df = load_classified(dev_output)
    else:
        dev_pred_df = classify_texts(
            dev_df["text"].tolist(),
            criteria_descriptions,
            dev_output,
            args.num_workers,
        )
    dev_pred_df["label"] = dev_df["label"].map(lambda x: trainer_full.class_to_index[x])

    # Evaluate labeling functions on dev set and filter by accuracy
    good_lfs = filter_lfs(
        dev_pred_df,
        trainer_full,
        args.accuracy_threshold,
        iter_dir / "metrics",
    )
    filtered_criteria_descriptions = {
        k: v for k, v in criteria_descriptions.items() if k in good_lfs
    }
    filtered_criteria_classes = {
        k: v for k, v in criteria_classes.items() if k in good_lfs
    }
    with open(iter_dir / "filtered_lfs.json", "w", encoding="utf-8") as f:
        json.dump(filtered_criteria_descriptions, f, ensure_ascii=False, indent=2)

    # Initialize trainer with filtered labeling functions
    trainer = SnorkelTrainer(
        filtered_criteria_descriptions,
        filtered_criteria_classes,
        classes,
    )

    # classify train and test with filtered criteria
    train_output = iter_dir / "classified" / "train.jsonl"
    test_output = iter_dir / "classified" / "test.jsonl"
    if train_output.exists():
        logger.info(f"Loading existing classification from {train_output}")
        train_pred_df = load_classified(train_output)
    else:
        train_pred_df = classify_texts(
            train_df["text"].tolist(),
            filtered_criteria_descriptions,
            train_output,
            args.num_workers,
        )
    if test_output.exists():
        logger.info(f"Loading existing classification from {test_output}")
        test_pred_df = load_classified(test_output)
    else:
        test_pred_df = classify_texts(
            test_df["text"].tolist(),
            filtered_criteria_descriptions,
            test_output,
            args.num_workers,
        )

    # Attach numeric label indices used by Snorkel
    train_pred_df["label"] = train_df["label"].map(lambda x: trainer.class_to_index[x])
    test_pred_df["label"] = test_df["label"].map(lambda x: trainer.class_to_index[x])

    # Generate weak label matrices for all splits
    for name, df in {"train": train_pred_df, "test": test_pred_df, "dev": dev_pred_df}.items():
        L = trainer.applier.apply(df)
        wl_path = iter_dir / "weak_labels" / f"{name}.jsonl"
        pd.DataFrame(L, columns=[lf.name for lf in trainer.lfs]).to_json(
            wl_path,
            orient="records",
            lines=True,
        )
        logger.info(f"Saved weak labels for {name} to {wl_path}")

    model_path = iter_dir / "models" / "label_model.pkl"
    if model_path.exists():
        # Load previously trained model
        logger.info(f"Loading label model from {model_path}")
        trainer.label_model.load(str(model_path))
    else:
        trainer.fit(train_pred_df)
        trainer.label_model.save(model_path)
        logger.info(f"Saved label model to {model_path}")

    # Evaluate the label model on the dev set
    preds = trainer.predict(dev_pred_df)
    metrics = compute_metrics(dev_pred_df["label"], preds)
    metrics_path = iter_dir / "metrics" / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    # Gather texts the model got wrong for next iteration
    wrong_df = dev_df[preds != dev_pred_df["label"]][["text", "label"]]
    wrong = [
        {"text": t, "label": str(l)}
        for t, l in zip(wrong_df["text"], wrong_df["label"])
    ]
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
    logger.remove() #remove the old handler. Else, the old one will work along with the new one you've added below'
    logger.add(sys.stderr, level="INFO") 
    main()
