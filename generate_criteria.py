import argparse
import json
from loguru import logger
from datasets import load_dataset

from src.criteria_generator import CriteriaGenerator


def load_samples(dataset_name: str, num_samples: int, split: str = "test") -> tuple[list[str], list[str]]:
    logger.debug(f"Loading dataset {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)
    text_col = next((c for c in ["text", "sentence", "utterance"] if c in dataset.column_names), dataset.column_names[0])
    label_col = next((c for c in ["label", "labels", "intent"] if c in dataset.column_names), dataset.column_names[-1])
    texts = dataset[text_col][:num_samples]
    raw_labels = dataset[label_col][:num_samples]
    label_feature = dataset.features.get(label_col)
    
    if hasattr(label_feature, "int2str"):
        labels = [label_feature.int2str(v) for v in raw_labels]
    else:
        labels = [str(v) for v in raw_labels]
    return texts, labels


def read_criteria(path: str) -> list[dict[str, str]]:
    logger.debug(f"Reading existing criteria from {path}")
    criteria = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            criteria.append(json.loads(line))
    return criteria


def generate_criteria(dataset, output, existing=None, samples=100) -> None:
    args = parser.parse_args()
    logger.info(
        f"Generating criteria for dataset '{ args.dataset}' using {args.samples} samples"
    )

    texts, labels = load_samples(args.dataset, args.samples)
    logger.debug(f"Loaded {len(texts)} texts")

    existing = read_criteria(args.existing) if args.existing else {}
    if existing:
        logger.info(f"Loaded {len(existing)} existing criteria from {args.existing}")

    generator = CriteriaGenerator(
        "prompts/lf_generation.txt", 
        "prompts/lf_deduplication.txt"
    )

    new_criteria = generator.get_new_criteria(
        args.dataset, texts, labels, existing_criteria=existing if existing else None
    )

    logger.info(f"Generated {len(new_criteria)} new criteria")

    if existing:
        mapping_new = {c["criterion"]: c for c in new_criteria}
        final = generator.deduplicate_new_criteria(existing, new_criteria)
        logger.info(f"After deduplication {len(final)} criteria remain", )
    else:
        final = new_criteria

    logger.info(f"Writing {len(final)} criteria to {args.output}")

    with open(args.output, "w", encoding="utf-8") as f:
        for item in final:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--existing", default=None)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    generate_criteria(
        dataset=args.dataset,
        output=args.output,
        existing=args.exisiting,
        samples=args.samples
    )
