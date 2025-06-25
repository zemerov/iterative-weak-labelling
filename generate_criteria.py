import argparse
import json
from loguru import logger
from datasets import load_dataset

from src.criteria_generator import CriteriaGenerator


def load_samples(dataset_name: str, num_samples: int, split: str = "test") -> tuple[list[str], list[str]]:
    logger.debug("Loading dataset %s", dataset_name)
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


def read_criteria(path: str) -> dict[str, str]:
    logger.debug("Reading existing criteria from %s", path)
    criteria = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            criteria[item["criterion"]] = item["description"]
    return criteria


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--existing", default=None)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    logger.info(
        "Generating criteria for dataset '{}' using {} samples", args.dataset, args.samples
    )

    texts, labels = load_samples(args.dataset, args.samples)
    logger.debug("Loaded {} texts", len(texts))

    existing = read_criteria(args.existing) if args.existing else {}
    if existing:
        logger.info("Loaded {} existing criteria from {}", len(existing), args.existing)

    generator = CriteriaGenerator(
        "prompts/lf_generation.txt", "prompts/lf_deduplication.txt"
    )
    new_criteria_raw = generator.get_new_criteria(
        args.dataset, texts, labels, existing_criteria=existing if existing else None
    )
    new_criteria = new_criteria_raw if isinstance(new_criteria_raw, list) else new_criteria_raw.get("criteria", [])
    logger.info("Generated %d new criteria", len(new_criteria))
    mapping_new = {c["criterion"]: c for c in new_criteria}

    if existing:
        deduped = generator.deduplicate_new_criteria(existing, {k: v["description"] for k, v in mapping_new.items()})
        logger.info("After deduplication %d criteria remain", len(deduped))
        final = [mapping_new[k] for k in deduped]
    else:
        final = new_criteria

    logger.info("Writing %d criteria to %s", len(final), args.output)

    with open(args.output, "w", encoding="utf-8") as f:
        for item in final:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Done")


if __name__ == "__main__":
    main()
