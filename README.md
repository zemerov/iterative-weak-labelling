# iterative-weak-labelling

## Criteria generation

Run `generate_criteria.py` to produce or extend criteria lists.

```bash
python generate_criteria.py --dataset DATASET_NAME --output out.jsonl [--existing criteria.jsonl]
```

The JSONL file stores one object per line with fields `criterion`, `description` and `class`.
When an existing file is provided, new criteria are deduplicated against it using GPT‑4.1‑2025‑04‑14.
