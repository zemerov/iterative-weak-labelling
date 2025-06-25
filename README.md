# iterative-weak-labelling

## Criteria generation

Run `generate_criteria.py` to produce or extend criteria lists. Generation uses structured output with a JSON schema.

```bash
python generate_criteria.py --dataset DATASET_NAME --output out.jsonl [--existing criteria.jsonl]
```

The JSONL file stores one object per line with fields `criterion`, `description` and `class`.
Both generation and deduplication rely on the GPT‑4.1‑2025‑04‑14 model.
