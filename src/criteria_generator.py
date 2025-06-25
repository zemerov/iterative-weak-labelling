from __future__ import annotations

import json
from jinja2 import Template

from llm_client import LLMQueryClient


class CriteriaGenerator:
    def __init__(self, generate_criteria_file: str, deduplicate_criteria_file: str, model: str = "gpt-4.1-2025-04-14") -> None:
        with open(generate_criteria_file, "r", encoding="utf-8") as file:
            self.generate_criteria_template = Template(file.read())

        with open(deduplicate_criteria_file, "r", encoding="utf-8") as file:
            self.deduplicate_criteria_template = Template(file.read())

        self.llm_client = LLMQueryClient()
        self.model = model

    def _llm_json(self, prompt: str) -> dict | list:
        messages = [{"role": "system", "content": prompt}]
        result = self.llm_client.generate(messages, model=self.model, temperature=0)
        return json.loads(result)

    def get_new_criteria(
        self,
        dataset_name: str,
        texts: list[str],
        labels: list[str],
        existing_criteria: dict[str, str] | None = None,
    ) -> list[dict[str, str]]:
        existing = (
            json.dumps(existing_criteria, ensure_ascii=False, indent=2)
            if existing_criteria
            else None
        )
        examples = json.dumps([
            {"text": t, "label": l} for t, l in zip(texts, labels)
        ], ensure_ascii=False, indent=2)
        prompt = self.generate_criteria_template.render(
            dataset_name=dataset_name,
            existing_criteria=existing,
            texts_with_labels=examples,
        )
        return self._llm_json(prompt)

    def deduplicate_new_criteria(
        self,
        existing: dict[str, str],
        new: dict[str, str],
    ) -> dict[str, str]:
        prompt = self.deduplicate_criteria_template.render(
            criteria=json.dumps(new, ensure_ascii=False, indent=2)
        )
        deduped_new = self._llm_json(prompt)
        union = {**existing, **deduped_new}
        prompt = self.deduplicate_criteria_template.render(
            criteria=json.dumps(union, ensure_ascii=False, indent=2)
        )
        deduped_union = self._llm_json(prompt)
        return {k: v for k, v in deduped_union.items() if k not in existing}
