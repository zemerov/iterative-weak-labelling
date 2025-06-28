from __future__ import annotations

import json
from jinja2 import Template
from loguru import logger

from .llm_client import LLMQueryClient

class CriteriaGenerator:
    def __init__(self, generate_criteria_file: str, deduplicate_criteria_file: str, model: str = "gpt-4.1-2025-04-14") -> None:
        with open(generate_criteria_file, "r", encoding="utf-8") as file:
            self.generate_criteria_template = Template(file.read())

        with open(deduplicate_criteria_file, "r", encoding="utf-8") as file:
            self.deduplicate_criteria_template = Template(file.read())

        self.llm_client = LLMQueryClient()
        self.model = model
        logger.info(f"CriteriaGenerator initialised with model {model}")

    def _deduplication_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "unique_criteria": {
                "type": "array",
                "items": {
                    "type": "string"
                }
                }
            },
            "required": ["unique_criteria"],
            "additionalProperties": False
            }

    def _generation_schema(self) -> dict:
        """JSON schema for the criteria generation response."""
        return {
            "type": "object",
            "properties": {
                "criteria": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "criterion": {"type": "string"},
                            "description": {"type": "string"},
                            "class": {"type": "string"},
                        },
                        "required": ["criterion", "description", "class"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["criteria"],
            "additionalProperties": False,
        }

    def _llm_json(self, prompt: str, schema: dict | None = None) -> dict | list:
        messages = [{"role": "system", "content": prompt}]
        logger.debug(f"Sending prompt to LLM:\n{prompt}")
        result = self.llm_client.generate(
            messages,
            model=self.model,
            temperature=0.7,
            schema=schema,
        )
        return json.loads(result) if isinstance(result, str) else result

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
        schema = self._generation_schema()
        result = self._llm_json(prompt, schema=schema)
        return result["criteria"]

    def deduplicate_new_criteria(
        self,
        existing: list[dict[str, str]],
        new: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        all_criteria = existing + new

        prompt = self.deduplicate_criteria_template.render(
            criteria=json.dumps(all_criteria, ensure_ascii=False, indent=2)
        )
        
        schema = self._deduplication_schema()
        deduped_criteria_names = self._llm_json(prompt, schema=schema)
        print(deduped_criteria_names)
        deduped_criteria_names = deduped_criteria_names['unique_criteria']
        
        return [
            x for x in all_criteria if x['criterion'] in deduped_criteria_names
        ]
