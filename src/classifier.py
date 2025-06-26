import json
from jinja2 import Template
from loguru import logger

from llm_client import LLMQueryClient


class DialogueCriteriaClassifier:
    def __init__(
        self,
        criteria_dict: dict[str, str],
        prompt_file: str,
        model: str,
        temperature: float = 0.01,
    ):
        """
        criteria_dict: {criteria_name: description} — names and descriptions for criteria to classify
        """
        self.criteria_dict = criteria_dict
        self.model = model
        self.temperature = temperature
        self.template = self._init_template(prompt_file)

        self.llm_client = LLMQueryClient()

    def _init_template(self, prompt_file: str) -> Template:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            template_str = f.read()
        
        return Template(template_str)

    def _construct_few_shot_prompt(self) -> str:
        descriptions = {
            criteria: description for criteria, description in self.criteria_dict.items()
        }

        return self.template.render(existing_small_tags=descriptions)

    def _build_json_schema(self) -> dict:
        properties = {
            "thoughts": {"type": "string"}
            } | {
            criterion: {"type": "boolean"} for criterion in self.criteria_dict
            }
        
        schema = {
            "type": "object",
            "properties": properties,
            "required": ["thoughts"] + list(self.criteria_dict.keys())
        }
        
        return schema

    def classify_text(self, dialogue: str) -> dict:
        system_message = self._construct_few_shot_prompt()
        messages = [
            {"role": "system", "content": system_message}, 
            {"role": "user", "content": dialogue}
        ]
        json_schema = self._build_json_schema()
        
        try:
            result = self.llm_client.generate(
                messages,
                model=self.model,
                temperature=self.temperature,
                schema=json_schema,
            )
        except Exception as e:
            logger.error(f"Error sending LLM request: {repr(e)}")
            return {label: None for label in self.criteria_dict}
        
        try:
            parsed_result = json.loads(result) if isinstance(result, str) else result
            labels = {label: parsed_result.get(label, None) for label in self.criteria_dict}
        except Exception as e:
            logger.error(f"Error parsing LLM response: {repr(e)}. Response: {result}")
            labels = {label: None for label in self.criteria_dict}
        return labels