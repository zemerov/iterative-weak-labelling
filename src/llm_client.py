import json
import os
from loguru import logger
import openai


class LLMQueryClient:
    VLLM_BASE_URL = "http://localhost:8000/v1"
    VLLM_KEY = "default-token"

    def __init__(self, max_retry: int = 5) -> None:
        openai_api_key = os.getenv("OPENAI_API_KEY", "")

        if not openai_api_key.strip():
            logger.warning(
                "OPENAI_API_KEY is not set. Provide it with OPENAI_API_KEY env variable."
            )
            self.openai_client = None
        else:
            self.openai_client = openai.Client(api_key=openai_api_key, timeout=120, max_retries=3)

        self.vllm_client = openai.Client(
            base_url=os.getenv("VLLM_BASE_URL", self.VLLM_BASE_URL),
            api_key=os.getenv("VLLM_TOKEN", self.VLLM_KEY)
        )
        
    def _generate_vllm(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        schema: dict | None,
        max_tokens: int = 2048,
    ) -> str:
        if schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": False,
                },
            }
        else:
            response_format = {"type": "json_object"}
        completion = self.vllm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

        return completion.choices[0].message.content
    
    def _generate_openai(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        schema: dict | None,
        max_tokens: int = 2048,
    ) -> str:
        if self.openai_client is None:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        if schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": schema,
                    "strict": True,
                },
            }
        else:
            response_format = {"type": "json_object"}
        completion = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

        return completion.choices[0].message.content

    def generate(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        schema: dict | None = None,
        max_tokens: int = 4096,
    ) -> str:
        if "gpt" in model:
            content = self._generate_openai(
                messages,
                model,
                temperature,
                schema,
                max_tokens=max_tokens,
            )
        else:
            content = self._generate_vllm(
                messages,
                model,
                temperature,
                schema,
                max_tokens=max_tokens,
            )

        return json.loads(content)

