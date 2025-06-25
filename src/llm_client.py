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
            self.openai_client = openai.Client(api_key=openai_api_key)

        self.vllm_client = openai.Client(
            base_url=os.getenv("VLLM_BASE_URL", self.VLLM_BASE_URL),
            api_key=os.getenv("VLLM_TOKEN", self.VLLM_KEY)
        )
        
    def _generate_vllm(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        extra_body: dict = None,
        max_tokens: int = 2048
    ) -> str:
        completion = self.vllm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body
        )
        
        return completion.choices[0].message.content
    
    def _generate_openai(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        extra_body: dict | None = None,
        max_tokens: int = 2048,
    ) -> str:
        if self.openai_client is None:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        completion = self.openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )

        return completion.choices[0].message.content

    def generate(
        self,
        messages: list[dict],
        extra_body: dict = None,
        max_tokens: int = 4096,
        model: str = None,
        temperature: float = None
    ) -> str:
        if "gpt" in model:
            return self._generate_openai(
                messages,
                model,
                temperature,
                extra_body=extra_body,
                max_tokens=max_tokens,
            )
        return self._generate_vllm(
            messages,
            model,
            temperature,
            extra_body=extra_body,
            max_tokens=max_tokens,
        )

