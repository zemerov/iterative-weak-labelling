import os
from loguru import logger
import openai


class LLMQueryClient:
    VLLM_BASE_URL = "http://localhost:8000/v1"
    VLLM_KEY = 'default-token'

    def __init__(self, max_retry: int = 5):
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        if not openai_api_key or not self.api_key.strip():
            logger.warning("OPENAI_API_KEY is not set. Provide it with OPENAI_API_KEY env variable.")
        else:
            self.openai_client = openai.Client(api_key=openai_api_key)

        self.vllm_client = openai.Client(
            base_url=os.gentenv("VLLM_BASE_URL", self.VLLM_BASE_URL), 
            api_key=os.getenv("VLLM_TOKEN", self.LOCAL_KEY)
        )

    def _get_vllm_answer(
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
        
        ans = completion.choices[0].message.content
        
        return ans
    
    def _openai_generate(self):
        raise NotImplementedError

    def generate(
        self, 
        messages: list[dict], 
        extra_body: dict = None, 
        max_tokens: int = 4096, 
        model: str = None, 
        temperature: float = None
    ) -> str:
        if "gpt" in model:
            return self._get_api_answer(messages, model, temperature, extra_body=extra_body, max_tokens=max_tokens)
        else:
            return self._get_vllm_answer(messages, model, temperature, extra_body=extra_body, max_tokens=max_tokens)