import os
"""
LLM client wrapper using the Poe API via OpenAI-compatible interface.
Uses client.responses.create (OpenAI Responses API format).
"""

import openai

client = openai.OpenAI(
    api_key=os.getenv("POE_API_KEY", ""),
    base_url="https://api.poe.com/v1",
)

DEFAULT_MODEL = "GPT-5.4-Nano"


def chat(
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> str:
    """
    Send a single system+user turn and return the model's text response.

    Uses the Responses API format:
      client.responses.create(model=..., input=[...])
    """
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.output_text.strip()
