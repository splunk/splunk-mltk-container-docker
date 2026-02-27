#!/usr/bin/env python3
"""Demo of how to use LiteLLM proxy to call different LLM providers"""

import openai
from dotenv import load_dotenv
import os
import random
load_dotenv()

print("=" * 60)
print("TESTING LOCAL MODELS (DIRECT ACCESS)")
print("=" * 60)

# Direct client for local Ollama models
direct_client = openai.OpenAI(
    api_key="not-needed",  # Ollama doesn't need API key
    base_url="http://localhost:11438/v1"  # Direct to Ollama
)

# Test 2 local models directly
for model in [
    "llama3.1:8b-instruct-q8_0",     # Local model 1
    "llama3.2:3b-instruct-q8_0"      # Local model 2
    ]:
    a, b = random.randint(1, 9), random.randint(1, 9)
    print(f"\n[DIRECT] Request to: {model}: What is {a} + {b}?")
    try:
        response = direct_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"What is {a} + {b}?"}],
            temperature=0.0,
            max_tokens=20
        )
        print(f"[DIRECT] Model answered: {str(response.choices[0].message.content)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("TESTING VIA LITELLM PROXY")
print("=" * 60)

# LiteLLM proxy client
litellm_client = openai.OpenAI(
    api_key=os.getenv("LITELLM_MASTER_KEY"),                                # Create virtual keys at /ui/?page=api-keys with custom quotas, throttling, expiration rules. See: https://docs.litellm.ai/docs/proxy/virtual_keys
    base_url=f"http://localhost:{os.getenv('RAW_LOGGING_PORT', '7011')}/v1" # LITELLM_PORT -> goes directly to LiteLLM proxy; RAW_LOGGING_PORT -> goes through 'raw_logger_proxy' container first
)

# Test 2 models via LiteLLM
for model in [
    # Syntax: provider/model ; provider_API_KEY= must exist in .env
    # Model specifics configs and general fallback are configured in litellm_config.yaml
    "ollama/llama3.1:8b-instruct-q4_K_M",           # Supported by LiteLLM: Caught via: litellm_config.yaml / - model_name: "ollama/*"
    "requesty/openai/gpt-4.1-nano-2025-04-14"       # Routed via LiteLLM: Caught via: litellm_config.yaml / - model_name: requesty/*
    ]:
    a, b = random.randint(1, 9), random.randint(1, 9)
    print(f"\n[LITELLM] Request to: {model}: What is {a} + {b}?")
    try:
        response = litellm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": f"What is {a} + {b}?"}],
            temperature=0.0,
            max_tokens=20
        )
        print(f"[LITELLM] Model answered: {str(response.choices[0].message.content)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("SUMMARY:")
print("- Direct access: Connects directly to Ollama at localhost:11438")
print("- LiteLLM proxy: Routes through localhost:7011 with logging")
print("=" * 60)
