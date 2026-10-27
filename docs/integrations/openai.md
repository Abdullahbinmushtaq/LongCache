# OpenAI Integration

LongCache provides a transparent client wrapper `CachedOpenAI` around the official `openai` Python SDK.

## Installation

```bash
uv pip install "longcache[openai]"
# or
pip install "longcache[openai]"
```

---

## Caching Chat Completions (`CachedOpenAI`)

Wrap an `OpenAI` client instance without modifying how you call `chat.completions.create()`:

```python
from openai import OpenAI
from longcache import LongCache
from longcache.integrations.openai import CachedOpenAI

client = CachedOpenAI(
    OpenAI(api_key="your-openai-api-key"),
    cache=LongCache(),
    namespace="production_chat"
)

# First request — executes remote OpenAI API call
completion1 = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a concise customer support assistant."},
        {"role": "user", "content": "What is the return policy for international orders?"}
    ],
    temperature=0.0
)
print(completion1.choices[0].message.content)

# Second request (paraphrased) — served instantly from local LongCache!
completion2 = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a concise customer support assistant."},
        {"role": "user", "content": "Can I return an item if I ordered from outside the country?"}
    ],
    temperature=0.0
)
print(completion2.choices[0].message.content)
```

## Anti-Poisoning Protection

`CachedOpenAI` automatically derives the composite key from:
- The `model` name (e.g. `gpt-4o` vs `gpt-4o-mini`)
- The `temperature` parameter
- The system instructions message

This guarantees that a response generated under `temperature=0.0` with strict tone will never be accidentally served to a query requesting creative brainstorming (`temperature=1.0`).
