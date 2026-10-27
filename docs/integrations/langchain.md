# LangChain Integration

LongCache provides first-class support for LangChain through `CachedLLM` and `CachedChain`.

## Installation

```bash
uv pip install "longcache[langchain]"
# or
pip install "longcache[langchain]"
```

---

## Caching Chat Models (`CachedLLM`)

`CachedLLM` acts as a drop-in replacement for any LangChain `BaseChatModel` or `BaseLLM`. It maintains the full `invoke()`, `ainvoke()`, `batch()`, and `abatch()` interfaces.

```python
from langchain_openai import ChatOpenAI
from longcache import LongCache
from longcache.integrations.langchain import CachedLLM

base_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
llm = CachedLLM(base_llm, cache=LongCache())

# Call 1: MISS (calls OpenAI API)
response1 = llm.invoke("What is your refund policy?")
print(response1.content)

# Call 2 (paraphrased): HIT (served from LongCache in <5ms, $0.00 cost)
response2 = llm.invoke("How do I return an item?")
print(response2.content)
```

---

## Caching LCEL Chains (`CachedChain`)

You can wrap any compiled LangChain Expression Language (LCEL) chain:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from longcache import LongCache
from longcache.integrations.langchain import CachedChain

prompt = ChatPromptTemplate.from_template("Answer the question briefly: {question}")
chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()

cached_chain = CachedChain(chain, cache=LongCache(), namespace="faq_chain")

# Executes chain and caches result
ans1 = cached_chain.invoke({"question": "Where is the Eiffel Tower?"})

# Instant hit on semantic paraphrase
ans2 = cached_chain.invoke({"question": "In what city can I visit the Eiffel Tower?"})
```
