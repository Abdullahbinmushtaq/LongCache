# Similarity & Thresholds

Selecting the correct similarity threshold is key to balancing cache hit rate with precision.

## Understanding Similarity Scores

Cosine similarity ranges between **0.0** (completely unrelated) to **1.0** (identical semantic meaning).

| Threshold Range | Match Character | Typical Use Case |
|---|---|---|
| **0.95 – 1.00** | Near-exact paraphrases | Highly sensitive compliance, financial transactions, legal analysis |
| **0.90 – 0.94** *(Default: 0.92)* | Natural paraphrasing | Customer support, FAQ bots, documentation search, RAG agents |
| **0.80 – 0.89** | Broad topical match | Exploratory search, recommendation prompts, general chat |
| **< 0.80** | Loose match | Risk of false-positive cache hits (not recommended for QA) |

---

## Example Match Scores

Here is how real query pairs score under `BAAI/bge-small-en-v1.5`:

| Query A | Query B | Score | Hit at 0.92? |
|---|---|---|---|
| "What is your refund policy?" | "How do I get a refund?" | **0.942** | ✅ Yes |
| "What are your support hours?" | "When is customer service open?" | **0.931** | ✅ Yes |
| "How to reset my password?" | "I forgot my password, how to fix?" | **0.928** | ✅ Yes |
| "What is your refund policy?" | "What is your cancellation policy?" | **0.841** | ❌ No (Different concepts) |
| "What is your refund policy?" | "How to bake a chocolate cake?" | **0.213** | ❌ No |

---

## Tuning the Threshold

You can tune the threshold globally in `CacheConfig` or override it per function call:

```python
from longcache import LongCache, CacheConfig

# Strict threshold for sensitive medical or legal bot
strict_config = CacheConfig(similarity_threshold=0.96)
strict_cache = LongCache(strict_config)

# Permissive threshold for general FAQ
faq_config = CacheConfig(similarity_threshold=0.88)
faq_cache = LongCache(faq_config)
```
