# LangGraph & Agent Integration

In complex multi-agent workflows built with LangGraph, checking the semantic cache before executing expensive reasoning nodes can dramatically reduce latency and costs.

## Installation

```bash
uv pip install "longcache[langchain]"
# or
pip install "longcache[langchain]"
```

---

## State Node Caching with `CachedState`

`CachedState.check()` inspects incoming state (such as the latest message in a conversation thread) and checks if an identical or semantically similar interaction was previously resolved.

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from longcache import LongCache
from longcache.integrations.langgraph import CachedState

cache = LongCache()

class AgentState(TypedDict):
    messages: list[dict]
    response: str
    cache_hit: bool

def cache_node(state: AgentState) -> dict:
    # Check cache based on the latest user message
    cache_result = CachedState.check(state, cache, query_key="messages", namespace="agent_faq")
    if cache_result.get("hit"):
        return {
            "response": cache_result["response"],
            "cache_hit": True
        }
    return {"cache_hit": False}

def route_after_cache(state: AgentState):
    if state.get("cache_hit"):
        return END
    return "agent_reasoning"

def agent_reasoning_node(state: AgentState) -> dict:
    query = state["messages"][-1]["content"]
    # Execute LLM tools / agent reasoning loop
    answer = call_expensive_agent(query)
    
    # Store answer into cache for future runs
    cache.set(query, answer, namespace="agent_faq")
    return {"response": answer, "cache_hit": False}

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("check_cache", cache_node)
workflow.add_node("agent_reasoning", agent_reasoning_node)

workflow.set_entry_point("check_cache")
workflow.add_conditional_edges("check_cache", route_after_cache)
workflow.add_edge("agent_reasoning", END)

app = workflow.compile()
```
