# AI sommelier solution v1

`ai_somelier.solution` is a stateless Python package. It does not start a server and keeps no
session state. A future runner can call `respond(request)` once per turn and persist the returned
raw `message` in its own dialogue history.

```python
from ai_somelier.solution import respond

request = {
    "wine": recognized_catalog_row,
    "candidates": top_five_catalog_rows,
    "history": [],
}
result = respond(request)
```

For the first turn, `history` is empty. The provider receives a strict JSON Schema and the result
contains parsed `content`, exactly two `suggestions`, and the raw OpenRouter assistant `message`.
The raw message retains `reasoning` and `reasoning_details` and should be stored unchanged.

For a later turn, append that raw message and the current user message:

```python
history = [
    first_result["message"],
    {"role": "user", "content": "Подойдёт к утке с ягодным соусом?"},
]
result = respond(
    {
        "wine": recognized_catalog_row,
        "candidates": top_five_catalog_rows,
        "history": history,
    },
)
```

Later turns request ordinary text rather than structured output, so `suggestions` is `None` and
`content` is the provider message content. `message` remains suitable for the next history turn.

The default expert file is the filled `ai_somelier/kb/EXPERT_KNOWLEDGE.json`. A runner may point
to another compatible file and override other settings without changing solution code:

```python
request["runtime"] = {
    "expert_knowledge_path": "ai_somelier/data/expert_knowledge.json",
    "timeout_seconds": 20.0,
}
```

The call uses `deepseek/deepseek-v4.1-flash` through OpenRouter and reads the API key from
`OPENROUTER_API_KEY`. `OPENROUTER_SITE_URL` and `OPENROUTER_APP_TITLE` are optional. The result's
`_trace` records model, prompt version/hash, selected knowledge axes, request, and raw response for
future reproducible runs.
