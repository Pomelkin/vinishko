---
name: ai-somelier-dev
description: Develop the repository's post-recognition digital sommelier and its expert knowledge base under ai_somelier. Use for requirements, knowledge axes, expert templates, system prompts, and implementation of the sommelier.
---

# AI sommelier development

The digital sommelier uses DeepSeek v4.1 Flash after the scanner has resolved one Catalog
card. Treat `ai_somelier/docs/REQUIREMENTS.md` as the product requirements.

## Current contract

- The only wine used to build a session's expert knowledge is the recognized wine.
- Select matching knowledge once from that wine's fields and inject the compiled knowledge into
  the system prompt once when the chat is created.
- Do not retrieve or inject new expert blocks based on later user messages, dishes, occasions,
  preferences, top-n candidates, or other wines.
- Read wine data from `data/catalog/catalog.csv` with a real CSV parser.
- Preserve unknowns: `Игристое = не определено` does not prove a still wine, and an empty
  positive flag does not prove a negative value.

## Expert knowledge files

- `ai_somelier/docs/EXPERT_KNOWLEDGE_TEMPLATE.json` is the strict fillable template. Its shape is
  `axis -> value -> string[]`; every string is one free-text expert bullet.
- The fixed axes are `color`, `effervescence`, `sweetness`, `sparkling_method`, `special_style`,
  `fortified`, `alcohol_band`, `grape_composition`, `grape`, `region`, `brand`,
  `aging_or_reserve`, and `organic`.
- `brand` contains every exact `Винодельня` value from the Catalog. `grape` contains every exact
  non-empty `Сорт винограда` value plus `unknown`.
- Expert bullets for one value should total about 400 characters, must not exceed 500 characters,
  and may be shorter when the subject is simple.
- `ai_somelier/docs/EXPERT_KNOWLEDGE_PLAN.md` states only what the expert should cover for each
  axis.

Do not fill expert knowledge, rename axes or values, or extend the template unless the user asks.
Keep this skill limited to confirmed project state; do not add plans or hypotheses on your own.
