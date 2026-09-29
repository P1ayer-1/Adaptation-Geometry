"""Versioned prompt templates. A template change creates a new dataset version (spec §10.3).

``v1`` is the canonical training/evaluation format; ``v1-alt`` is the alternate format used
for robustness evaluation so that maps cannot exploit template-specific artifacts.

Each template is split into a ``header`` (the instruction) and a ``query`` (one input), so
the same template can also build a *few-shot* prompt: header, then k worked examples
(query + target), then the query. The few-shot base is the reference that separates what an
adapter teaches from what k demonstrations of the output format already give the base.
"""

from __future__ import annotations

from typing import Any, Sequence

TEMPLATES: dict[str, dict[str, str]] = {
    "v1": {"header": "{instruction}\n\n", "query": "Input: {input}\nOutput:", "target_prefix": " ",
           "shot_separator": "\n\n", "stop": "\nInput:"},
    "v1-alt": {"header": "### Instruction\n{instruction}\n\n", "query": "### Input\n{input}\n\n### Response\n",
               "target_prefix": "", "shot_separator": "\n\n", "stop": "\n### Input"},
}


def _template(version: str) -> dict[str, str]:
    try:
        return TEMPLATES[version]
    except KeyError as e:
        raise KeyError(f"unknown prompt template {version!r}; known {sorted(TEMPLATES)}") from e


def build_prompt(instruction: str, input_text: str, version: str = "v1") -> str:
    tmpl = _template(version)
    return tmpl["header"].format(instruction=instruction) + tmpl["query"].format(input=input_text)


def target_text(target: str, version: str = "v1") -> str:
    return _template(version)["target_prefix"] + target


def build_fewshot_prompt(instruction: str, shots: Sequence[dict[str, Any]], input_text: str,
                         version: str = "v1") -> str:
    """Header, then each shot as ``query + target``, then the query. With no shots this equals
    :func:`build_prompt` exactly."""
    tmpl = _template(version)
    parts = [tmpl["header"].format(instruction=instruction)]
    for ex in shots:
        parts.append(tmpl["query"].format(input=ex["input"]) + target_text(ex["target"], version)
                     + tmpl["shot_separator"])
    parts.append(tmpl["query"].format(input=input_text))
    return "".join(parts)


def stop_marker(version: str = "v1") -> str:
    """Where a few-shot continuation starts inventing the next example; generations are cut here."""
    return _template(version)["stop"]
