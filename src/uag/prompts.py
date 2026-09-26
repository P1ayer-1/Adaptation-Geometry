"""Versioned prompt templates. A template change creates a new dataset version (spec §10.3).

``v1`` is the canonical training/evaluation format; ``v1-alt`` is the alternate format used
for robustness evaluation so that maps cannot exploit template-specific artifacts.
"""

from __future__ import annotations

TEMPLATES: dict[str, dict[str, str]] = {
    "v1": {"prompt": "{instruction}\n\nInput: {input}\nOutput:", "target_prefix": " "},
    "v1-alt": {"prompt": "### Instruction\n{instruction}\n\n### Input\n{input}\n\n### Response\n",
               "target_prefix": ""},
}


def build_prompt(instruction: str, input_text: str, version: str = "v1") -> str:
    try:
        tmpl = TEMPLATES[version]
    except KeyError as e:
        raise KeyError(f"unknown prompt template {version!r}; known {sorted(TEMPLATES)}") from e
    return tmpl["prompt"].format(instruction=instruction, input=input_text)


def target_text(target: str, version: str = "v1") -> str:
    return TEMPLATES[version]["target_prefix"] + target
