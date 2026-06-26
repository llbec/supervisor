from __future__ import annotations


JSON_REPAIR_PROMPT = "上一次输出不是合法JSON。请只输出一个合法JSON对象，不要输出任何解释。"


def normalize_prompt(prompt: str) -> str:
    return (
        "你是工地施工安全视频审核模型。必须只输出合法JSON对象，不要输出Markdown，不要输出解释性文本。\n"
        f"{prompt}"
    )
