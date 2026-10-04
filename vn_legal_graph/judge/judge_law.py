"""judge_law: LLM check of whether a retrieved article applies to a case.

Port of the original repo's core/judge/judge_law.py:
1. For every judge_dep element of the article, ask true/false
   (JUDGE_ELEMENT_PROMPT, one call per element).
2. Ask once more whether the article applies, given the elements found
   true and false (JUDGE_LAW_FINAL_PROMPT).

``mode="gop"`` replaces step 1 by a single call per article that answers
all elements as a JSON object keyed by element number
(JUDGE_ELEMENTS_BATCH_PROMPT). That variant is
not in the original; it exists because step 1 costs one call per element
(30-40 for the drug articles).

The final prompt adds one rule the original lacks: an article applies when
its basic elements (usually khoản 1) hold, and unmet aggravating
circumstances do not count against it. Without it, the trial run rejected
8 of 23 correct articles: the core element was answered true, but the
37-39 aggravating-circumstance questions of the drug articles were false
and the final call read that as "does not apply". A second added rule
handles exclusions in the article text itself (Điều 249: "mà không nhằm
mục đích mua bán ..."): if the excluded element holds, the article does not
apply. The trial accepted 249 for a case where "Có nhằm mục đích mua bán
... không?" had been answered true.

The original treats a missing or unreadable answer as "not true" ("true"
not in answer). Here an element answer that is neither true nor false is
kept apart as unknown so it can be counted.
"""
from __future__ import annotations

import json
import re
from typing import Callable, Dict, List, Optional, Sequence

from ..prompts.vi import JUDGE_ELEMENT_PROMPT, JUDGE_ELEMENTS_BATCH_PROMPT, JUDGE_LAW_FINAL_PROMPT

Generate = Callable[..., str]

MODES = ("trung-thanh", "gop")
RELATED_MAX_CHARS = 6000
CASE_MAX_CHARS = 6000


def parse_bool(text: str) -> Optional[bool]:
    """First "true"/"false" in the answer, or None."""
    m = re.search(r"\b(true|false)\b", text.lower())
    return None if m is None else m.group(1) == "true"


def parse_numbered_bools(text: str, n: int) -> Optional[List[Optional[bool]]]:
    """A JSON object {"1": true, "2": false, ...} -> list of n answers, None
    where an element is missing or not a boolean. Returns None only if no
    JSON object can be read.

    Keyed by element number because gpt-4o-mini, asked for a plain list of
    36-45 booleans, returned 1-2 items too many or too few in 22 of 31
    trial calls, which made the whole answer unusable."""
    first, last = text.find("{"), text.rfind("}")
    if first == -1 or last < first:
        return None
    try:
        data = json.loads(text[first : last + 1].lower())
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    out: List[Optional[bool]] = []
    for i in range(1, n + 1):
        value = data.get(str(i))
        out.append(value if isinstance(value, bool) else None)
    return out


def render_related(related_laws: Sequence) -> str:
    """Guidance documents first, then cross-referenced articles, cut at
    RELATED_MAX_CHARS. Cross-references can be whole articles (Điều 249
    cites Điều 248, ~2,000 chars) and used to push the guidance past the
    cut."""
    items = list(related_laws or [])
    items.sort(key=lambda x: 0 if isinstance(x, dict) and x.get("loai") == "van_ban_huong_dan" else 1)
    parts = []
    for item in items:
        if isinstance(item, dict):
            parts.append(f"{item.get('id', '')}: {item.get('text', '')}")
        else:
            parts.append(str(item))
    return " | ".join(parts)[:RELATED_MAX_CHARS]


def element_prompts(case_text: str, law: Dict) -> List[str]:
    return [
        JUDGE_ELEMENT_PROMPT.format(
            law=law["description"].replace("\n", " "),
            related=render_related(law.get("related_laws")),
            element=element,
            case=case_text[:CASE_MAX_CHARS],
        )
        for element in law["judge_dep"]
    ]


def batch_prompt(case_text: str, law: Dict) -> str:
    elements = "\n".join(f"{i}. {e}" for i, e in enumerate(law["judge_dep"], 1))
    return JUDGE_ELEMENTS_BATCH_PROMPT.format(
        n=len(law["judge_dep"]),
        law=law["description"].replace("\n", " "),
        related=render_related(law.get("related_laws")),
        elements=elements,
        case=case_text[:CASE_MAX_CHARS],
    )


def final_prompt(case_text: str, law: Dict, true_list: List[str], false_list: List[str]) -> str:
    return JUDGE_LAW_FINAL_PROMPT.format(
        case=case_text[:CASE_MAX_CHARS],
        law=law["description"],
        true_list=true_list,
        false_list=false_list,
    )


def judge_law(generate: Generate, case_text: str, law: Dict, mode: str = "trung-thanh") -> Dict:
    """Returns {"ap_dung": bool, "dung": [...], "sai": [...], "khong_ro": [...],
    "loi_doc_danh_sach": bool, "tra_loi_cuoi": str}."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    elements = list(law.get("judge_dep") or [])
    true_list: List[str] = []
    false_list: List[str] = []
    unknown: List[str] = []
    batch_failed = False

    if mode == "trung-thanh":
        for element, prompt in zip(elements, element_prompts(case_text, law)):
            verdict = parse_bool(generate(prompt, max_tokens=16))
            (true_list if verdict else false_list if verdict is False else unknown).append(element)
    elif elements:
        verdicts = parse_numbered_bools(
            generate(batch_prompt(case_text, law), max_tokens=1024), len(elements)
        )
        if verdicts is None:
            batch_failed = True
            unknown = elements
        else:
            for element, verdict in zip(elements, verdicts):
                (true_list if verdict else false_list if verdict is False else unknown).append(element)

    answer = generate(final_prompt(case_text, law, true_list, false_list), max_tokens=16)
    return {
        "ap_dung": parse_bool(answer) is True,
        "dung": true_list,
        "sai": false_list,
        "khong_ro": unknown,
        "loi_doc_danh_sach": batch_failed,
        "tra_loi_cuoi": answer,
    }


def rerank_by_judgment(predicted: Sequence[int], applies: Dict[int, bool]) -> List[int]:
    """Articles judged applicable first, then the rest, each group keeping
    its retrieval order. Articles not judged stay after both groups."""
    accepted = [a for a in predicted if applies.get(a) is True]
    rejected = [a for a in predicted if applies.get(a) is False]
    unjudged = [a for a in predicted if a not in applies]
    return accepted + rejected + unjudged
