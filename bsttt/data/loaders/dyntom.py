"""
DynToM loader.
Normalize the Hugging Face `YangXiao-nlp/DynToM` dataset into a common schema.

Dataset structure (per trial):
  trial_id -> {
    "stage": {
      "main character": "Henry Lozano",
      "story": {
        "scenario 1": { "background": "...", "dialogue": [...] },
        ...
        "scenario 5": { ... }
      },
      "sketch": {
        "mental states analysis in every scenario": {
          "scenario 1": { "belief": "...", "emotion": "...", "intention": "...", "action": "..." },
          ...
        }
      }
    },
    "question": {
      "type_a_what_1": {
        "question": "What is the belief of X in scenario 1?",
        "true answer": "d",   # a letter a-h
        "options": ["a. ...", "b. ...", ...]
      },
      ...
    }
  }
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Optional
from huggingface_hub import hf_hub_download
import re


@dataclass(frozen=True)
class DynToMExample:
    id: str
    story: str      # Scenario text(s) used as context
    question: str   # Question text ("")  for Turns
    choices: List[str]     # Option strings ("a. ...")
    answer: str            # True answer letter ("a"-"h") or "" for Turns
    scenario_name: str     # trial_id
    turn_id: int           # Which scenario number (1-5)
    action: str = ""       # Golden action for TTT
    main_character: str = ""
    characters_info: str = ""  # Full character descriptions + relationships
    question_type: str = "dyntom"   # "dyntom" = Turn, "dyntom_q" = Question


def load_dyntom_processed(
    *,
    split: str = "train",
    limit: Optional[int] = None,          # Max number of QUESTIONS to load (not total examples)
    streaming: bool = True,
    trials_limit: Optional[int] = None,   # Max number of trials to load
) -> List[DynToMExample]:
    print(f"Loading DynToM.json from Hugging Face Hub...")
    path = hf_hub_download(
        repo_id="YangXiao-nlp/DynToM",
        filename="DynToM.json",
        repo_type="dataset"
    )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total trials in DynToM.json: {len(data)}")

    examples: List[DynToMExample] = []
    n_questions = 0

    for trial_id, record in data.items():
        # --- Respect trials_limit ---
        if trials_limit is not None and len([e for e in examples if e.question_type == "dyntom" and e.turn_id == 1]) >= trials_limit:
            break

        stage = record.get("stage", {})
        main_char = stage.get("main character", "")
        chars_info = stage.get("characters information", "")
        social_setting = stage.get("social setting", "")
        story_dict = stage.get("story", {})
        sketch = stage.get("sketch", {})
        mental_states = sketch.get("mental states analysis in every scenario", {})
        relationships = sketch.get("relationships among characters", {})
        
        # Build full character context string
        char_context_parts = []
        if social_setting:
            char_context_parts.append(f"Setting: {social_setting}")
        if chars_info:
            char_context_parts.append(f"{chars_info}")
        if relationships:
            for rel_key, rel_val in relationships.items():
                if isinstance(rel_val, str):
                    char_context_parts.append(f"Relationship: {rel_val}")
        full_char_info = "\n".join(char_context_parts)

        def _snum(key: str) -> int:
            m = re.search(r"(\d+)$", str(key))
            return int(m.group(1)) if m else 0

        scenario_keys = sorted(story_dict.keys(), key=_snum)

        # Build a map from scenario number -> full text (background + dialogue)
        scenario_texts: dict = {}
        scenario_actions: dict = {}
        for skey in scenario_keys:
            m = re.search(r"(\d+)$", str(skey))
            s_num = int(m.group(1)) if m else 0
            s_data = story_dict[skey]
            bg = s_data.get("background", "")
            dlg_str = ""
            for d in s_data.get("dialogue", []):
                if isinstance(d, dict):
                    for char, text in d.items():
                        if text:
                            dlg_str += f"{char}: {text}\n"
                elif isinstance(d, str) and d:
                    dlg_str += d + "\n"
            scenario_texts[s_num] = f"Scenario {s_num}:\nBackground: {bg}\nDialogue:\n{dlg_str}"
            # mental_states keys may be "scenario 1" or "scenario_1"
            ms_val = (mental_states.get(f"scenario {s_num}") or
                      mental_states.get(f"scenario_{s_num}") or {})
            scenario_actions[s_num] = ms_val.get("action", "")

        # --- 1. Extract Scenario Turns (used as support for TTT) ---
        for s_num, text in sorted(scenario_texts.items()):
            examples.append(DynToMExample(
                id=f"{trial_id}_s{s_num}",
                story=text,
                question="",
                choices=[],
                answer="",
                scenario_name=trial_id,
                turn_id=s_num,
                action=scenario_actions[s_num],
                main_character=main_char,
                characters_info=full_char_info,
                question_type="dyntom",
            ))

        # --- 2. Extract Evaluation Questions ---
        questions = record.get("question", {})
        for q_id, q_data in questions.items():
            if limit is not None and n_questions >= limit:
                break

            q_text = q_data.get("question", "")
            if not q_text:
                continue

            # Find which scenario this question targets
            match = re.search(r"scenario (\d+)", q_text.lower())
            target_s_num = int(match.group(1)) if match else max(scenario_texts.keys(), default=5)

            # The story context = the target scenario text
            story_ctx = scenario_texts.get(target_s_num, "")

            choices = q_data.get("options", [])
            true_letter = q_data.get("true answer", "").strip().lower()
            
            # Find the full choice string starting with that letter
            full_answer = ""
            for c in choices:
                if c.lower().startswith(f"{true_letter}."):
                    full_answer = c
                    break
            if not full_answer and choices and true_letter:
                # Fallback: index by letter position
                idx = ord(true_letter) - ord('a')
                if 0 <= idx < len(choices):
                    full_answer = choices[idx]

            examples.append(DynToMExample(
                id=f"{trial_id}_{q_id}",
                story=story_ctx,
                question=q_text,
                choices=choices,
                answer=full_answer,
                scenario_name=trial_id,
                turn_id=target_s_num,
                main_character=main_char,
                characters_info=full_char_info,
                action="",
                question_type=q_data.get("question type", "dyntom_q"),
            ))
            n_questions += 1

    print(f"Loaded {len(examples)} total examples: "
          f"{sum(1 for e in examples if e.question_type == 'dyntom')} turns + "
          f"{n_questions} questions from DynToM.")
    return examples
