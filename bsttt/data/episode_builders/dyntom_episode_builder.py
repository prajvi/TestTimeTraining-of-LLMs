"""
DynToM episode builder.

For each evaluation question (query), build an episode where:
  - query: the target question (type_d or other)
  - support: a set of PRECEDING questions from the same trial that act as ToM training data.
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Literal
from bsttt.data.loaders.dyntom import DynToMExample


@dataclass(frozen=True)
class DynToMEpisode:
    episode_id: str
    scenario_name: str
    support: List[DynToMExample]   # Support Questions or Turns
    query: DynToMExample            # The target evaluation question
    turn_id: int
    mode: str = "questions"         # "questions" = BSTTT, "turns" = Generic TTT


class DynToMEpisodeBuilder:
    def __init__(self, examples: Sequence[DynToMExample], seed: int = 42):
        self._rng = random.Random(seed)
        # Group by trial (scenario_name)
        self._scenarios: Dict[str, List[DynToMExample]] = {}
        for ex in examples:
            self._scenarios.setdefault(ex.scenario_name, []).append(ex)

    def build_episodes(
        self, 
        support_size: int = 4, 
        mode: Literal["questions", "turns"] = "questions"
    ) -> List[DynToMEpisode]:
        episodes = []
        for name, items in self._scenarios.items():
            # Separate turns and questions
            turns = sorted(
                [it for it in items if it.question_type == "dyntom"],
                key=lambda x: x.turn_id
            )
            # All available questions for the trial
            all_questions = [it for it in items if it.question_type != "dyntom"]
            
            for q in all_questions:
                if mode == "turns":
                    # Current logic: support is raw narrative turns
                    eligible = [t for t in turns if t.turn_id <= q.turn_id]
                    support = eligible[-support_size:] if support_size > 0 else []
                else:
                    # BSTTT Logic: support is a sampled set of OTHER questions from the same 
                    # or preceding turns in the same trial.
                    # This ensures the model learns the SOCIAL LOGIC of the trial.
                    eligible = [oq for oq in all_questions if oq.id != q.id and oq.turn_id <= q.turn_id]
                    if len(eligible) >= support_size:
                        idxs = list(range(len(eligible)))
                        self._rng.shuffle(idxs)
                        support = [eligible[i] for i in idxs[:support_size]]
                    else:
                        # Fallback to whatever is available if trial is small
                        support = eligible
                
                episodes.append(DynToMEpisode(
                    episode_id=q.id,
                    scenario_name=name,
                    support=support,
                    query=q,
                    turn_id=q.turn_id,
                    mode=mode
                ))
        return episodes
