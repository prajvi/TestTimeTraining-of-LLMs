"""
SimpleToM episode builder.

We build episodes from a common `scenario_name` family:
  - support: sampled from *behavior* subset within the scenario family
  - query: sampled from the specified query task subset (mental_state/behavior/judgment)

The query is held out from the support by default.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from bsttt.data.loaders.simpletom import QuestionType, SimpleToMExample, split_examples_by_type


@dataclass(frozen=True)
class SimpleToMEpisode:
    episode_id: str
    scenario_name: str
    support: List[SimpleToMExample]
    query: SimpleToMExample
    query_task: QuestionType


class SimpleToMEpisodeBuilder:
    """
    Build support/query episodes for SimpleToM.

    By default:
      - support_size=k
      - support examples are drawn from `question_type='behavior'`
      - query task is configurable via `query_task`
      - fast reset logic (resetting weights) is handled in trainers, not here
    """

    def __init__(
        self,
        examples: Sequence[SimpleToMExample],
        *,
        seed: int = 42,
    ) -> None:
        self._rng = random.Random(seed)
        self._by_type = split_examples_by_type(examples)

        # Index by scenario_name for quick sampling.
        self._by_scenario_and_type: Dict[str, Dict[QuestionType, List[SimpleToMExample]]] = {}
        for ex in examples:
            bucket = self._by_scenario_and_type.setdefault(ex.scenario_name, {"mental_state": [], "behavior": [], "judgment": []})
            bucket[ex.question_type].append(ex)

    def scenario_names(self) -> List[str]:
        return list(self._by_scenario_and_type.keys())

    def get_mental_state_example(self, scenario_name: str) -> Optional[SimpleToMExample]:
        """Return any mental-state example for the scenario (for ms_reminder baseline)."""
        bucket = self._by_scenario_and_type.get(scenario_name, {})
        ms_list = bucket.get("mental_state", [])
        return ms_list[0] if ms_list else None

    def _sample_query(
        self,
        *,
        scenario_name: str,
        query_task: QuestionType,
    ) -> SimpleToMExample:
        candidates = self._by_scenario_and_type.get(scenario_name, {}).get(query_task, [])
        if not candidates:
            raise ValueError(f"No query candidates for scenario_name='{scenario_name}', task='{query_task}'")
        return self._rng.choice(candidates)

    def _sample_support(
        self,
        *,
        scenario_name: str,
        query: SimpleToMExample,
        support_task: QuestionType,
        support_size: int,
    ) -> List[SimpleToMExample]:
        support_pool = self._by_scenario_and_type.get(scenario_name, {}).get(support_task, [])
        if not support_pool:
            raise ValueError(f"No support candidates for scenario_name='{scenario_name}', task='{support_task}'")

        # Hold out query if it comes from the same task.
        support_pool = [ex for ex in support_pool if ex.id != query.id]
        if len(support_pool) < support_size:
            raise ValueError(
                f"Not enough support examples for scenario_name='{scenario_name}'. "
                f"Need {support_size} from task='{support_task}' excluding query id={query.id}. "
                f"Available: {len(support_pool)}"
            )

        # Sample without replacement.
        idxs = list(range(len(support_pool)))
        self._rng.shuffle(idxs)
        chosen = idxs[:support_size]
        return [support_pool[i] for i in chosen]

    def build_episode(
        self,
        *,
        scenario_name: str,
        query_task: QuestionType,
        support_size: int,
        support_task: QuestionType = "behavior",
        episode_id: Optional[str] = None,
    ) -> SimpleToMEpisode:
        query = self._sample_query(scenario_name=scenario_name, query_task=query_task)
        support = self._sample_support(
            scenario_name=scenario_name,
            query=query,
            support_task=support_task,
            support_size=support_size,
        )
        eid = episode_id or f"{scenario_name}:{query_task}:{query.id}"
        return SimpleToMEpisode(
            episode_id=eid,
            scenario_name=scenario_name,
            support=support,
            query=query,
            query_task=query_task,
        )

    def build_episodes(
        self,
        *,
        query_task: QuestionType,
        support_size: int,
        num_episodes: int,
        support_task: QuestionType = "behavior",
        raise_on_shortfall: bool = True,
        max_attempts: Optional[int] = None,
    ) -> List[SimpleToMEpisode]:
        """
        Build (support, query) episodes.

        Unlike a single-pass over `scenario_names`, we sample scenarios with replacement
        until we reach `num_episodes` (or hit `max_attempts`). This makes evaluation robust
        when some scenario/task combinations are rare.
        """
        episodes: List[SimpleToMEpisode] = []
        scenario_list = self.scenario_names()
        if not scenario_list:
            if raise_on_shortfall:
                raise ValueError(f"No scenario names available for task='{query_task}'.")
            return []

        # Each episode attempt can fail (e.g., not enough support examples after hold-out).
        # We allow multiple attempts to reach the requested episode count.
        attempts = 0
        max_attempts_eff = max_attempts if max_attempts is not None else max(50, num_episodes * 50)

        while len(episodes) < num_episodes and attempts < max_attempts_eff:
            attempts += 1
            scenario_name = self._rng.choice(scenario_list)
            try:
                ep = self.build_episode(
                    scenario_name=scenario_name,
                    query_task=query_task,
                    support_size=support_size,
                    support_task=support_task,
                )
            except ValueError:
                continue
            episodes.append(ep)

        if len(episodes) != num_episodes and raise_on_shortfall:
            raise ValueError(
                f"Could only build {len(episodes)}/{num_episodes} episodes for task='{query_task}' "
                f"(max_attempts={max_attempts_eff})."
            )
        return episodes


def sanity_check_episode_builder(
    *,
    examples: Sequence[SimpleToMExample],
    support_size: int = 4,
    num_episodes: int = 3,
) -> None:
    builder = SimpleToMEpisodeBuilder(examples, seed=0)
    for qt in ["mental_state", "behavior", "judgment"]:
        eps = builder.build_episodes(query_task=qt, support_size=support_size, num_episodes=num_episodes)
        for ep in eps:
            assert len(ep.support) == support_size
            assert ep.query.question_type == qt
            assert all(s.scenario_name == ep.scenario_name for s in ep.support)
            assert ep.query.scenario_name == ep.scenario_name
            assert all(s.question_type == "behavior" for s in ep.support)
            # Hold-out check
            assert all(s.id != ep.query.id for s in ep.support)
        print(f"[sanity_check] built {len(eps)} episodes for query_task='{qt}'")


if __name__ == "__main__":
    from bsttt.data.loaders.simpletom import load_simpletom_processed

    processed = load_simpletom_processed(cache_max_items_per_subset=16, force_rebuild=False)
    sanity_check_episode_builder(examples=processed, support_size=4, num_episodes=2)
    print("Episode builder sanity checks passed.")

