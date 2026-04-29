"""
BSTTT V1 trainer for SimpleToM.

Implements episodic LoRA-based test-time adaptation using:
  - action reconstruction loss only

Action reconstruction:
  For each support example with observed action (answer choice), maximize
  log p(correct_option_tokens | prompt_tokens).
  Operationally: CE over option log-likelihoods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from bsttt.data.loaders.simpletom import SimpleToMExample
from bsttt.models.hf_lm_wrapper import HFCausalLMWrapper
from bsttt.trainers.prompt_baselines import build_ms_reminder_prompt, build_vanilla_prompt


SimpleToMEpisodeRow = Dict[str, Any]


@dataclass(frozen=True)
class BSTTTLoRAConfig:
    adapt_steps: int = 3
    lr: float = 1e-4
    weight_decay: float = 0.0
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    # BSTTT Milestone 7: Temporal Smoothness
    enable_temporal_smoothness: bool = False
    temporal_smoothness_weight: float = 0.5
    # BSTTT V2: belief-action consistency
    enable_belief_action_consistency: bool = False
    consistency_margin: float = 0.1
    consistency_weight: float = 1.0
    lora_top_fraction: float = 0.25  # apply LoRA to top N% of layers
    max_seq_len: Optional[int] = None
    option_prefix: str = " "
    # Keep model in eval mode during adaptation to avoid train-only kernel paths
    # that can be incompatible on some cluster torch builds (e.g., gpt-oss float8 paths).
    force_train_mode: bool = False


class BSTTTLoRATrainer:
    def __init__(
        self,
        *,
        wrapper: HFCausalLMWrapper,
        cfg: BSTTTLoRAConfig,
        seed: int = 42,
    ) -> None:
        self.wrapper = wrapper
        self.cfg = cfg
        self.seed = seed

        # Import peft lazily so `eval_simpletom.py` can be imported without deps at module import time.
        from peft import LoraConfig, get_peft_model  # type: ignore

        base_model = self.wrapper.model

        target_modules = self._infer_lora_target_modules(base_model)
        layers_to_transform = self._infer_lora_layers(base_model, top_fraction=self.cfg.lora_top_fraction)

        lora_cfg = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=self.cfg.lora_alpha,
            lora_dropout=self.cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
            layers_to_transform=layers_to_transform,
            inference_mode=False,
        )

        # Replace underlying model with LoRA-wrapped model so wrapper scoring uses LoRA params.
        peft_model = get_peft_model(base_model, lora_cfg)
        self.wrapper.model = peft_model

        # Snapshot initial LoRA parameter values for episode resets.
        self._init_lora_params: Dict[str, torch.Tensor] = {}
        for name, p in self.wrapper.model.named_parameters():
            if p.requires_grad:
                self._init_lora_params[name] = p.detach().cpu().clone()

        # Optimizer over trainable (LoRA) parameters.
        params = [p for p in self.wrapper.model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found after LoRA wrapping.")
        self.optim = torch.optim.AdamW(params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        # Make sure base is in eval mode by default; only LoRA params will be updated.
        self.wrapper.model.eval()

    def _infer_lora_target_modules(self, model: Any) -> List[str]:
        # Prefer common Linear submodule suffixes across Llama/Qwen-style decoders.
        candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"]
        found: List[str] = []
        for cand in candidates:
            for name, _ in model.named_modules():
                if name.endswith(cand):
                    found.append(cand)
                    break
        # Deduplicate while preserving order.
        seen = set()
        out: List[str] = []
        for x in found:
            if x not in seen:
                seen.add(x)
                out.append(x)
        if not out:
            # Fallback: let peft error with a clearer message than an empty list.
            out = ["q_proj"]
        return out

    def _infer_lora_layers(self, model: Any, *, top_fraction: float) -> Optional[List[int]]:
        n = getattr(model.config, "num_hidden_layers", None)
        if n is None:
            n = getattr(model.config, "n_layer", None)
        if n is None:
            return None
        if n <= 0:
            return None
        k = int(n * top_fraction)
        if k <= 0:
            k = 1
        start = n - k
        return list(range(start, n))

    def reset_fast_weights(self) -> None:
        """Reset only trainable (LoRA) params to their initial values."""
        for name, p in self.wrapper.model.named_parameters():
            if not p.requires_grad:
                continue
            init = self._init_lora_params.get(name)
            if init is None:
                continue
            p.data.copy_(init.to(p.device))

        # Also reset optimizer state (e.g., Adam moments) to isolate episodes.
        # This avoids momentum carryover across episodes in BSTTT evaluation.
        try:
            self.optim.state.clear()  # type: ignore[attr-defined]
        except Exception:
            pass

    def _build_support_batches(
        self,
        support_examples: Sequence[SimpleToMExample],
        *,
        support_prompt_style: str = "vanilla",
        support_ms_hint: Optional[str] = None,
    ) -> Tuple[List[str], List[List[str]], torch.Tensor]:
        prompts: List[str] = []
        options: List[List[str]] = []
        targets: List[int] = []
        for ex in support_examples:
            # Strip existing letter prefixes (e.g., "a. " -> "") if present
            # This avoids double-lettering when build_vanilla_prompt adds A./B./C.
            clean_choices = []
            for c in ex.choices:
                if len(c) >= 3 and c[1] == '.' and c[0].isalpha():
                    clean_choices.append(c[3:].strip())
                else:
                    clean_choices.append(c)
            
            # Strip the answer's letter prefix too
            answer_clean = ex.answer
            if len(answer_clean) >= 3 and answer_clean[1] == '.' and answer_clean[0].isalpha():
                answer_clean = answer_clean[3:].strip()
            
            if support_prompt_style == "ms_reminder":
                ms_hint = support_ms_hint if support_ms_hint is not None else "unknown"
                prompts.append(build_ms_reminder_prompt(ex.story, ex.question, clean_choices, ms_hint))
            else:
                prompts.append(build_vanilla_prompt(ex.story, ex.question, clean_choices))
            options.append(clean_choices)
            try:
                targets.append(clean_choices.index(answer_clean))
            except ValueError as e:
                raise ValueError(f"Support answer not in choices: answer={answer_clean!r} choices={clean_choices}") from e
        target_idx = torch.tensor(targets, device=self.wrapper.model.device, dtype=torch.long)
        return prompts, options, target_idx

    def _action_reconstruction_loss(
        self,
        support_scores: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        # support_scores: [B, C] log p(option | prompt)
        log_probs = torch.log_softmax(support_scores, dim=-1)
        n = support_scores.shape[0]
        idx = torch.arange(n, device=support_scores.device, dtype=torch.long)
        loss = -log_probs[idx, target_idx].mean()
        return loss

    def _is_awareness_choice(self, answer_choice: str) -> bool:
        """
        Map mental-state answer text into an 'aware?' boolean.

        SimpleToM uses (typically) Yes/No for mental-state choice.
        """
        t = str(answer_choice).strip().lower()
        if t in ("yes", "aware", "aware."):
            return True
        if t in ("no", "unaware", "unaware."):
            return False
        # Fallback: treat unknown answers as "unaware".
        return False

    def _predict_mental_state_choice(
        self,
        *,
        mental_state_example: SimpleToMExample,
    ) -> str:
        """
        Predict the answer choice text for the scenario's mental-state question.
        """
        prompt = build_vanilla_prompt(mental_state_example.story, mental_state_example.question, mental_state_example.choices)
        options = [list(mental_state_example.choices)]
        scores = self.wrapper.score_options(
            prompts=[prompt],
            options=options,
            batch_size=1,
            max_seq_len=self.cfg.max_seq_len,
            option_prefix=self.cfg.option_prefix,
        )[0]
        return mental_state_example.choices[scores.pred_index]

    def _predict_awareness_from_mental_state(
        self,
        *,
        mental_state_example: SimpleToMExample,
    ) -> bool:
        """Predict awareness using option scoring for the mental-state example."""
        pred_choice = self._predict_mental_state_choice(mental_state_example=mental_state_example)
        return self._is_awareness_choice(pred_choice)

    def _temporal_smoothness_loss(
        self,
        hidden_states_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Milestone 7: L_sm = sum || z_{t+1} - z_t ||^2
        Encourages representations to be smooth over time.
        """
        if len(hidden_states_list) < 2:
            return torch.tensor(0.0, device=self.wrapper.model.device)
        
        losses = []
        for i in range(len(hidden_states_list) - 1):
            h_t = hidden_states_list[i]
            h_next = hidden_states_list[i+1]
            # Use MSE between the latent representation of consecutive turns
            losses.append(torch.mean((h_next - h_t)**2))
        
        return torch.stack(losses).mean()

    @torch.enable_grad()
    def adapt_on_support(
        self,
        *,
        support_examples: Sequence[SimpleToMExample],
        mental_state_example: Optional[SimpleToMExample] = None,
        bsttt_loss: str = "action_reconstruction",
        support_prompt_style: str = "vanilla",
        support_ms_hint: Optional[str] = None,
    ) -> Tuple[List[float], List[float], List[float], bool, Optional[bool], Optional[bool]]:
        if self.cfg.force_train_mode:
            self.wrapper.model.train()
        else:
            self.wrapper.model.eval()

        if bsttt_loss != "next_token_loss":
            support_prompts, support_options, target_idx = self._build_support_batches(
                support_examples,
                support_prompt_style=support_prompt_style,
                support_ms_hint=support_ms_hint,
            )
        else:
            support_prompts, support_options, target_idx = [], [], None
        loss_total_curve: List[float] = []
        loss_action_curve: List[float] = []
        loss_cons_curve: List[float] = []

        consistency_used = bool(bsttt_loss == "action_reconstruction_plus_consistency" and mental_state_example is not None)
        predicted_awareness: Optional[bool] = None
        gold_awareness: Optional[bool] = None

        if consistency_used:
            gold_awareness = self._is_awareness_choice(mental_state_example.answer)
            with torch.no_grad():
                predicted_awareness = self._predict_awareness_from_mental_state(
                    mental_state_example=mental_state_example,
                )

        for _ in range(self.cfg.adapt_steps):
            self.optim.zero_grad(set_to_none=True)
            
            if bsttt_loss == "next_token_loss":
                # Generic TTT: next-token loss on support stories.
                # We use the full story text as the target.
                stories = [ex.story for ex in support_examples]
                inputs = self.wrapper.tokenizer(stories, return_tensors="pt", padding=True, truncation=True).to(self.wrapper.model.device)
                
                labels = inputs["input_ids"].clone()
                labels[inputs["attention_mask"] == 0] = -100
                
                # We need hidden states for temporal loss
                outputs = self.wrapper.model(**inputs, labels=labels, output_hidden_states=True, use_cache=False)
                loss_main = outputs.loss
                
                if self.cfg.enable_temporal_smoothness:
                    # Collect the last hidden states for each scenario in the batch
                    # Hidden states shape: [num_layers, batch, seq, hidden]
                    last_h = outputs.hidden_states[-1] # [B, S, H]
                    
                    # Compute scenario-level representations (e.g. mean or last token)
                    # For simplicity, we'll use the mean over the sequence
                    rep_list = [last_h[bi].mean(dim=0) for bi in range(len(stories))]
                    
                    loss_sm = self._temporal_smoothness_loss(rep_list)
                    loss_total = loss_main + float(self.cfg.temporal_smoothness_weight) * loss_sm
                else:
                    loss_total = loss_main
            else:
                # BSTTT: ToM-aligned action reconstruction.
                # Score each example separately to handle variable-length option lists.
                # DynToM questions can have 7 or 8 options, so batching fails.
                per_example_losses = []
                for prompt_i, opts_i, tgt_i in zip(support_prompts, support_options, target_idx):
                    scores_i = self.wrapper.score_options_tensor(
                        prompts=[prompt_i],
                        options=[opts_i],
                        batch_size=1,
                        max_seq_len=self.cfg.max_seq_len,
                        option_prefix=self.cfg.option_prefix,
                    )  # [1, num_options]
                    tgt_i_single = tgt_i.unsqueeze(0)  # [1]
                    per_example_losses.append(self._action_reconstruction_loss(scores_i, tgt_i_single))
                loss_main = torch.stack(per_example_losses).mean()
                loss_total = loss_main

            if consistency_used and predicted_awareness is not None and mental_state_example is not None:
                # Consistency loss depends on scores; if we didn't compute scores for next_token_loss, we shouldn't reach here.
                # But we ensure scores are computed for consistency if needed.
                if bsttt_loss == "next_token_loss":
                    # This case is currently not enabled by the CLI's choices, but for completeness:
                    scores = self.wrapper.score_options_tensor(
                        prompts=support_prompts,
                        options=support_options,
                        batch_size=min(4, len(support_prompts)),
                        max_seq_len=self.cfg.max_seq_len,
                        option_prefix=self.cfg.option_prefix,
                    )
                loss_cons = self._belief_action_consistency_loss(
                    support_scores=scores,
                    support_examples=support_examples,
                    predicted_awareness=predicted_awareness,
                    mental_state_example=mental_state_example,
                )
                loss_total = loss_main + float(self.cfg.consistency_weight) * loss_cons
                loss_cons_curve.append(float(loss_cons.detach().cpu().item()))

            loss_action_curve.append(float(loss_main.detach().cpu().item()))
            loss_total.backward()
            self.optim.step()
            loss_total_curve.append(float(loss_total.detach().cpu().item()))

        self.wrapper.model.eval()
        return (
            loss_total_curve,
            loss_action_curve,
            loss_cons_curve,
            consistency_used,
            predicted_awareness,
            gold_awareness,
        )

    @torch.no_grad()
    def predict_query_option(
        self,
        *,
        query_example: SimpleToMExample,
        query_prompt_style: str = "vanilla",
        query_ms_hint: Optional[str] = None,
    ) -> Tuple[str, str, int, List[float]]:
        if query_prompt_style == "ms_reminder":
            ms_hint = query_ms_hint if query_ms_hint is not None else "unknown"
            prompt = build_ms_reminder_prompt(query_example.story, query_example.question, query_example.choices, ms_hint)
        else:
            prompt = build_vanilla_prompt(query_example.story, query_example.question, query_example.choices)
        prompts = [prompt]
        options = [list(query_example.choices)]
        mc_out = self.wrapper.score_options(
            prompts=prompts,
            options=options,
            batch_size=1,
            max_seq_len=self.cfg.max_seq_len,
            option_prefix=self.cfg.option_prefix,
        )[0]
        pred_choice = query_example.choices[mc_out.pred_index]
        return prompt, pred_choice, mc_out.pred_index, mc_out.option_scores

    def run_episode(
        self,
        *,
        episode_id: str,
        episode_scenario_name: str,
        query_example: SimpleToMExample,
        support_examples: Sequence[SimpleToMExample],
        query_task: str,
        mental_state_example: Optional[SimpleToMExample] = None,
        bsttt_loss: str = "action_reconstruction",
        query_prompt_style: str = "vanilla",
        support_prompt_style: str = "vanilla",
    ) -> SimpleToMEpisodeRow:
        """
        Run:
          - reset to initial LoRA weights
          - predict query before adaptation
          - adapt on support with action reconstruction
          - predict query after adaptation
        """
        self.reset_fast_weights()

        query_ms_hint_before: Optional[str] = None
        if (
            query_prompt_style == "ms_reminder"
            and query_example.question_type != "mental_state"
            and mental_state_example is not None
        ):
            query_ms_hint_before = self._predict_mental_state_choice(mental_state_example=mental_state_example)

        prompt, pred_before, pred_idx_before, _scores_before = self.predict_query_option(
            query_example=query_example,
            query_prompt_style=query_prompt_style,
            query_ms_hint=query_ms_hint_before,
        )
        # Store support loss + support predictions for debugging.
        # (Support predictions before adaptation can help diagnose objective alignment issues.)
        support_pred_before: List[str] = []
        with torch.no_grad():
            support_prompts, support_options, _target_idx = self._build_support_batches(
                support_examples,
                support_prompt_style=support_prompt_style,
                support_ms_hint=query_ms_hint_before,
            )
            support_mcs = self.wrapper.score_options(
                prompts=support_prompts,
                options=support_options,
                batch_size=min(4, len(support_prompts)),
                max_seq_len=self.cfg.max_seq_len,
                option_prefix=self.cfg.option_prefix,
            )
            for ex, mc in zip(support_examples, support_mcs):
                support_pred_before.append(ex.choices[mc.pred_index])

        # Adapt on support (action-only, or action+belief consistency depending on config).
        loss_curve = self.adapt_on_support(
            support_examples=support_examples,
            mental_state_example=mental_state_example,
            bsttt_loss=bsttt_loss,
            support_prompt_style=support_prompt_style,
            support_ms_hint=query_ms_hint_before,
        )

        query_ms_hint_after: Optional[str] = query_ms_hint_before
        if (
            query_prompt_style == "ms_reminder"
            and query_example.question_type != "mental_state"
            and mental_state_example is not None
        ):
            query_ms_hint_after = self._predict_mental_state_choice(mental_state_example=mental_state_example)

        _prompt2, pred_after, pred_idx_after, option_scores_after = self.predict_query_option(
            query_example=query_example,
            query_prompt_style=query_prompt_style,
            query_ms_hint=query_ms_hint_after,
        )
        correct_before = pred_before == query_example.answer
        correct_after = pred_after == query_example.answer

        return {
            "episode_id": episode_id,
            "scenario_name": episode_scenario_name,
            "query_task": query_task,
            "query_id": query_example.id,
            "question": query_example.question,
            "choices": query_example.choices,
            "answer": query_example.answer,
            # For prompt inspection + qualitative printing:
            # treat post-adaptation as the main prediction.
            "pred_choice": pred_after,
            "pred_index": pred_idx_after,
            "option_scores": option_scores_after,
            "pred_choice_before": pred_before,
            "pred_index_before": pred_idx_before,
            "correct_before": correct_before,
            "pred_choice_after": pred_after,
            "pred_index_after": pred_idx_after,
            "prompt": prompt,
            "correct_after": correct_after,
            "correct": correct_after,
            "support_pred_before": support_pred_before,
            "support_loss_curve": loss_curve[0],
            "support_action_loss_curve": loss_curve[1],
            "support_consistency_loss_curve": loss_curve[2],
            "consistency_used": loss_curve[3],
            "predicted_awareness": loss_curve[4],
            "gold_awareness": loss_curve[5],
        }
