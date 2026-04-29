"""
Hugging Face causal LM wrapper for multiple-choice scoring.

Goal: score each answer option by log-likelihood under the model given a shared prompt.
This wrapper is intentionally generic so we can swap base models later.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _patch_missing_torch_float8_dtype() -> None:
    """
    Compatibility shim for cluster images where torch lacks newer float8 symbols.

    Some optional model/kernel paths reference `torch.float8_e8m0fnu` directly.
    On older torch builds this attribute is absent and raises at runtime.
    """
    if hasattr(torch, "float8_e8m0fnu"):
        return
    fallback = getattr(torch, "float8_e4m3fn", None) or getattr(torch, "bfloat16", None)
    if fallback is None:
        return
    try:
        setattr(torch, "float8_e8m0fnu", fallback)
    except Exception:
        # Best effort only; if torch forbids setattr we continue without patching.
        pass


_patch_missing_torch_float8_dtype()


@dataclass(frozen=True)
class MCScoringOutput:
    option_scores: List[float]
    pred_index: int


class HFCausalLMWrapper:
    """
    Generic wrapper around a HF causal LM for multiple-choice scoring.

    Scoring approach:
      score(option) = log p(option_tokens | prompt_tokens)
    where option_tokens are appended to a shared prompt.
    """

    def __init__(
        self,
        *,
        model_name_or_path: str,
        trust_remote_code: bool = False,
        dtype: Optional[str] = "bfloat16",
        device_map: Union[str, Dict[str, int], None] = "auto",
        max_memory: Optional[Dict[Union[int, str], str]] = None,
        use_fast_tokenizer: bool = True,
        use_safetensors: bool = True,
        hf_token: Optional[str] = None,
        force_chat_template: Optional[bool] = None,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self._force_chat_template = force_chat_template
        token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        auth_kwargs: Dict[str, Any] = {}
        if token:
            auth_kwargs["token"] = token
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=use_fast_tokenizer,
            trust_remote_code=trust_remote_code,
            **auth_kwargs,
        )

        # Ensure we can pad in batching.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch_dtype = None
        if dtype is not None:
            torch_dtype = getattr(torch, dtype, None)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
            **auth_kwargs,
        )
        self.model.eval()
        self._use_chat_template_for_prompts = self._should_use_chat_template()

    def _should_use_chat_template(self) -> bool:
        if self._force_chat_template is not None:
            return bool(self._force_chat_template)
        model_name = (self.model_name_or_path or "").lower()
        if "gpt-oss" not in model_name:
            return False
        return hasattr(self.tokenizer, "apply_chat_template")

    def _format_prompt(
        self,
        prompt: str,
        *,
        for_generation: bool,
    ) -> str:
        if not self._use_chat_template_for_prompts:
            return prompt
        if not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt
        messages = [{"role": "user", "content": prompt}]
        try:
            templated = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=for_generation,
            )
            if isinstance(templated, str) and templated.strip():
                return templated
        except Exception:
            return prompt
        return prompt

    @torch.inference_mode()
    def score_options(
        self,
        *,
        prompts: Sequence[str],
        options: Sequence[Sequence[str]],
        option_prefix: str = " ",
        batch_size: int = 4,
        max_seq_len: Optional[int] = None,
        truncate_from_left: bool = True,
    ) -> List[MCScoringOutput]:
        """
        Score multiple-choice options for each example.

        Args:
          prompts: length B, shared prompt per example.
          options: length B, each an option list (len C_i usually constant).
          option_prefix: string prepended to each option before tokenization (default: leading space).
          batch_size: micro-batch size across examples.
        """
        if len(prompts) != len(options):
            raise ValueError(f"prompts and options must have same length. Got {len(prompts)} vs {len(options)}")
        outputs: List[MCScoringOutput] = []

        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            batch_options = options[start:end]

            # Tokenize prompts once per example (shared across options).
            prompt_ids_list: List[List[int]] = []
            prompt_lens: List[int] = []
            for p in batch_prompts:
                prompt_text = self._format_prompt(p, for_generation=True)
                ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
                prompt_ids_list.append(ids)
                prompt_lens.append(len(ids))

            # We score per-option index. This keeps the masking logic simple and is fast enough for prototype sizes.
            num_options = max(len(opts) for opts in batch_options)

            # Initialize scores.
            batch_scores: List[List[float]] = []
            for opts in batch_options:
                if not opts:
                    raise ValueError("Empty options list encountered.")
                batch_scores.append([float("-inf")] * len(opts))

            for opt_i in range(num_options):
                # Collect sequences that have this option index.
                active_indices: List[int] = []
                option_token_ids_list: List[List[int]] = []

                for bi, opts in enumerate(batch_options):
                    if opt_i >= len(opts):
                        continue
                    active_indices.append(bi)
                    opt_text = options[start + bi][opt_i]
                    opt_ids = self.tokenizer(option_prefix + opt_text, add_special_tokens=False).input_ids
                    option_token_ids_list.append(opt_ids)

                if not active_indices:
                    continue

                # Build padded input_ids.
                input_ids_list: List[List[int]] = []
                attention_mask_list: List[List[int]] = []
                full_lens: List[int] = []
                prompt_lens_eff: List[int] = []
                for idx_in_batch, opt_ids in zip(active_indices, option_token_ids_list):
                    p_ids = prompt_ids_list[idx_in_batch]
                    full_ids = p_ids + opt_ids
                    if max_seq_len is not None and len(full_ids) > max_seq_len:
                        if len(opt_ids) >= max_seq_len:
                            raise ValueError(
                                f"Option too long for max_seq_len. opt_len={len(opt_ids)} max_seq_len={max_seq_len}"
                            )
                        if not truncate_from_left:
                            raise NotImplementedError("truncate_from_left=False not implemented")
                        # Keep option intact at the end; keep only the tail of the prompt.
                        full_ids = full_ids[-max_seq_len:]
                        p_len_eff = max_seq_len - len(opt_ids)
                    else:
                        p_len_eff = len(p_ids)
                    input_ids_list.append(full_ids)
                    attention_mask_list.append([1] * len(full_ids))
                    full_lens.append(len(full_ids))
                    prompt_lens_eff.append(p_len_eff)

                max_len = max(full_lens)
                pad_id = self.tokenizer.pad_token_id

                padded_input_ids = []
                padded_attention_mask = []
                for ids, attn in zip(input_ids_list, attention_mask_list):
                    pad_len = max_len - len(ids)
                    padded_input_ids.append(ids + [pad_id] * pad_len)
                    padded_attention_mask.append(attn + [0] * pad_len)

                input_ids_tensor = torch.tensor(padded_input_ids, device=self.model.device)
                attention_mask_tensor = torch.tensor(padded_attention_mask, device=self.model.device)

                logits = self.model(input_ids_tensor, attention_mask=attention_mask_tensor).logits

                # token_logprobs[t] = log p(token at position t+1 | previous tokens)
                log_probs = torch.log_softmax(logits, dim=-1)
                target_ids = input_ids_tensor[:, 1:]
                # [B, L-1]
                token_logprobs = torch.gather(
                    log_probs[:, :-1, :],
                    dim=2,
                    index=target_ids.unsqueeze(-1),
                ).squeeze(-1)

                # Sum logprobs over option token span for each active sequence.
                for seq_i, bi in enumerate(active_indices):
                    p_len_eff = prompt_lens_eff[seq_i]
                    opt_ids = option_token_ids_list[seq_i]
                    opt_len = len(opt_ids)
                    # Option tokens occupy original positions [p_len, p_len+opt_len-1]
                    # token_logprobs indices correspond to original positions shifted by -1.
                    start_t = p_len_eff - 1
                    if start_t < 0:
                        raise ValueError(
                            f"Prompt too short after truncation (prompt_len_eff={p_len_eff}). "
                            "Increase max_seq_len or ensure prompt ends with a token before options."
                        )
                    end_t = start_t + opt_len

                    # Clamp to avoid odd edge-cases.
                    end_t = min(end_t, token_logprobs.shape[1])
                    score = token_logprobs[seq_i, start_t:end_t].sum().item()
                    batch_scores[bi][opt_i] = score

            # Select prediction.
            for bi in range(end - start):
                scores = batch_scores[bi]
                pred_index = int(torch.tensor(scores).argmax().item())
                outputs.append(MCScoringOutput(option_scores=scores, pred_index=pred_index))

        return outputs

    def score_options_tensor(
        self,
        *,
        prompts: Sequence[str],
        options: Sequence[Sequence[str]],
        option_prefix: str = " ",
        batch_size: int = 4,
        max_seq_len: Optional[int] = None,
        truncate_from_left: bool = True,
    ) -> torch.Tensor:
        """
        Differentiable multiple-choice scoring.

        Returns:
          scores: Tensor of shape [B, C] where C is the number of options (assumes constant C).
                  score[i, j] = log p(option_j_tokens | prompt_i_tokens).
        """
        if len(prompts) != len(options):
            raise ValueError(f"prompts and options must have same length. Got {len(prompts)} vs {len(options)}")
        if len(prompts) == 0:
            return torch.empty((0, 0), device=self.model.device)
        if any(len(opts) == 0 for opts in options):
            raise ValueError("Empty options list encountered.")

        # For BSTTT V1, SimpleToM is 2-way. We keep the implementation strict to reduce masking complexity.
        c0 = len(options[0])
        if any(len(opts) != c0 for opts in options):
            raise ValueError("score_options_tensor currently requires constant number of options per example.")

        all_chunks: List[torch.Tensor] = []
        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            batch_options = options[start:end]
            bsz = len(batch_prompts)

            prompt_ids_list: List[List[int]] = []
            prompt_lens: List[int] = []
            for p in batch_prompts:
                prompt_text = self._format_prompt(p, for_generation=True)
                ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids
                prompt_ids_list.append(ids)
                prompt_lens.append(len(ids))

            # Compute each option column separately to keep the masking logic simple.
            opt_cols: List[torch.Tensor] = []
            for opt_i in range(c0):
                input_ids_list: List[List[int]] = []
                attention_mask_list: List[List[int]] = []
                prompt_lens_eff: List[int] = []
                opt_lens: List[int] = []
                opt_ids_list: List[List[int]] = []

                for bi in range(bsz):
                    p_ids = prompt_ids_list[bi]
                    opt_text = batch_options[bi][opt_i]
                    opt_ids = self.tokenizer(option_prefix + opt_text, add_special_tokens=False).input_ids

                    full_ids = p_ids + opt_ids
                    if max_seq_len is not None and len(full_ids) > max_seq_len:
                        if len(opt_ids) >= max_seq_len:
                            raise ValueError(
                                f"Option too long for max_seq_len. opt_len={len(opt_ids)} max_seq_len={max_seq_len}"
                            )
                        if not truncate_from_left:
                            raise NotImplementedError("truncate_from_left=False not implemented")
                        full_ids = full_ids[-max_seq_len:]
                        p_len_eff = max_seq_len - len(opt_ids)
                    else:
                        p_len_eff = len(p_ids)

                    input_ids_list.append(full_ids)
                    attention_mask_list.append([1] * len(full_ids))
                    prompt_lens_eff.append(p_len_eff)
                    opt_lens.append(len(opt_ids))
                    opt_ids_list.append(opt_ids)

                max_len = max(len(ids) for ids in input_ids_list)
                pad_id = self.tokenizer.pad_token_id

                padded_input_ids: List[List[int]] = []
                padded_attention_mask: List[List[int]] = []
                for ids, attn in zip(input_ids_list, attention_mask_list):
                    pad_len = max_len - len(ids)
                    padded_input_ids.append(ids + [pad_id] * pad_len)
                    padded_attention_mask.append(attn + [0] * pad_len)

                input_ids_tensor = torch.tensor(padded_input_ids, device=self.model.device)
                attention_mask_tensor = torch.tensor(padded_attention_mask, device=self.model.device)

                logits = self.model(input_ids_tensor, attention_mask=attention_mask_tensor).logits
                # Use fp32 for stable log-softmax + CE.
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                target_ids = input_ids_tensor[:, 1:]
                token_logprobs = torch.gather(
                    log_probs[:, :-1, :],
                    dim=2,
                    index=target_ids.unsqueeze(-1),
                ).squeeze(-1)  # [B, L-1]

                scores_per_example: List[torch.Tensor] = []
                for seq_i in range(bsz):
                    p_len_eff = prompt_lens_eff[seq_i]
                    opt_len = opt_lens[seq_i]
                    start_t = p_len_eff - 1
                    if start_t < 0:
                        raise ValueError(
                            f"Prompt too short after truncation (prompt_len_eff={p_len_eff}). "
                            "Ensure prompt ends with a token before options."
                        )
                    end_t = min(start_t + opt_len, token_logprobs.shape[1])
                    scores_per_example.append(token_logprobs[seq_i, start_t:end_t].sum())

                opt_cols.append(torch.stack(scores_per_example, dim=0))  # [B]

            scores_chunk = torch.stack(opt_cols, dim=1)  # [B, C]
            all_chunks.append(scores_chunk)

        return torch.cat(all_chunks, dim=0)

    @torch.inference_mode()
    def generate(
        self,
        *,
        prompts: Sequence[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        do_sample: bool = False,
        top_p: Optional[float] = None,
        batch_size: int = 4,
        max_seq_len: Optional[int] = None,
        truncate_from_left: bool = True,
    ) -> List[str]:
        """
        Generate text from prompts (e.g. for CoT-style free-form answers).

        Args:
          prompts: length N, input prompts.
          max_new_tokens: maximum tokens to generate per prompt.
          temperature: sampling temperature (0 => greedy).
          do_sample: if True, sample; else greedy decode.
          top_p: nucleus sampling threshold (optional).
          batch_size: micro-batch size across prompts.
        """
        outputs: List[str] = []
        gen_kw = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature if do_sample else 1.0,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if top_p is not None:
            gen_kw["top_p"] = top_p

        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]

            input_ids_list: List[List[int]] = []
            for p in batch_prompts:
                prompt_text = self._format_prompt(p, for_generation=True)
                ids = self.tokenizer(prompt_text, add_special_tokens=True, return_tensors=None).input_ids
                if max_seq_len is not None and len(ids) > max_seq_len:
                    if truncate_from_left:
                        ids = ids[-max_seq_len:]
                    else:
                        ids = ids[:max_seq_len]
                input_ids_list.append(ids)

            max_len = max(len(ids) for ids in input_ids_list)
            pad_id = self.tokenizer.pad_token_id
            padded = [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids_list]
            input_ids_tensor = torch.tensor(padded, device=self.model.device)

            out_ids = self.model.generate(input_ids_tensor, **gen_kw)

            for i, (full_ids, inp_ids) in enumerate(zip(out_ids, input_ids_list)):
                new_part = full_ids[len(inp_ids):]
                text = self.tokenizer.decode(new_part, skip_special_tokens=True)
                outputs.append(text.strip())

        return outputs
