import logging
import time
from typing import Any, Callable, Iterable, List, MutableMapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def pad_input_ids(
    input_ids_list: List[torch.Tensor],
    min_pad_length: int = 0,
    is_causal_mask=True,
) -> Tuple[torch.Tensor, MutableMapping[str, Any]]:
    """
    Convert a list of Tensors to a rectangular tensor. Return extra padding kwargs for the position_ids and mask, since
    this will be required to properly handle the rectangular tensor for certain models.

    Parameters
    ----------
    input_ids_list: List[torch.Tensor]
        a list of Tensors of varied length
    min_pad_length: int
        pad to a min length provided. If the min_pad_length is less than the largest input_ids in the input_ids_list,
        padding will be determined based on the largest length input_ids.

    Returns
    -------
    Tuple[torch.Tensor, MutableMapping[str, Any]]
        A rectangular 2d padded tensor and a mapping containing the mask and position_ids typically used in forward pass
        in fms models
        A mapping from mask to a 3d causal mask and from position_ids to a 2d rectangular position_ids tensor
    """
    max_len = max([min_pad_length] + [seq.size(0) for seq in input_ids_list])

    padded_input_ids_list = []
    mask_list = []
    position_ids_list = []
    for input_ids_i in input_ids_list:
        seq_len = input_ids_i.size(0)
        pads = torch.zeros(
            max_len - seq_len, dtype=torch.long, device=input_ids_i.device
        )
        non_pads = torch.ones(seq_len, dtype=torch.bool, device=input_ids_i.device)

        # Setting this to 0, however if 0 is the eos, we will end up truncating the output if using truncate_after_eos
        # once this workflow works for nested tensor, this can probably be removed
        padded_input_ids_list.append(torch.cat((pads, input_ids_i)))

        # computing this as it's lightweight but could potentially be skipped
        mask_list.append(torch.cat((pads.bool(), non_pads)))

        pos_ids_pads = pads
        pos_ids_seq = torch.arange(
            0, seq_len, dtype=torch.long, device=input_ids_i.device
        )
        position_ids_list.append(torch.cat((pos_ids_pads, pos_ids_seq)))

    input_ids = torch.stack(padded_input_ids_list)
    padding_kwargs = {}
    mask = torch.stack(mask_list)
    mask = mask.unsqueeze(-1) == mask.unsqueeze(-2)
    # this is a causal mask for generation
    if is_causal_mask:
        mask = mask.tril()
    mask = torch.where(mask.logical_not(), -torch.inf, 0.0)
    padding_kwargs["mask"] = mask

    position_ids = torch.stack(position_ids_list)
    padding_kwargs["position_ids"] = position_ids

    return input_ids, padding_kwargs


def __update_padding_kwargs(
    use_cache: bool, model_specific_kwargs: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """Generic function to prepare any model specific keyword arguments"""
    # extend the attention mask
    mask = model_specific_kwargs.get("mask", None)
    if mask is not None:
        # get the last row of the 3d mask
        mask = mask[:, -1:, :]
        # extend the mask one slot
        mask = torch.cat(
            (
                mask,
                torch.zeros(mask.size(0), 1, 1, device=mask.device),
            ),
            dim=2,
        )
        model_specific_kwargs["mask"] = mask
        if torch._dynamo.config.dynamic_shapes:
            torch._dynamo.mark_dynamic(mask, 2)

    # extend the position_ids
    position_ids = model_specific_kwargs.get("position_ids", None)
    if position_ids is not None:
        if use_cache:
            position_ids = position_ids[:, -1:] + 1
        else:
            position_ids = torch.cat(
                (position_ids, position_ids[:, -1:] + 1),
                dim=1,
            )
        model_specific_kwargs["position_ids"] = position_ids
    return model_specific_kwargs


def _make_cache_contiguous(
    past_key_value_states: List[List[torch.Tensor]],
) -> List[List[torch.Tensor]]:
    # kv updates are required for torch.compile with
    # mode='reduce-overhead'
    n_kv_s: List[List[torch.Tensor]] = []
    for layer_idx in range(len(past_key_value_states)):
        n_kv_s.append([])
        for tensor_idx in range(len(past_key_value_states[layer_idx])):
            n_kv_s[layer_idx].append(
                past_key_value_states[layer_idx][tensor_idx]
                .clone(memory_format=torch.contiguous_format)
                .detach()
            )
    return n_kv_s


def _make_cache_dynamic(
    past_key_value_states: List[List[torch.Tensor]],
) -> List[List[torch.Tensor]]:
    # kv updates are required for torch.compile with
    # mode='reduce-overhead'
    for layer in past_key_value_states:
        if isinstance(layer, Iterable):
            for tensor in layer:
                torch._dynamo.mark_dynamic(tensor, 2)
    return past_key_value_states


def generate(
    model: Union[torch.nn.Module, Callable],
    input_ids: torch.Tensor,
    max_seq_len: int = 4096,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 10,
    do_sample: bool = True,
    num_beams: int = 1,
    use_cache: bool = False,
    contiguous_cache: bool = False,
    eos_token_id: Optional[int] = None,
    timing: str = "per-token",
    post_iteration_hook: Optional[
        Callable[
            [int, torch.Tensor, torch.Tensor, MutableMapping[str, Any]],
            Tuple[torch.Tensor, MutableMapping[str, Any]],
        ]
    ] = None,
    extra_kwargs: Optional[MutableMapping[str, Any]] = None,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, float, float],
]:
    """
    Same as before but:
      - if timing=="per-token", returns (ids, first_token_time, mean_inter_token_time)
      - forces generation of exactly max_new_tokens
    """
    if num_beams != 1:
        raise NotImplementedError("generate() does yet not support beam search")

    # set up batch & kwargs exactly as before
    kwargs: MutableMapping[str, Any] = {}
    if extra_kwargs is not None:
        kwargs.update(extra_kwargs)

    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
    else:
        raise TypeError("input_ids must be a Tensor")

    batch_size = input_ids.size(0)
    eos_found = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

    result = input_ids
    next_input = input_ids
    kwargs["past_key_value_states"] = None
    kwargs["use_cache"] = use_cache

    prompt_length = input_ids.shape[1]

    # Timing arrays
    first_token_time = None
    inter_times: List[float] = []

    # Pre-fill timing
    if timing == "per-token":
        torch.cuda.synchronize()
        start_time = time.time()

    for i in range(max_new_tokens):
        input_ids = next_input[:, -max_seq_len :]

        # update padding/mask/pos_ids exactly as before
        if i > 0:
            kwargs = __update_padding_kwargs(use_cache, kwargs)

        # model call
        if timing == "per-token":
            torch.cuda.synchronize()
            t0 = time.time()

        output = model(input_ids, **kwargs)

        if use_cache:
            logits, past = output
            kwargs["past_key_value_states"] = past
            if contiguous_cache:
                kwargs["past_key_value_states"] = _make_cache_contiguous(past)
            if torch._dynamo.config.dynamic_shapes:
                kwargs["past_key_value_states"] = _make_cache_dynamic(past)
        else:
            logits = output

        if timing == "per-token":
            torch.cuda.synchronize()
            t1 = time.time()
            delta = t1 - t0
            if i == 0:
                first_token_time = delta
            else:
                inter_times.append(delta)

        # sampling / greedy exactly as before
        if "only_last_token" not in kwargs:
            logits = logits[:, -1, :]
        if do_sample:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_val = torch.multinomial(probs, num_samples=1)
        else:
            next_val = torch.argmax(logits, dim=-1).unsqueeze(-1)

        if post_iteration_hook is not None:
            next_val, kwargs = post_iteration_hook(
                i + prompt_length, logits, next_val, kwargs
            )

        result = torch.cat([result, next_val], dim=-1)

        if eos_token_id is not None:
            eos_found |= (next_val.squeeze(-1) == eos_token_id)
            if eos_found.all():
                break

        next_input = next_val if use_cache else result

    # return
    if timing == "per-token":
        mean_inter = sum(inter_times) / len(inter_times)
        return result, first_token_time, mean_inter
    else:
        return result


def truncate_after_eos(
    result: torch.Tensor, eos_token_id: Union[int, "Any | None"]
) -> torch.Tensor:
    """
    Helper function to return a truncated sequence of token IDs stopping at
    (and including) the 'end of sentence' token.
    Currently only handles unbatched sequences.
    """
    if eos_token_id is None:
        return result
    eos_idx = torch.where(result == eos_token_id)
    eos_index = eos_idx[0]
    if eos_index.shape[0] >= 1:
        index = eos_index[0]
        result = result[: index + 1]
    return result


def trim_prefix(result: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    """
    Helper function to return a trimmed sequence of token IDs where
    all padding tokens (always 0 on our code) are removed.

    Examples:
    [0 0 0 0 1 2 3 4] with pad_token_id = 0 returns [1 2 3 4]
    [0 0 0 0 1 2 3 4] with pad_token_id = 5 returns [0 0 0 0 1 2 3 4]
    [1 2 3 4 0 1] with pad_token_id = 0 returns [1 2 3 4 0 1]

    Args:
    result: A 1D sequence of tokens
    pad_token_id: Token ID that will be trimmed from the start of the
        sequence
    """
    if result[0] != pad_token_id:
        return result
    output_diff = (result != pad_token_id).diff()
    first_real_token_idx = torch.where(output_diff > 0)
    if first_real_token_idx[0].numel() == 0:
        return result
    bos_index = first_real_token_idx[0][0]
    result = result[bos_index + 1 :]
    return result
