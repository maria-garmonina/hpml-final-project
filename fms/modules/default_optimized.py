from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fms.utils.activation import str_to_activation


def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int):
    """
    UNCHANGED
    Padding x tensor with `pad_size` on the seq_len dim (dim=1)
    Assumes that we only have tensors of either size 4 or 3
    """
    pad_shape = (
        (0, 0, 0, 0, 0, pad_size, 0, 0)
        if len(input_tensor.shape) == 4
        else (0, 0, 0, pad_size, 0, 0)
    )
    return torch.nn.functional.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(input_tensor, pad_size, chunk_size):
    """
    UNCHANGED
    Padding input_tensor with `pad_size` on the seq_len dim (dim=1) and
    simultaneously splitting it into chunk sequences.
    Assumes that we only have tensors of either size 4 or 3
    """
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)
    if len(input_tensor.shape) == 3:
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2]
        )
    else:
        return input_tensor.reshape(
            input_tensor.shape[0],
            -1,
            chunk_size,
            input_tensor.shape[2],
            input_tensor.shape[3],
        )


def segment_sum(input_tensor):
    """
    UNCHANGED
    More stable segment sum calculation. Uses cumulative sums and masking instead of direct subtractions.
    """
    chunk_size = input_tensor.size(-1)
    # expand to [...., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=-1)
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)
    mask = torch.tril(torch.ones(chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool), diagonal=0)
    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


class RMSNormGated(nn.Module):
    """
    UNCHANGED
    """
    def __init__(self, emb_dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(emb_dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)

        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return self.weight * hidden_states.to(input_dtype)


class SSMCacheUnit:
    """
    UNCHANGED
    """
    def __init__(
        self,
        emb_dim: int,
        nheads: int,
        head_dim: int,
        conv_kernel,
        expand: float,
        n_groups: int,
        state_size: int,
        batch_size: int,
        dtype: torch.dtype,
        device: Optional[str] = None,
    ):
        self.seqlen_offset = 0
        self.dtype = dtype
        self.conv_kernel_size = conv_kernel
        self.intermediate_size = int(expand * emb_dim)
        self.has_previous_state = False

        self.conv_state = torch.zeros(
            batch_size,
            self.intermediate_size + 2 * n_groups * state_size,
            self.conv_kernel_size,
            device=device,
            dtype=dtype,
        )
        self.ssm_state = torch.zeros(
            batch_size, nheads, head_dim, state_size, device=device, dtype=dtype
        )

    def update_conv_state(
        self, new_conv_state: torch.Tensor, cache_position: torch.Tensor
    ):
        conv_state = self.conv_state
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        conv_state[:, :, cache_position] = new_conv_state.to(conv_state.device)
        self.conv_state.zero_()
        self.conv_state += conv_state
        return self.conv_state


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    UNCHANGED
    Tunes out hidden states for padding tokens
    """
    if (
        attention_mask is not None
        and attention_mask.shape[1] > 1
        and attention_mask.shape[0] > 1
    ):
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * (attention_mask[:, -1, :, None] == 0)).to(
            dtype
        )
    return hidden_states


# optim_ssm.py
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fms.utils.activation import str_to_activation
from .helpers import (
    pad_tensor_by_size,
    reshape_into_chunks,
    segment_sum,
    apply_mask_to_padding_states,
    RMSNormGated,
    SSMCacheUnit,
)


class Default_Optimized_SSM(nn.Module):
    def __init__(
        self,
        nheads: int,
        emb_dim: int,
        state_size: int,
        conv_kernel: int,
        expand: float,
        use_bias: bool,
        use_conv_bias: bool,
        activation_fn: str,
        norm_eps: float,
        n_groups: int,
        head_dim: int,
        chunk_size: int,
    ):
        super().__init__()

        self.nheads            = nheads
        self.emb_dim           = emb_dim
        self.ssm_state_size    = state_size
        self.conv_kernel_size  = conv_kernel
        self.intermediate_size = int(expand * self.emb_dim)
        self.use_conv_bias     = use_conv_bias
        self.act               = str_to_activation(activation_fn)

        self.layer_norm_epsilon = norm_eps
        self.n_groups   = n_groups
        self.head_dim   = head_dim
        self.chunk_size = chunk_size

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d   = nn.Conv1d(
            in_channels  = self.conv_dim,
            out_channels = self.conv_dim,
            kernel_size  = conv_kernel,
            padding      = conv_kernel - 1,
            groups       = self.conv_dim,
            bias         = use_conv_bias,
        )

        projection_size = self.intermediate_size + self.conv_dim + self.nheads
        self.in_proj = nn.Linear(self.emb_dim, projection_size, bias=use_bias)

        self.dt_bias = nn.Parameter(torch.ones(self.nheads))
        A            = torch.arange(1, self.nheads + 1)
        self.A_log   = nn.Parameter(torch.log(A))
        self.D       = nn.Parameter(torch.ones(self.nheads))

        self.norm     = RMSNormGated(self.intermediate_size, eps=self.layer_norm_epsilon)
        self.out_proj = nn.Linear(self.intermediate_size, self.emb_dim, bias=use_bias)
        self.time_step_limit = (0.0, float("inf"))

        # pre‑compute triangular boolean masks
        excl = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool), -1)
        incl = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool),  0)
        self.register_buffer("mask_excl", excl[None, None, None])  # shape [1,1,1,L,L]
        self.register_buffer("mask_incl", incl[None, None, None])

    @torch.compile(backend="inductor", dynamic=False)
    def forward(
        self,
        input_states,
        mask,
        past_key_value_state: Optional[SSMCacheUnit] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        # 1. Gated‑MLP linear projection
        input_states = apply_mask_to_padding_states(input_states, mask)
        projected_states = self.in_proj(input_states)
        gate, hidden_states_B_C, dt = projected_states.split(
            [self.intermediate_size, self.conv_dim, self.nheads], dim=-1
        )

        # Decide whether we can reuse the cached convolution/SSM states
        use_precomputed_states = (
            past_key_value_state is not None
            and past_key_value_state.has_previous_state
            and seq_len == 1
            and past_key_value_state.conv_state.shape[0]
            == past_key_value_state.ssm_state.shape[0]
            == batch_size
            and cache_position is not None
        )

        # 2. Convolution sequence transformation
        if use_precomputed_states:
            # auto-regressive path
            past_key_value_state.conv_state = past_key_value_state.conv_state.roll(-1, dims=-1)
            past_key_value_state.conv_state[:, :, -1] = hidden_states_B_C[:, 0, :].to(
                past_key_value_state.conv_state.device
            )
            conv_states = past_key_value_state.conv_state.to(self.conv1d.weight.device)
            hidden_states_B_C = (conv_states * self.conv1d.weight.squeeze(1)).sum(-1)
            if self.use_conv_bias:
                hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
            hidden_states_B_C = self.act(hidden_states_B_C)
        else:
            # full‑sequence path
            hidden_states_B_C_t = hidden_states_B_C.transpose(1, 2)
            if past_key_value_state is not None:
                pad_len = self.conv_kernel_size - hidden_states_B_C_t.shape[-1]
                past_key_value_state.conv_state.copy_(F.pad(hidden_states_B_C_t, (pad_len, 0)))
            hidden_states_B_C = self.act(
                self.conv1d(hidden_states_B_C_t)[..., :seq_len].transpose(1, 2)
            )

        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, mask)

        # 3. SSM transform
        # Split into hidden / B / C and 
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size,
                self.n_groups * self.ssm_state_size,
                self.n_groups * self.ssm_state_size,
            ],
            dim=-1,
        )

        A = -torch.exp(self.A_log.float())  # shape [num_heads]

        # auto-regressive path (same as default)
        if use_precomputed_states:
            past_key_value_state.has_previous_state = True

            # discretize dt, A, B
            dt_single = F.softplus(dt[:, 0, :] + self.dt_bias).clamp(*self.time_step_limit)
            dt_single = dt_single[:, :, None].expand(batch_size, self.nheads, self.head_dim)

            A_mat = A[:, None, None].expand(self.nheads, self.head_dim, self.ssm_state_size).to(torch.float32)
            dA = torch.exp(dt_single[:, :, :, None] * A_mat).to(past_key_value_state.ssm_state.device)

            B_single = (
                B.reshape(batch_size, self.n_groups, -1)[..., None, :]
                .expand(batch_size, self.n_groups, self.nheads // self.n_groups, -1)
                .reshape(batch_size, self.nheads, -1)
            )
            dB = dt_single[:, :, :, None] * B_single[:, :, None, :]

            hidden_per_head = hidden_states.reshape(batch_size, self.nheads, self.head_dim)

            # update SSM state
            past_key_value_state.ssm_state.copy_(
                past_key_value_state.ssm_state * dA + dB * hidden_per_head[:, :, :, None]
            )

            # compute SSM output y
            C_single = (
                C.reshape(batch_size, self.n_groups, -1)[..., None, :]
                .expand(batch_size, self.n_groups, self.nheads // self.n_groups, -1)
                .reshape(batch_size, self.nheads, -1)
            )
            ssm_states = past_key_value_state.ssm_state.to(C_single.dtype)
            y = torch.bmm(
                ssm_states.reshape(batch_size * self.nheads, self.head_dim, self.ssm_state_size),
                C_single.reshape(batch_size * self.nheads, self.ssm_state_size, 1),
            ).reshape(batch_size, self.nheads, self.head_dim)

            # D‑skip connection
            y = y + hidden_per_head * self.D[:, None]
            y = y.reshape(batch_size, 1, -1)  # [B, 1, intermediate_size]

        # full‑sequence
        else:
            dt_full = F.softplus(dt + self.dt_bias).clamp(*self.time_step_limit)

            hidden_full = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            B_full      = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            C_full      = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B_full      = B_full.repeat(1, 1, self.nheads // self.n_groups, 1)
            C_full      = C_full.repeat(1, 1, self.nheads // self.n_groups, 1)

            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
            D_residual = self.D[:, None] * pad_tensor_by_size(hidden_full, pad_size)

            hidden_full = hidden_full * dt_full[..., None]
            A_scaled    = A.to(hidden_full.dtype) * dt_full

            hidden_chunks, A_chunks, B_chunks, C_chunks = [
                reshape_into_chunks(t, pad_size, self.chunk_size)
                for t in (hidden_full, A_scaled, B_full, C_full)
            ]  # shapes: [B, n_chunks, L, H, ...] / [B, H, n_chunks, L] after permute below

            # intra‑chunk triangular exponentials
            A_perm = A_chunks.permute(0, 3, 1, 2)  # [B, H, n_chunks, L]
            lower_excl = A_perm[..., None].expand(-1, -1, -1, -1, self.chunk_size)
            lower_excl = lower_excl.masked_fill(~self.mask_excl, 0)
            segsum = torch.cumsum(lower_excl, dim=-2)
            L_tri  = torch.exp(segsum.masked_fill(~self.mask_incl, float("-inf")))

            A_cumsum = torch.cumsum(A_perm, dim=-1)  # [B, H, n_chunks, L]

            # G, M, Y_diag
            G = (
                C_chunks[:, :, :, None, :, :] *
                B_chunks[:, :, None, :, :, :]
            ).sum(-1)  # [B, n_chunks, L, L, H]
            M = G * L_tri.permute(0, 2, 3, 4, 1)
            Y_diag = (M[..., None] * hidden_chunks[:, :, None]).sum(3)  # [B, n_chunks, L, H, D]

            # local SSM states
            decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)
            B_decay = B_chunks * decay_states.permute(0, 2, 3, 1)[..., None]
            states = (B_decay[..., None, :] * hidden_chunks[..., None]).sum(2)  # [B, n_chunks, H, D]

            if past_key_value_state is not None and past_key_value_state.has_previous_state:
                previous_states = past_key_value_state.ssm_state[:, None].to(states.device)
            else:
                previous_states = torch.zeros_like(states[:, :1])

            states_cat = torch.cat([previous_states, states], dim=1)  # [B, n_chunks+1, H, D]

            chunk_sum = A_cumsum[..., -1]  # [B, H, n_chunks]
            decay_chunk = torch.exp(segment_sum(F.pad(chunk_sum, (1, 0)))).transpose(1, 3)
            new_states = (decay_chunk[..., None, None] * states_cat[:, :, None]).sum(1)  # [B, n_chunks+1, H, D]

            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # off‑diagonal output
            state_decay_out = torch.exp(A_cumsum)
            C_times_states = C_chunks[..., None, :] * states[:, :, None]
            Y_off = C_times_states.sum(-1) * state_decay_out.permute(0, 2, 3, 1)[..., None]

            # combine diagonal & off‑diag, reshape
            y = Y_diag + Y_off  # [B, n_chunks, L, H, D]
            y = y.reshape(batch_size, -1, self.nheads, self.head_dim) + D_residual
            if pad_size:
                y = y[:, :seq_len]
            y = y.reshape(batch_size, seq_len, -1)

            # write SSM state to cache
            if past_key_value_state is not None:
                past_key_value_state.ssm_state.copy_(ssm_state)
                past_key_value_state.has_previous_state = True

        # 4. Final linear projection
        contextualised_states = self.out_proj(self.norm(y, gate).to(dtype))
        return contextualised_states, past_key_value_state
