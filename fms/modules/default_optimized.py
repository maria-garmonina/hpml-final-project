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


class SSM(nn.Module):
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

        # ---------------- constants ----------------
        self.nheads           = nheads
        self.emb_dim          = emb_dim
        self.ssm_state_size   = state_size
        self.conv_kernel_size = conv_kernel
        self.intermediate_size = int(expand * emb_dim)
        self.use_conv_bias    = use_conv_bias
        self.act              = str_to_activation(activation_fn)

        self.n_groups   = n_groups
        self.head_dim   = head_dim
        self.chunk_size = chunk_size

        # ------------- depth‑wise conv -------------
        self.conv_dim = self.intermediate_size + 2 * n_groups * state_size
        self.conv1d   = nn.Conv1d(
            in_channels = self.conv_dim,
            out_channels= self.conv_dim,
            kernel_size = conv_kernel,
            padding     = conv_kernel - 1,
            groups      = self.conv_dim,
            bias        = use_conv_bias,
        )

        # ---------------- projections --------------
        proj_size    = self.intermediate_size + self.conv_dim + nheads
        self.in_proj = nn.Linear(emb_dim, proj_size, bias=use_bias)

        # SSM parameters
        self.dt_bias = nn.Parameter(torch.ones(nheads))
        A            = torch.arange(1, nheads + 1)
        self.A_log   = nn.Parameter(torch.log(A))
        self.D       = nn.Parameter(torch.ones(nheads))

        # output
        self.norm      = RMSNormGated(self.intermediate_size, eps=norm_eps)
        self.out_proj  = nn.Linear(self.intermediate_size, emb_dim, bias=use_bias)
        self.time_step_limit = (0.0, float("inf"))

        # -------- pre‑compute intra‑chunk masks ----
        excl = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool), -1)
        incl = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool),  0)
        self.register_buffer("mask_excl", excl[None, None, None])  # [1,1,1,L,L]
        self.register_buffer("mask_incl", incl[None, None, None])

    # ------------------------------------------------------------------ #
    @torch.compile(backend="inductor", dynamic=True)
    def forward(
        self,
        input_states: Tensor,
        mask: Tensor,
        past_key_value_state: Optional[SSMCacheUnit] = None,
        cache_position: Optional[Tensor] = None,
        **_,
    ):
        bsz, seqlen, _ = input_states.shape
        dtype = input_states.dtype

        # ---- 1. gated‑MLP projection -----------------------------------
        x = apply_mask_to_padding_states(input_states, mask)
        proj = self.in_proj(x)
        gate, hidden_BC, dt = proj.split(
            [self.intermediate_size, self.conv_dim, self.nheads], dim=-1
        )


        use_pre = (
            past_key_value_state is not None
            and past_key_value_state.has_previous_state
            and seqlen == 1
            and past_key_value_state.conv_state.shape[0]
            == past_key_value_state.ssm_state.shape[0]
            == bsz
            and cache_position is not None
        )

        # ---- 2. depth‑wise convolution ---------------------------------
        if use_pre:
            # roll cache, insert newest vector
            conv_state = past_key_value_state.conv_state.roll(-1, dims=-1)
            conv_state[:, :, -1] = hidden_BC[:, 0, :].to(conv_state.device)
            past_key_value_state.conv_state.copy_(conv_state)

            cs  = conv_state.to(self.conv1d.weight.device)
            hBC = (cs * self.conv1d.weight.squeeze(1)).sum(-1)
            if self.use_conv_bias:
                hBC = hBC + self.conv1d.bias
            hidden_BC = self.act(hBC)
        else:
            hb_t = hidden_BC.transpose(1, 2)
            if past_key_value_state is not None:
                pad_len = self.conv_kernel_size - hb_t.shape[-1]
                past_key_value_state.conv_state.copy_(F.pad(hb_t, (pad_len, 0)))
            hidden_BC = self.act(
                self.conv1d(hb_t)[..., :seqlen].transpose(1, 2)
            )

        hidden_BC = apply_mask_to_padding_states(hidden_BC, mask)

        # ---- 3. split into hidden / B / C -------------------------------
        hidden, B, C = torch.split(
            hidden_BC,
            [self.intermediate_size,
             self.n_groups * self.ssm_state_size,
             self.n_groups * self.ssm_state_size],
            dim=-1,
        )

        A = -torch.exp(self.A_log.float())               # [H]


        if use_pre:
            past_key_value_state.has_previous_state = True

            # ----- discretise -------------------------------------------
            dt1 = F.softplus(dt[:, 0, :] + self.dt_bias) \
                      .clamp(*self.time_step_limit)          # [B,H]
            dt1 = dt1[:, :, None].expand(bsz, self.nheads, self.head_dim)  # [B,H,D]

            A_mat = A[:, None, None].expand(self.nheads,
                                            self.head_dim,
                                            self.ssm_state_size).to(torch.float32)
            dA = torch.exp(dt1[:, :, :, None] * A_mat)[None]  # [1,B,H,D,N] after broadcast
            dA = dA.squeeze(0).to(past_key_value_state.ssm_state.device)

            B1 = (
                B.reshape(bsz, self.n_groups, -1)[..., None, :]
                .expand(bsz, self.n_groups,
                        self.nheads // self.n_groups, -1)
                .reshape(bsz, self.nheads, -1)
            )
            dB = dt1[:, :, :, None] * B1[:, :, None, :]  # [B,H,D,N]

            hid1 = hidden.reshape(bsz, self.nheads, self.head_dim)

            # ----- state update -----------------------------------------
            past_key_value_state.ssm_state.copy_(
                past_key_value_state.ssm_state * dA + dB * hid1[:, :, :, None]
            )

            C1 = (
                C.reshape(bsz, self.n_groups, -1)[..., None, :]
                .expand(bsz, self.n_groups,
                        self.nheads // self.n_groups, -1)
                .reshape(bsz, self.nheads, -1)
            )
            ssm = past_key_value_state.ssm_state.to(C1.dtype)
            y = torch.bmm(
                ssm.reshape(bsz * self.nheads, self.head_dim, self.ssm_state_size),
                C1.reshape(bsz * self.nheads, self.ssm_state_size, 1),
            ).reshape(bsz, self.nheads, self.head_dim)

            y = y + hid1 * self.D[:, None]        # D‑skip
            y = y.reshape(bsz, 1, -1)             # [B,1,intermediate]


        else:
            dt2 = F.softplus(dt + self.dt_bias).clamp(*self.time_step_limit)

            hid2 = hidden.reshape(bsz, seqlen, -1, self.head_dim).float()
            B2   = B.reshape(bsz, seqlen, -1, self.ssm_state_size).float()
            C2   = C.reshape(bsz, seqlen, -1, self.ssm_state_size).float()
            B2   = B2.repeat(1, 1, self.nheads // self.n_groups, 1)
            C2   = C2.repeat(1, 1, self.nheads // self.n_groups, 1)

            pad_sz   = (self.chunk_size - seqlen % self.chunk_size) % self.chunk_size
            D_resid  = self.D[:, None] * pad_tensor_by_size(hid2, pad_sz)

            hid2 = hid2 * dt2[:, :, None]
            A_s  = A.to(hid2.dtype) * dt2

            hid_c, A_c_raw, B_c, C_c = [
                reshape_into_chunks(t, pad_sz, self.chunk_size)
                for t in (hid2, A_s, B2, C2)
            ]                                        # hid_c: [B,C,L,H,D]

            # ---- intra‑chunk triangular exp‑sum ------------------------
            A_perm = A_c_raw.permute(0, 3, 1, 2)      # [B,H,C,L]
            L_ex   = A_perm[..., None].expand(-1,-1,-1,-1, self.chunk_size)  # [B,H,C,L,L]
            L_ex   = L_ex.masked_fill(~self.mask_excl, 0)
            segsum = torch.cumsum(L_ex, dim=-2)
            L      = torch.exp(
                         segsum.masked_fill(~self.mask_incl, float("-inf"))
                     )                                # [B,H,C,L,L]

            A_cumsum = torch.cumsum(A_perm, dim=-1)   # [B,H,C,L]

            # ---- G, M, Y_diag ------------------------------------------
            G = (
                C_c[:, :, :, None, :, :] *
                B_c[:, :, None, :, :, :]
            ).sum(-1)                                 # [B,C,L,L,H]
            M = G * L.permute(0,2,3,4,1)              # [B,C,L,L,H]
            Y_diag = (M[..., None] * hid_c[:, :, None]).sum(3)  # [B,C,L,H,D]

            # ---- local states ------------------------------------------
            decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)  # [B,H,C,L]
            B_decay = B_c * decay_states.permute(0,2,3,1)[..., None]
            states = (B_decay[..., None, :] * hid_c[..., None]).sum(2)  # [B,C,H,D]

            if past_key_value_state is not None and past_key_value_state.has_previous_state:
                previous_states = past_key_value_state.ssm_state[:, None].to(states.device)
            else:
                previous_states = torch.zeros_like(states[:, :1])

            states_cat = torch.cat([previous_states, states], dim=1)  # [B,C+1,H,D]

            chunk_sum = A_cumsum[..., -1]             # [B,H,C]
            decay_chunk = torch.exp(
                segment_sum(F.pad(chunk_sum, (1, 0)))   # → [B,H,C+1,C+1]
            ).transpose(1, 3)                          # [B,C+1,C+1,H]

            new_states = (decay_chunk[..., None, None] *
                          states_cat[:, :, None]).sum(1)  # [B,C+1,H,D]

            states, ssm_state = new_states[:, :-1], new_states[:, -1]

            # ---- off‑diagonal output -----------------------------------
            state_decay_out = torch.exp(A_cumsum)     # [B,H,C,L]
            Cst = C_c[..., None, :] * states[:, :, None]
            Y_off = Cst.sum(-1) * \
                    state_decay_out.permute(0,2,3,1)[..., None]

            # ---- combine & reshape -------------------------------------
            y = Y_diag + Y_off                        # [B,C,L,H,D]
            y = y.reshape(bsz, -1, self.nheads, self.head_dim) + D_resid
            if pad_sz:
                y = y[:, :seqlen]
            y = y.reshape(bsz, seqlen, -1)

            # write cache
            if past_key_value_state is not None:
                past_key_value_state.ssm_state.copy_(ssm_state)
                past_key_value_state.has_previous_state = True

        # ---- 4. output projection --------------------------------------
        out = self.out_proj(self.norm(y, gate).to(dtype))
        return out, past_key_value_state
