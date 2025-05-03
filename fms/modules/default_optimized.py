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
        # [bsz, seq_len multiple of chunk_size, nheads] -> [bsz, -1, chunk_size, nheads]
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2]
        )
    else:
        # [bsz, seq_len multiple of chunk_size, nheads, head_dim/state_size] 
        # -> [bsz, -1, chunk_size, nheads, head_dim/state_size]
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
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)
    mask = torch.tril(
        torch.ones(
            chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool
        ),
        diagonal=-1,
    )
    input_tensor = input_tensor.masked_fill(~mask, 0)
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    mask = torch.tril(
        torch.ones(
            chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool
        ),
        diagonal=0,
    )
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
        self.nheads = nheads
        self.emb_dim = emb_dim
        self.ssm_state_size = state_size
        self.intermediate_size = int(expand * emb_dim)
        self.use_conv_bias = use_conv_bias
        self.act = str_to_activation(activation_fn)
        self.n_groups = n_groups
        self.head_dim = head_dim
        self.chunk_size = chunk_size

        # convolution + gated-MLP dims
        self.conv_dim = self.intermediate_size + 2 * n_groups * state_size
        self.conv1d   = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=use_conv_bias,
            kernel_size=conv_kernel,
            groups=self.conv_dim,
            padding=conv_kernel - 1,
        )
        proj_size = self.intermediate_size + self.conv_dim + nheads
        self.in_proj = nn.Linear(emb_dim, proj_size, bias=use_bias)

        # SSM parameters
        self.dt_bias = nn.Parameter(torch.ones(nheads))
        A = torch.arange(1, nheads + 1)
        self.A_log    = nn.Parameter(torch.log(A))
        self.D        = nn.Parameter(torch.ones(nheads))
        self.norm     = RMSNormGated(self.intermediate_size, eps=norm_eps)
        self.out_proj = nn.Linear(self.intermediate_size, emb_dim, bias=use_bias)
        self.time_step_limit = (0.0, float("inf"))

        # buffers for segment-sum masks
        mask_excl = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=-1)
        mask_incl = torch.tril(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
        self.register_buffer("mask_excl", mask_excl)
        self.register_buffer("mask_incl", mask_incl)

    @torch.compile(backend="inductor", dynamic=True)
    def forward(
        self,
        input_states: torch.Tensor,
        mask: torch.Tensor,
        past_key_value_state: Optional[SSMCacheUnit] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        bsz, seq_len, _ = input_states.shape
        dtype, device  = input_states.dtype, input_states.device

        # Gated MLP + convolution
        x    = apply_mask_to_padding_states(input_states, mask)
        proj = self.in_proj(x)
        gate, hidden_BC, dt = proj.split(
            [self.intermediate_size, self.conv_dim, self.nheads], dim=-1
        )

        use_pre = (
            past_key_value_state is not None
            and past_key_value_state.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )

        if use_pre:
            # same as default
            conv_state = past_key_value_state.conv_state.roll(shifts=-1, dims=-1)
            conv_state[:, :, -1] = hidden_BC[:, 0, :].to(conv_state.device)
            past_key_value_state.conv_state.copy_(conv_state)
            cs = conv_state.to(self.conv1d.weight.device)
            hBC = (cs * self.conv1d.weight.squeeze(1)).sum(dim=-1)
            if self.use_conv_bias:
                hBC = hBC + self.conv1d.bias
            hidden_BC = self.act(hBC)
        else:
            hb_t = hidden_BC.transpose(1, 2)
            hidden_BC = self.act(
                self.conv1d(hb_t)[..., :seq_len].transpose(1, 2)
            )
        hidden_BC = apply_mask_to_padding_states(hidden_BC, mask)

        # Split into hidden, B, C
        hidden, B, C = torch.split(
            hidden_BC,
            [self.intermediate_size,
             self.n_groups * self.ssm_state_size,
             self.n_groups * self.ssm_state_size],
            dim=-1,
        )

        # SSM transform
        A = -torch.exp(self.A_log.float())

        if use_pre:
            # same as default
            dt1 = dt[:, 0, :][:, None, :].transpose(1, 2)
            dt1 = dt1.expand(bsz, self.nheads, self.head_dim)
            dt1 = F.softplus(dt1 + self.dt_bias[:, None]).clamp(*self.time_step_limit)
            A_mat = A[..., None, None].expand(self.nheads, self.head_dim, self.ssm_state_size)
            dA    = torch.exp(dt1[..., None] * A_mat).to(past_key_value_state.ssm_state.device)
            Bg = (
                B.reshape(bsz, self.n_groups, -1)[..., None, :]
                .expand(bsz, self.n_groups, self.nheads//self.n_groups, -1)
                .reshape(bsz, -1, self.ssm_state_size)
            )
            dB   = dt1[..., None] * Bg[..., None, :]
            hid  = hidden.reshape(bsz, -1, self.head_dim)
            dBx  = (dB * hid[..., None]).to(past_key_value_state.ssm_state.device)
            past_key_value_state.ssm_state.copy_(
                past_key_value_state.ssm_state * dA + dBx
            )
            # C-based output
            Cg = (
                C.reshape(bsz, self.n_groups, -1)[..., None, :]
                .expand(bsz, self.n_groups, self.nheads//self.n_groups, -1)
                .reshape(bsz, -1, self.ssm_state_size)
            )
            s_flat = past_key_value_state.ssm_state.to(Cg.device, Cg.dtype)
            y = torch.bmm(
                s_flat.view(bsz*self.nheads, self.head_dim, self.ssm_state_size),
                Cg.view(bsz*self.nheads, self.ssm_state_size, 1),
            ).view(bsz, self.nheads, self.head_dim)
            y = (y + hid * self.D[:, None]).view(bsz, -1)[:, None, :]
        else:
            # full-sequence fused path
            dt2 = F.softplus(dt + self.dt_bias).clamp(*self.time_step_limit)
            hid = hidden.reshape(bsz, seq_len, self.nheads, self.head_dim)
            pad = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
            total = seq_len + pad
            nc = total // self.chunk_size

            # chunk data
            hpad = pad_tensor_by_size(hid, pad)
            hidden_chunks = hpad.view(bsz, nc, self.chunk_size, self.nheads, self.head_dim)
            B_ = pad_tensor_by_size(
                B.reshape(bsz, seq_len, -1, self.ssm_state_size)
                 .repeat(1,1,self.nheads//self.n_groups,1),
                pad
            ).view(bsz, nc, self.chunk_size, self.nheads, self.ssm_state_size)
            C_ = pad_tensor_by_size(
                C.reshape(bsz, seq_len, -1, self.ssm_state_size)
                 .repeat(1,1,self.nheads//self.n_groups,1),
                pad
            ).view(bsz, nc, self.chunk_size, self.nheads, self.ssm_state_size)

            Aseq = (A.to(hid.dtype) * dt2).view(bsz, seq_len, -1)
            Ach  = pad_tensor_by_size(Aseq, pad)
            Ach  = reshape_into_chunks(Ach, 0, self.chunk_size)
            A_chunks = Ach.permute(0,3,1,2).contiguous()

            # segment sum
            expA = A_chunks[..., None].expand(*A_chunks.size(), self.chunk_size)
            expA = expA.masked_fill(~self.mask_excl.unsqueeze(0).unsqueeze(0), 0)
            ss   = torch.cumsum(expA, dim=-2)
            L    = torch.exp(ss.masked_fill(~self.mask_incl.unsqueeze(0).unsqueeze(0), float('-inf')))

            # intra-chunk
            G = torch.einsum('b c i h n, b c j h n -> b c i j h', C_, B_)
            M = (G[..., None] * L.permute(0,2,3,4,1)[..., None]).sum(dim=-1)
            Yd = (M[..., None] * hidden_chunks[:,:,None]).sum(dim=3)

            # local states & inter-chunk
            decay   = torch.exp(A_chunks[..., -1:] - A_chunks)                       # [B, H, C, L]
            Bdec    = B_ * decay.permute(0,2,3,1)[..., None]                         # [B, C, L, H, state_size]
            states  = Bdec.sum(dim=2)

            prev = (
                past_key_value_state.ssm_state[:, None,...].to(states.device)
                if past_key_value_state else
                torch.zeros_like(states[:,:1])
            )

            cat       = torch.cat([prev, states], dim=1)  # [B, C+1, H, S]
        
            # sum over the last A_chunks value per chunk, pad on the left for "prev"
            A_leg     = torch.cumsum(A_chunks[..., -1], dim=-1)           # [bsz, nheads, C]
            A_pad     = F.pad(A_leg, (1,0))                              # [bsz, nheads, C+1]
        
            # build a (C+1)x(C+1) lower-triangular mask
            C1        = A_pad.size(-1)
            chunk_mask = torch.tril(
                torch.ones(C1, C1, dtype=torch.bool, device=device),
                diagonal=0,
            )[None,None,:,:]                                            # [1,1,C+1,C+1]
        
            # segment-sum over the chunk axis
            A_exp     = A_pad[..., None].expand(*A_pad.shape, C1)        # [bsz, nheads, C+1, C+1]
            A_exp     = A_exp.masked_fill(~chunk_mask, 0)
            S         = torch.cumsum(A_exp,    dim=-2)                  # [bsz,nheads,C+1,C+1]
            S         = S.masked_fill(~chunk_mask, float('-inf'))
            decay_ch  = torch.exp(S).transpose(1,3)                     # [bsz, C+1, C+1, nheads]
        
            # decay_ch: [B, C+1, C+1, H]
            d = decay_ch.unsqueeze(-1)      # [B, C+1, C+1, H, 1]
            c = cat.unsqueeze(1)            # [B, 1,  C+1, H, S]
            new_full = (d * c).sum(dim=1)   # [B, C+1, H, S]
            
            states, ssm_state = new_full[:, :-1], new_full[:, -1]

            if past_key_value_state: past_key_value_state.ssm_state.copy_(ssm_state)

            Cst    = C_[...,None,:] * states[:,:,None]
            sout   = torch.exp(A_chunks).permute(0,2,3,1)
            Yoff   = Cst.sum(dim=-1).mul(sout.unsqueeze(1))
            
            D_res  = hpad * self.D.view(1,1,self.nheads,1)

            y = Yd + Yoff.unsqueeze(-1)  # -> [B, nc, chunk_size, H, D]
            y = y.reshape(bsz, total, self.nheads, self.head_dim)  # [B, total, H, D]

            D_res = hpad * self.D.view(1, 1, self.nheads, 1)                     # [B, total, H, D]
            y  = y + D_res

            if pad:
                y = y[:, :seq_len]

        # gating + normalization + output
        gated = self.norm(y, gate)
        out   = self.out_proj(gated.to(dtype))
        return out, past_key_value_state
