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
        self.conv_kernel_size = conv_kernel
        self.use_conv_bias = use_conv_bias
        self.act = str_to_activation(activation_fn)
        self.n_groups = n_groups
        self.head_dim = head_dim
        self.chunk_size = chunk_size

        # convolution + gated-MLP dims
        self.conv_dim = self.intermediate_size + 2 * n_groups * state_size
        self.conv1d = nn.Conv1d(
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
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(nheads))
        self.norm = RMSNormGated(self.intermediate_size, eps=norm_eps)
        self.out_proj = nn.Linear(self.intermediate_size, emb_dim, bias=use_bias)
        self.time_step_limit = (0.0, float("inf"))

        # buffers for manual segment-sum masks
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
        dtype, device = input_states.dtype, input_states.device

        # 1. Gated-MLP + conv input
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

        # 2. Convolution
        if use_pre:
            # identical to default
            conv_state = past_key_value_state.conv_state.roll(shifts=-1, dims=-1)
            conv_state[:, :, -1] = hidden_BC[:, 0, :].to(conv_state.device)
            past_key_value_state.conv_state.copy_(conv_state)

            cs  = conv_state.to(self.conv1d.weight.device)
            hBC = (cs * self.conv1d.weight.squeeze(1)).sum(dim=-1)
            if self.use_conv_bias:
                hBC = hBC + self.conv1d.bias
            hidden_BC = self.act(hBC)
        else:
            # initialize cache
            hb_t = hidden_BC.transpose(1, 2)
            if past_key_value_state is not None:
                pad_len    = self.conv_kernel_size - hb_t.shape[-1]
                conv_states = F.pad(hb_t, (pad_len, 0))
                past_key_value_state.conv_state.copy_(conv_states)
            hidden_BC = self.act(
                self.conv1d(hb_t)[..., :seq_len].transpose(1, 2)
            )

        hidden_BC = apply_mask_to_padding_states(hidden_BC, mask)

        # Split into hidden / B / C 
        hidden, B, C = torch.split(
            hidden_BC,
            [self.intermediate_size,
             self.n_groups * self.ssm_state_size,
             self.n_groups * self.ssm_state_size],
            dim=-1,
        )

        # Common SSM params
        A = -torch.exp(self.A_log.float())

        #auto-regressive path
        if use_pre:
            cache_device = past_key_value_state.ssm_state.device

            # Discretize dt
            dt1 = dt[:, 0, :]                     # [bsz, nheads]
            dt1 = dt1.unsqueeze(-1)               # [bsz, nheads, 1]
            dt1 = dt1.expand(bsz, self.nheads, self.head_dim)
            dt1 = F.softplus(dt1 + self.dt_bias.unsqueeze(-1))
            dt1 = dt1.clamp(*self.time_step_limit)

            # Build A matrix -> dA
            A_t = A[..., None, None].expand(
                self.nheads, self.head_dim, self.ssm_state_size
            ).to(torch.float32)
            dA = torch.exp(dt1.unsqueeze(-1) * A_t).to(cache_device)

            # Discretize B -> dB
            B1 = B.reshape(bsz, self.n_groups, -1)[..., None, :]  # [bsz,n_groups,1,ssm_state_size]
            B1 = B1.expand(
                bsz, self.n_groups, self.nheads // self.n_groups, B1.shape[-1]
            ).contiguous()                                         # [bsz,n_groups,nheads/groups,ssm_state_size]
            B1 = B1.reshape(bsz, -1, B1.shape[-1])                 # [bsz,nheads,ssm_state_size]
            dB = dt1.unsqueeze(-1) * B1.unsqueeze(-2)              # [bsz,nheads,head_dim,ssm_state_size]

            # Hidden → match (bsz,nheads,head_dim)
            hid1 = hidden.reshape(bsz, -1, self.head_dim)

            # State update: s_new = s_old * dA + dB * hid
            dBx = (dB * hid1.unsqueeze(-1)).to(cache_device)
            past_key_value_state.ssm_state.copy_(
                past_key_value_state.ssm_state * dA + dBx
            )

            # Build C for output
            C1 = C.reshape(bsz, self.n_groups, -1)[..., None, :]
            C1 = C1.expand(
                bsz, self.n_groups, self.nheads // self.n_groups, C1.shape[-1]
            ).contiguous()
            C1 = C1.reshape(bsz, -1, C1.shape[-1])

            # y = BMM(ssm_state, C)
            ssm = past_key_value_state.ssm_state.to(
                device=C1.device, dtype=C1.dtype
            )  # [bsz, nheads, head_dim, ssm_state_size]
            ssm_r = ssm.view(bsz*self.nheads, self.head_dim, self.ssm_state_size)
            C_r   = C1.view(bsz*self.nheads, self.ssm_state_size, 1)
            y     = torch.bmm(ssm_r, C_r).view(bsz, self.nheads, self.head_dim)

            # D skip connection
            Dval = self.D.unsqueeze(-1)  # [nheads,1]
            y    = (y + hid1 * Dval).to(y.dtype)

            # Reshape to [bsz,1,intermediate_size]
            y = y.reshape(bsz, -1)[:, None, :]

        # full seq
        else:
            # time-step discretization
            dt2 = F.softplus(dt + self.dt_bias).clamp(*self.time_step_limit)

            # reshape for chunking
            hid2 = hidden.reshape(bsz, seq_len, -1, self.head_dim).float()
            B2   = B.reshape(bsz, seq_len, -1, self.ssm_state_size).float()\
                       .repeat(1,1,self.nheads//self.n_groups,1)
            C2   = C.reshape(bsz, seq_len, -1, self.ssm_state_size).float()\
                       .repeat(1,1,self.nheads//self.n_groups,1)

            pad_size   = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
            D_residual = self.D[..., None] * pad_tensor_by_size(hid2, pad_size)

            # discretize hidden & A
            hid2 = hid2 * dt2.unsqueeze(-1)
            A2   = A.to(hid2.dtype) * dt2

            # chunk everything
            hidden_chunks = reshape_into_chunks(hid2, pad_size, self.chunk_size)
            A_chunks      = reshape_into_chunks(A2,   pad_size, self.chunk_size)
            B_chunks      = reshape_into_chunks(B2,   pad_size, self.chunk_size)
            C_chunks      = reshape_into_chunks(C2,   pad_size, self.chunk_size)

            # intra-chunk mask & sum
            expA = A_chunks.unsqueeze(-1).expand(*A_chunks.size(), self.chunk_size)
            expA = expA.masked_fill(~self.mask_excl.unsqueeze(0).unsqueeze(0), 0)
            ss   = torch.cumsum(expA, dim=-2)
            L    = torch.exp(
                ss.masked_fill(~self.mask_incl.unsqueeze(0).unsqueeze(0),
                               float("-inf"))
            )

            # diagonal block output
            G   = torch.einsum('b c i h n, b c j h n -> b c i j h',
                              C_chunks, B_chunks)
            M   = (G.unsqueeze(-1)
                     * L.permute(0,2,3,4,1).unsqueeze(-1)
                  ).sum(dim=-1)
            Yd  = (M.unsqueeze(-1) * hidden_chunks.unsqueeze(2)).sum(dim=3)

            # local states (fused)
            decay = torch.exp(A_chunks[..., -1:] - A_chunks)
            Bdec  = B_chunks * decay.permute(0,2,3,1).unsqueeze(-1)
            states= (Bdec.unsqueeze(-2) * hidden_chunks.unsqueeze(-1)).sum(dim=2)

            # inter-chunk recurrence
            if past_key_value_state:
                prev = past_key_value_state.ssm_state.mean(dim=2)\
                           .unsqueeze(1).to(states.device)
            else:
                prev = torch.zeros_like(states[:, :1])
            cat = torch.cat([prev, states], dim=1)

            A_leg     = torch.cumsum(A_chunks[..., -1], dim=-1)
            A_pad     = F.pad(A_leg, (1, 0))
            C1        = A_pad.size(-1)
            chunk_mask= torch.tril(
                torch.ones(C1, C1, device=device, dtype=torch.bool),
                diagonal=0
            )[None,None,:,:]
            A_e       = A_pad.unsqueeze(-1).expand(*A_pad.shape, C1)
            A_e       = A_e.masked_fill(~chunk_mask, 0)
            S2        = torch.cumsum(A_e, dim=-2)
            S2        = S2.masked_fill(~chunk_mask, float("-inf"))
            decay_ch  = torch.exp(S2).transpose(1,3)
            new_full  = (decay_ch.unsqueeze(-1) * cat.unsqueeze(1)).sum(dim=1)
            ssm_flat  = new_full[:, -1]
            ssm_exp   = ssm_flat.unsqueeze(2)\
                           .expand(-1, -1, self.head_dim, -1)
            past_key_value_state.ssm_state.copy_(ssm_exp)

            # off-diagonal output
            Cst  = C_chunks * states.unsqueeze(2)
            Yoff = Cst.sum(dim=-1)\
                   * torch.exp(A_chunks).permute(0,2,3,1).unsqueeze(-1)

            # combine + D residual
            y = Yd + Yoff.unsqueeze(-1)
            total = seq_len + pad_size
            y = y.reshape(bsz, total, self.nheads * self.head_dim)
            y = y + D_residual.reshape(bsz, total,
                                       self.nheads * self.head_dim)
            if pad_size > 0:
                y = y[:, :seq_len]

        # 4. Final norm & projection
        gated = self.norm(y, gate)
        out   = self.out_proj(gated.to(dtype))
        return out, past_key_value_state
