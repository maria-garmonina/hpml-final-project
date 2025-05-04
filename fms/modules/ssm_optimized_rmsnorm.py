from typing import Optional
import torch
import torch.nn as nn
from fms.utils.activation import str_to_activation
import torch.nn.functional as F

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

        # convolution + point‑wise projections
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
        self.norm = RMSNormGated(self.intermediate_size, eps=norm_eps)
        self.D = nn.Parameter(torch.ones(nheads))

        self.time_step_limit = (0.0, float("inf"))
        self.out_proj = nn.Linear(self.intermediate_size, emb_dim, bias=use_bias)

        # helpers
        self._hidden_buf: Optional[torch.Tensor] = None

    @torch.compile(backend="inductor", dynamic=True)
    def forward(
        self,
        input_states: torch.Tensor,
        mask: Optional[torch.Tensor],
        past_key_value_state: Optional[SSMCacheUnit] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        bsz, seq_len, _ = input_states.shape
        dtype = input_states.dtype

        # 1. gated‑MLP projection

        input_states = apply_mask_to_padding_states(input_states, mask)
        proj = self.in_proj(input_states)
        gate, hidden_BC, dt = proj.split(
            [self.intermediate_size, self.conv_dim, self.nheads], dim=-1
        )

        # detect cache vs full‑sequence path
        use_pre = (
            past_key_value_state is not None
            and past_key_value_state.has_previous_state
            and seq_len == 1
            and past_key_value_state.conv_state.shape[0] == bsz
            and past_key_value_state.ssm_state.shape[0] == bsz
            and cache_position is not None
        )

        if use_pre:
            conv_state = past_key_value_state.conv_state.roll(-1, dims=-1)
            conv_state[:, :, -1] = hidden_BC[:, 0, :].to(conv_state.device)
            past_key_value_state.conv_state.copy_(conv_state)
            cs = conv_state.to(self.conv1d.weight.device)
            hidden_BC = (cs * self.conv1d.weight.squeeze(1)).sum(dim=-1)
            if self.use_conv_bias:
                hidden_BC.add_(self.conv1d.bias)
            hidden_BC = self.act(hidden_BC)
        else:
            hbt = hidden_BC.transpose(1, 2)
            hidden_BC = self.act(self.conv1d(hbt)[..., :seq_len].transpose(1, 2))

        hidden_BC = apply_mask_to_padding_states(hidden_BC, mask)


        hidden, B, C = torch.split(
            hidden_BC,
            [
                self.intermediate_size,
                self.n_groups * self.ssm_state_size,
                self.n_groups * self.ssm_state_size,
            ],
            dim=-1,
        )


        # 4. SSM transform

        A = -torch.exp(self.A_log.float())

        if use_pre:
            cache_device = past_key_value_state.ssm_state.device

            dt1 = F.softplus(dt[:, 0, :][:, None, :] + self.dt_bias[:, None])
            dt1 = dt1.clamp(*self.time_step_limit).expand(bsz, self.nheads, self.head_dim)

            A_mat = A[..., None, None].expand(self.nheads, self.head_dim, self.ssm_state_size)
            dA = torch.exp(dt1[..., None] * A_mat).to(device=cache_device)

            Bg = B.reshape(bsz, self.n_groups, -1)[..., None, :]
            Bg = Bg.expand(bsz, self.n_groups, self.nheads // self.n_groups, Bg.shape[-1]).contiguous()
            Bg = Bg.reshape(bsz, -1, Bg.shape[-1])
            dB = dt1[..., None] * Bg[..., None, :]

            hidden_h = hidden.reshape(bsz, -1, self.head_dim)
            dBx = (dB * hidden_h[..., None]).to(device=cache_device)
            past_key_value_state.ssm_state.copy_(past_key_value_state.ssm_state * dA + dBx)

            Cg = C.reshape(bsz, self.n_groups, -1)[..., None, :]
            Cg = Cg.expand(bsz, self.n_groups, self.nheads // self.n_groups, Cg.shape[-1]).contiguous()
            Cg = Cg.reshape(bsz, -1, Cg.shape[-1])

            ssm_states = past_key_value_state.ssm_state.to(device=Cg.device, dtype=Cg.dtype)
            y = torch.bmm(
                ssm_states.view(bsz * self.nheads, self.head_dim, self.ssm_state_size),
                Cg.view(bsz * self.nheads, self.ssm_state_size, 1),
            ).view(bsz, self.nheads, self.head_dim)

            D_mat = self.D[:, None].expand(self.nheads, self.head_dim)
            y = (y + hidden_h * D_mat).to(y.dtype).reshape(bsz, -1)[:, None, :]

        else:
            dt2 = F.softplus(dt + self.dt_bias).clamp(*self.time_step_limit)

            # reshape hidden → 4‑D then pad + chunk
            hid = hidden.reshape(bsz, seq_len, self.nheads, self.head_dim)
            hid = hid * dt2.view(bsz, seq_len, self.nheads, 1)
            pad = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
            total = seq_len + pad
            n_chunks = total // self.chunk_size

            hid_pad = pad_tensor_by_size(hid, pad)
            hid_chunks = hid_pad.view(bsz, n_chunks, self.chunk_size, self.nheads, self.head_dim)

            # prepare B & C (repeat heads, pad, chunk)
            B_ = B.reshape(bsz, seq_len, -1, self.ssm_state_size)
            C_ = C.reshape(bsz, seq_len, -1, self.ssm_state_size)
            rep = self.nheads // self.n_groups
            B_ = B_.repeat(1, 1, rep, 1).contiguous()
            C_ = C_.repeat(1, 1, rep, 1).contiguous()

            B_pad = pad_tensor_by_size(B_, pad)
            C_pad = pad_tensor_by_size(C_, pad)
            B_chunks = B_pad.view(bsz, n_chunks, self.chunk_size, self.nheads, self.ssm_state_size)
            C_chunks = C_pad.view(bsz, n_chunks, self.chunk_size, self.nheads, self.ssm_state_size)

            # A ← (A * dt)  then pad + chunk  then permute
            A_seq = (A.to(hid.dtype) * dt2).view(bsz, seq_len, -1)
            A_pad = pad_tensor_by_size(A_seq, pad)
            A_chunks = reshape_into_chunks(A_pad, 0, self.chunk_size)  # [b, n_c, chunk, nheads]
            A_chunks = A_chunks.permute(0, 3, 1, 2).contiguous()       # [b, nheads, n_c, chunk]

            A_cumsum = torch.cumsum(A_chunks, dim=-1)                  # same as reference

            L = torch.exp(segment_sum(A_chunks))                       # identical lower‑tri mask

            # 1. intra‑chunk output (Y_diag)
            G_inter = C_chunks[:, :, :, None, :, :] * B_chunks[:, :, None, :, :, :]
            G = G_inter.sum(dim=-1)                                    # sum over ssm_state_size
            M_inter = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_inter.sum(dim=-1)
            Y_diag = (M[..., None] * hid_chunks[:, :, None]).sum(dim=3)

            # 2. local (chunk) states
            decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
            B_decay = B_chunks * decay_states.permute(0, -2, -1, 1)[..., None]
            states = (B_decay[..., None, :] * hid_chunks[..., None]).sum(dim=2)

            # 3. off‑diagonal term (uses cumulative sum, like reference)
            state_decay_out = torch.exp(A_cumsum)                      # *** fixed ***
            C_times_states = C_chunks[..., None, :] * states[:, :, None, ...]
            state_decay_out_perm = state_decay_out.permute(0, 2, 3, 1)
            Y_off = C_times_states.sum(-1) * state_decay_out_perm[..., None]

            # 4. D‑residual
            D_res = hid_pad * self.D.view(1, 1, self.nheads, 1)

            # combine & reshape
            y = Y_diag + Y_off + D_res.view(bsz, n_chunks, self.chunk_size, self.nheads, self.head_dim)
            y = y.view(bsz, total, self.nheads, self.head_dim)
            if pad > 0:
                y = y[:, :seq_len]
            y = y.reshape(bsz, seq_len, -1).contiguous()

        # 5. gated RMSNorm + linear out
        out = self.norm(y, gate)
        out = self.out_proj(out.to(dtype))
        return out, past_key_value_state
