from typing import Optional

import torch
import torch.nn as nn

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
        super(SSM, self).__init__()
        self.nheads = nheads
        self.emb_dim = emb_dim
        self.ssm_state_size = state_size
        self.conv_kernel_size = conv_kernel
        self.intermediate_size = int(expand * self.emb_dim)
        self.use_conv_bias = use_conv_bias
        self.activation = activation_fn
        self.act = str_to_activation(activation_fn)

        self.layer_norm_epsilon = norm_eps
        self.n_groups = n_groups
        self.head_dim = head_dim
        self.chunk_size = chunk_size

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=use_conv_bias,
            kernel_size=conv_kernel,
            groups=self.conv_dim,
            padding=conv_kernel - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.nheads
        self.in_proj = nn.Linear(
            self.emb_dim,
            projection_size,
            bias=use_bias,
        )

        # time step projection (discretization)
        self.dt_bias = nn.Parameter(torch.ones(self.nheads))

        # S4D real init
        A = torch.arange(1, self.nheads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = RMSNormGated(self.intermediate_size, eps=self.layer_norm_epsilon)
        self.D = nn.Parameter(torch.ones(self.nheads))

        self.time_step_limit = (0.0, float("inf"))
        self.time_step_min = 0.001
        self.time_step_max = 0.1
        self.out_proj = nn.Linear(self.intermediate_size, self.emb_dim, bias=use_bias)

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

        # 1. Gated MLP's linear projection
        input_states = apply_mask_to_padding_states(input_states, mask)
        projected_states = self.in_proj(input_states)
        gate, hidden_states_B_C, dt = projected_states.split(
            [self.intermediate_size, self.conv_dim, self.nheads], dim=-1
        )

        use_precomputed_states = (
            past_key_value_state is not None
            and past_key_value_state.has_previous_state
            and seq_len == 1
            and past_key_value_state.conv_state.shape[0] == batch_size
            and past_key_value_state.ssm_state.shape[0] == batch_size
            and cache_position is not None
        )

        # 2. Convolution sequence transformation
        if use_precomputed_states:
            # Single-step (auto-regressive) path
            past_key_value_state.conv_state = past_key_value_state.conv_state.roll(
                shifts=-1, dims=-1
            )
            past_key_value_state.conv_state[:, :, -1] = hidden_states_B_C[:, 0, :].to(
                past_key_value_state.conv_state.device
            )
            conv_states = past_key_value_state.conv_state.to(
                device=self.conv1d.weight.device
            )
            hidden_states_B_C = torch.sum(
                conv_states * self.conv1d.weight.squeeze(1), dim=-1
            )
            if self.use_conv_bias:
                hidden_states_B_C = hidden_states_B_C + self.conv1d.bias
            hidden_states_B_C = self.act(hidden_states_B_C)
        else:
            # Full-sequence path
            hidden_states_B_C_transposed = hidden_states_B_C.transpose(1, 2)

            # CHANGED: remove storing conv state so it does not pass chunk info forward
            # -----------------------------------------------------------------------
            # if past_key_value_state is not None:
            #     conv_states = nn.functional.pad(
            #         hidden_states_B_C_transposed,
            #         (self.conv_kernel_size - hidden_states_B_C_transposed.shape[-1], 0),
            #     )
            #     past_key_value_state.conv_state.copy_(conv_states)
            # -----------------------------------------------------------------------

            hidden_states_B_C = self.act(
                self.conv1d(hidden_states_B_C_transposed)[..., :seq_len].transpose(1, 2)
            )

        hidden_states_B_C = apply_mask_to_padding_states(hidden_states_B_C, mask)
        hidden_states, B, C = torch.split(
            hidden_states_B_C,
            [
                self.intermediate_size,
                self.n_groups * self.ssm_state_size,
                self.n_groups * self.ssm_state_size,
            ],
            dim=-1,
        )

        # 3. SSM transformation
        A = -torch.exp(self.A_log.float())  # [nheads]

        if use_precomputed_states:
            # Single-step (auto-regressive) path remains unchanged
            # This logic is necessary for incremental generation
            cache_device = past_key_value_state.ssm_state.device
            dt = dt[:, 0, :][:, None, ...]
            dt = dt.transpose(1, 2).expand(batch_size, dt.shape[-1], self.head_dim)
            dt_bias = self.dt_bias[..., None].expand(
                self.dt_bias.shape[0], self.head_dim
            )
            dt = torch.nn.functional.softplus(dt + dt_bias.to(dt.dtype))
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])
            A = A[..., None, None].expand(self.nheads, self.head_dim, self.ssm_state_size)
            dA = (torch.exp(dt[..., None] * A)).to(device=cache_device)

            B = B.reshape(batch_size, self.n_groups, -1)[..., None, :]
            B = B.expand(
                batch_size, self.n_groups, self.nheads // self.n_groups, B.shape[-1]
            ).contiguous()
            B = B.reshape(batch_size, -1, B.shape[-1])
            dB = dt[..., None] * B[..., None, :]

            hidden_states = hidden_states.reshape(batch_size, -1, self.head_dim)
            dBx = (dB * hidden_states[..., None]).to(device=cache_device)
            past_key_value_state.ssm_state.copy_(
                past_key_value_state.ssm_state * dA + dBx
            )

            # subsequent output
            C = C.reshape(batch_size, self.n_groups, -1)[..., None, :]
            C = C.expand(
                batch_size, self.n_groups, self.nheads // self.n_groups, C.shape[-1]
            ).contiguous()
            C = C.reshape(batch_size, -1, C.shape[-1])

            ssm_states = past_key_value_state.ssm_state.to(
                device=C.device, dtype=C.dtype
            )
            ssm_states_reshaped = ssm_states.view(
                batch_size * self.nheads, self.head_dim, self.ssm_state_size
            )
            C_reshaped = C.view(batch_size * self.nheads, self.ssm_state_size, 1)
            y = torch.bmm(ssm_states_reshaped, C_reshaped)
            y = y.view(batch_size, self.nheads, self.head_dim)

            D = self.D[..., None].expand(self.D.shape[0], self.head_dim)
            y = (y + hidden_states * D).to(y.dtype)
            y = y.reshape(batch_size, -1)[:, None, ...]
        else:
            # CHANGED: remove cross-chunk state transfer so each chunk is independent
            dt = nn.functional.softplus(dt + self.dt_bias)
            dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])

            hidden_states = hidden_states.reshape(batch_size, seq_len, -1, self.head_dim).float()
            B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
            B = B.repeat(1, 1, self.nheads // self.n_groups, 1)
            C = C.repeat(1, 1, self.nheads // self.n_groups, 1)
            pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

            D_residual = self.D[..., None] * pad_tensor_by_size(hidden_states, pad_size)

            hidden_states = hidden_states * dt[..., None]
            A = A.to(hidden_states.dtype) * dt

            # chunk each
            hidden_states, A, B, C = [
                reshape_into_chunks(t, pad_size, self.chunk_size)
                for t in (hidden_states, A, B, C)
            ]

            A = A.permute(0, 3, 1, 2)  # [bsz, nheads, #chunks, chunk_size]
            A_cumsum = torch.cumsum(A, dim=-1)

            # 1. Intra-chunk output (like "attention" inside the chunk)
            L = torch.exp(segment_sum(A))
            G_intermediate = (
                C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
            )  # shape: (b, c, l, s, h, n)
            G = G_intermediate.sum(dim=-1)  # shape: (b, c, l, s, h)
            M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
            M = M_intermediate.sum(dim=-1)
            Y_diag = (M[..., None] * hidden_states[:, :, None]).sum(dim=3)

            # 2. Compute local chunk states: no boundary pass
            decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
            B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
            states = (B_decay[..., None, :] * hidden_states[..., None]).sum(dim=2)

            # CHANGED: remove cross-chunk recurrence
            # ---------------------------------------
            # no previous_states or chunk-to-chunk carry
            # ---------------------------------------

            # 4. State -> output conversion within each chunk
            state_decay_out = torch.exp(A_cumsum)
            C_times_states = C[..., None, :] * states[:, :, None, ...]
            state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
            Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

            # Final chunk-local output
            y = Y_diag + Y_off

            # Add D-residual, remove chunk padding
            y = y + D_residual
            if pad_size > 0:
                y = y[:, :seq_len, :, :]

            # shape back
            y = y.reshape(batch_size, seq_len, -1)

            # CHANGED: skip updating ssm_state in the cache
            # ---------------------------------------
            # if ssm_state is not None and past_key_value_state is not None:
            #     past_key_value_state.ssm_state.copy_(ssm_state)

        # end chunk-based path
        # ---------------------------------------

        # 5. Gated RMSNorm and final linear projection
        scan_output = self.norm(y, gate)
        contextualized_states = self.out_proj(scan_output.to(dtype))

        return contextualized_states, past_key_value_state