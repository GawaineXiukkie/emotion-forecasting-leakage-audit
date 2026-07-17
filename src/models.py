"""
Causal sequence models. Input [B,T,D] -> shift logits [B,T] (logit at t predicts shift t->t+1).
All are strictly causal: the logit at position t depends only on x_{<=t}. This IS the
anti-leakage boundary at the model level.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _sinusoidal_positions(length: int, dim: int, device, dtype) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype)
                    * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
    return pe


class CausalGRU(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(d_in, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)  # unidirectional => causal
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, spk=None) -> torch.Tensor:   # x: [B,T,D]; spk ignored
        h, _ = self.rnn(x)
        return self.head(h).squeeze(-1)                   # [B,T]


class CausalStateFactorizedForecaster(nn.Module):
    """Causal multi-task forecaster with explicit emotion-state factorization.

    The shared GRU sees only x_<=t. Three heads predict (i) current emotion,
    (ii) the target future emotion, and (iii) shift directly. A deployable
    factorized shift probability can therefore be formed as
    ``1 - sum_k p(y_t=k) p(y_target=k)`` without a gold current label.
    """
    def __init__(self, d_in: int, num_emotions: int, hidden: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(d_in, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.shift_head = nn.Linear(hidden, 1)
        self.current_head = nn.Linear(hidden, num_emotions)
        self.future_head = nn.Linear(hidden, num_emotions)

    def forward_all(self, x: torch.Tensor):
        h, _ = self.rnn(x)
        h = self.dropout(h)
        return (self.shift_head(h).squeeze(-1),
                self.current_head(h), self.future_head(h))

    def forward(self, x: torch.Tensor, spk=None) -> torch.Tensor:
        return self.forward_all(x)[0]


class LookaheadGRU(nn.Module):
    """P2.1 dose-response: causal GRU backbone + an explicit peek at the mean-pooled features
    of the next k utterances (k=0 -> no peek, structurally equivalent to CausalGRU with an
    always-zero extra input). Vectorized via cumsum (no python time-step loop -> fast on MPS).
    Isolates 'how many future utterances leak' as a single controlled knob."""
    def __init__(self, d_in: int, hidden: int = 128, k_lookahead: int = 0, dropout: float = 0.1):
        super().__init__()
        self.k = k_lookahead
        self.rnn = nn.GRU(d_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden + d_in, 1))

    def forward(self, x: torch.Tensor, spk=None) -> torch.Tensor:
        h, _ = self.rnn(x)                                   # [B,T,H], causal
        B, T, D = x.shape
        if self.k == 0:
            look = torch.zeros(B, T, D, device=x.device, dtype=x.dtype)
        else:
            csum = torch.cumsum(x, dim=1)                     # csum[:,i,:] = sum_{j<=i} x_j
            idx_hi = torch.clamp(torch.arange(T, device=x.device) + self.k, max=T - 1)
            hi = csum[:, idx_hi, :]
            diff = hi - csum                                  # sum_{t+1..min(t+k,T-1)} x
            count = (idx_hi - torch.arange(T, device=x.device)).clamp(min=1).float().view(1, T, 1)
            look = diff / count                                # mean-pooled future window
        return self.head(torch.cat([h, look], dim=-1)).squeeze(-1)


class LeakyBiGRU(nn.Module):
    """LEAKY by design: bidirectional GRU -> the state at position n sees future utterances
    (including the target n+1). Used ONLY to quantify how much leakage inflates metrics
    (Stage D headline). Never a legitimate baseline."""
    def __init__(self, d_in: int, hidden: int = 128, layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.rnn = nn.GRU(d_in, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, 1))

    def forward(self, x: torch.Tensor, spk=None) -> torch.Tensor:
        h, _ = self.rnn(x)
        return self.head(h).squeeze(-1)


class _CausalConv1d(nn.Module):
    """1D conv with left-only padding so output t sees inputs <= t."""
    def __init__(self, c_in, c_out, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)

    def forward(self, x):                                  # x: [B,C,T]
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class CausalTCN(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, levels: int = 4, k: int = 3, dropout: float = 0.1):
        super().__init__()
        blocks, c = [], d_in
        for i in range(levels):
            blocks += [_CausalConv1d(c, hidden, k, dilation=2 ** i), nn.ReLU(), nn.Dropout(dropout)]
            c = hidden
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, spk=None) -> torch.Tensor:   # x: [B,T,D]; spk ignored
        h = self.tcn(x.transpose(1, 2)).transpose(1, 2)   # [B,T,hidden]
        return self.head(h).squeeze(-1)


class CausalTransformer(nn.Module):
    """Optional contrast model. Subsequent mask => position t attends only to <= t."""
    def __init__(self, d_in: int, d_model: int = 128, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        enc = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=4 * d_model,
                                         dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, spk=None) -> torch.Tensor:
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.enc(self.proj(x), mask=mask)
        return self.head(h).squeeze(-1)


class FutureWindowTransformer(nn.Module):
    """One architecture with a single controlled future-attention window.

    ``k_future=0`` is causal; positive k exposes exactly the next k positions;
    ``k_future=None`` exposes the full dialogue. Parameters, positional encoding,
    pooling, and optimization are otherwise identical across conditions.
    """
    def __init__(self, d_in: int, hidden: int = 128, heads: int = 4, layers: int = 2,
                 dropout: float = 0.1, k_future: int | None = 0):
        super().__init__()
        self.k_future = k_future
        self.proj = nn.Linear(d_in, hidden)
        enc = nn.TransformerEncoderLayer(hidden, heads, dim_feedforward=4 * hidden,
                                         dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, spk=None) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.proj(x) + _sinusoidal_positions(T, self.proj.out_features,
                                                  x.device, x.dtype)[None]
        if self.k_future is None:
            mask = None
        else:
            ii = torch.arange(T, device=x.device)
            # query i may attend to key j only when j <= i + k.
            mask = ii[None, :] > (ii[:, None] + self.k_future)
        padding = x.abs().sum(dim=-1) == 0
        h = self.enc(h, mask=mask, src_key_padding_mask=padding)
        return self.head(h).squeeze(-1)


# P1.3: hidden size for CausalGRU chosen so param count (~912k) matches LeakyBiGRU(hidden=128,
# ~887k, within 3%) -- isolates the leakage effect from the capacity effect (bidirectional GRU
# has ~2x the parameters of a same-hidden unidirectional GRU).
CAPACITY_MATCHED_HIDDEN = 240


def build_model(name: str, d_in: int, **kw) -> nn.Module:
    from .models_families import CausalDialogueRNN, CausalDialogueGCN, CausalDAGERC
    from .models_efc_baselines import CausalPEC, CausalPseudoFuture
    if name == "gru_wide":
        return CausalGRU(d_in, hidden=CAPACITY_MATCHED_HIDDEN, **kw)
    if name.startswith("gru_look"):
        k = int(name[len("gru_look"):])
        return LookaheadGRU(d_in, k_lookahead=k, **kw)
    if name.startswith("transformer_future"):
        suffix = name[len("transformer_future"):]
        k = None if suffix == "all" else int(suffix)
        return FutureWindowTransformer(d_in, k_future=k, **kw)
    if name == "pec_fixed":
        return CausalPEC(d_in, corrected_recency=True, **kw)
    if name == "pseudofuture_fixed":
        return CausalPseudoFuture(d_in, mask_aux_padding=True, **kw)
    return {"gru": CausalGRU, "tcn": CausalTCN, "transformer": CausalTransformer,
            "gru_leaky": LeakyBiGRU, "dialoguernn": CausalDialogueRNN,
            "dialoguegcn": CausalDialogueGCN, "dagerc": CausalDAGERC,
            "pec": CausalPEC, "pseudofuture": CausalPseudoFuture}[name](d_in, **kw)
