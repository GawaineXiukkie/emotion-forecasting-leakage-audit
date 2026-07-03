"""
Causal re-implementations of three ERC method families, adapted to target-absent shift
forecasting under the leakage-safe protocol.

These are not the original repos (which are bidirectional ERC and would leak). Each captures
the family's defining inductive bias, made strictly causal (position t uses only x_<=t, spk_<=t):
  - CausalDialogueRNN : speaker-state recurrence (global + per-party + emotion GRU cells).
  - CausalDialogueGCN : windowed, speaker-relational graph convolution over past utterances.
  - CausalDAGERC      : directed (past-only) speaker-relational attention along the conversation.

All: forward(x [B,T,D], spk [B,T] long) -> shift logits [B,T].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_PARTIES = 9


class CausalDialogueRNN(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, parties: int = MAX_PARTIES, dropout: float = 0.1):
        super().__init__()
        self.H, self.P = hidden, parties
        self.g_cell = nn.GRUCell(d_in + hidden, hidden)   # global: utterance + current party state
        self.p_cell = nn.GRUCell(d_in + hidden, hidden)   # party: utterance + global context
        self.e_cell = nn.GRUCell(hidden, hidden)          # emotion: party state
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        dev = x.device
        g = torch.zeros(B, self.H, device=dev)
        e = torch.zeros(B, self.H, device=dev)
        party = torch.zeros(B, self.P, self.H, device=dev)
        ar = torch.arange(B, device=dev)
        outs = []
        for t in range(T):
            x_t = x[:, t, :]
            s_t = spk[:, t].clamp(0, self.P - 1)
            cur = party[ar, s_t]                                  # [B,H] current speaker state
            g = self.g_cell(torch.cat([x_t, cur], -1), g)        # update global (uses past only)
            new_p = self.p_cell(torch.cat([x_t, g], -1), cur)    # update speaking party
            party = party.clone()
            party[ar, s_t] = new_p
            e = self.e_cell(new_p, e)                             # emotion state
            outs.append(self.head(e))
        return torch.cat(outs, dim=1)                             # [B,T]


def _relation_masks(spk: torch.Tensor, window: int):
    """Return row-normalized same/diff-speaker causal-windowed adjacency [B,T,T]."""
    B, T = spk.shape
    dev = spk.device
    ii = torch.arange(T, device=dev)
    causal = (ii[None, :] <= ii[:, None]) & (ii[:, None] - ii[None, :] <= window)  # [T,T]
    base = causal[None].expand(B, T, T)
    eq = spk[:, :, None] == spk[:, None, :]                                         # [B,T,T]
    same = (eq & base).float()
    diff = ((~eq) & base).float()
    same = same / (same.sum(-1, keepdim=True) + 1e-6)
    diff = diff / (diff.sum(-1, keepdim=True) + 1e-6)
    return same, diff


class CausalDialogueGCN(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, layers: int = 2, window: int = 6, dropout: float = 0.1):
        super().__init__()
        self.window = window
        self.proj = nn.Linear(d_in, hidden)
        self.W_self = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.W_same = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.W_diff = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        same, diff = _relation_masks(spk, self.window)
        h = F.relu(self.proj(x))
        for ws, wsa, wdf in zip(self.W_self, self.W_same, self.W_diff):
            agg = torch.bmm(same, wsa(h)) + torch.bmm(diff, wdf(h))   # relational, causal
            h = self.drop(F.relu(ws(h) + agg))
        return self.head(h).squeeze(-1)


class CausalDAGERC(nn.Module):
    def __init__(self, d_in: int, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.H = hidden
        self.proj = nn.Linear(d_in, hidden)
        self.q = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.k = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.v = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.o = nn.ModuleList(nn.Linear(2 * hidden, hidden) for _ in range(layers))
        self.b_same = nn.Parameter(torch.zeros(layers))
        self.b_diff = nn.Parameter(torch.zeros(layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        dev = x.device
        ii = torch.arange(T, device=dev)
        causal = (ii[None, :] <= ii[:, None])                         # [T,T] past-only (incl self)
        eq = spk[:, :, None] == spk[:, None, :]                       # [B,T,T]
        h = F.relu(self.proj(x))
        for li in range(len(self.q)):
            scores = torch.bmm(self.q[li](h), self.k[li](h).transpose(1, 2)) / (self.H ** 0.5)
            scores = scores + torch.where(eq, self.b_same[li], self.b_diff[li])   # relational bias
            scores = scores.masked_fill(~causal[None], float("-inf"))
            ctx = torch.bmm(torch.softmax(scores, dim=-1), self.v[li](h))
            h = self.drop(F.relu(self.o[li](torch.cat([h, ctx], -1))))
        return self.head(h).squeeze(-1)
