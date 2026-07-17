"""
Two causal re-implementations of emotion-forecasting-specific strategies, built from the
strategy descriptions in the literature rather than the original authors' code (which isn't
available), so these are "-style" re-implementations and not claimed to be architecturally
identical to any specific published system.

  CausalPEC          : sequence + self-dependency + recency decomposition, in the spirit of
                       Altarawneh et al. (2023)'s PEC framing.
  CausalPseudoFuture  : two-stage predict-then-classify, in the spirit of the pseudo-future
                       strategy used by NSF/PUGCN-family approaches.

Both expose forward(x, spk) -> logits [B, T], the same interface as every other model in the
harness (src/models.py, src/models_families.py), so they plug into src/train.py unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_PARTIES = 9


class CausalPEC(nn.Module):
    """Three components, concatenated into one discriminative head:
      - sequence:       causal GRU over the whole dialogue.
      - self-dependency: a per-speaker GRUCell state, updated only on that speaker's own
                         prior utterances (their history, independent of other speakers).
      - recency:         exponentially-decayed weighted average of the last `recency_m`
                         raw utterance vectors (speaker-agnostic).

    If the self-dependency term's continuous GRUCell state were collapsed down to a discrete
    "last own emotion" lookup, it would reduce to exactly the transition matrix's
    conditioning variable (current label, current speaker) -- the transition matrix baseline
    is effectively a degenerate special case of this term.
    """
    def __init__(self, d_in: int, hidden: int = 128, parties: int = MAX_PARTIES,
                recency_m: int = 5, recency_decay: float = 0.7, dropout: float = 0.1,
                corrected_recency: bool = False):
        super().__init__()
        self.H, self.P, self.M, self.decay = hidden, parties, recency_m, recency_decay
        self.corrected_recency = corrected_recency
        self.seq_rnn = nn.GRU(d_in, hidden, batch_first=True)
        self.self_cell = nn.GRUCell(d_in, hidden)
        self.recency_proj = nn.Linear(d_in, hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(3 * hidden, 1))

    def forward(self, x: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        dev = x.device

        # --- sequence ---
        h_seq, _ = self.seq_rnn(x)                                   # [B,T,H], causal

        # --- self-dependency: per-party GRUCell, state BEFORE the current utterance ---
        party = torch.zeros(B, self.P, self.H, device=dev, dtype=x.dtype)
        ar = torch.arange(B, device=dev)
        self_states = []
        for t in range(T):
            s_t = spk[:, t].clamp(0, self.P - 1)
            self_states.append(party[ar, s_t])          # self-history strictly BEFORE t
            new_p = self.self_cell(x[:, t, :], party[ar, s_t])
            party = party.clone()
            party[ar, s_t] = new_p
        h_self = torch.stack(self_states, dim=1)                     # [B,T,H]

        # --- recency: causal exponentially-weighted window over the last M utterances ---
        # Legacy experiments used exponents 0..M-1 over an oldest->newest window,
        # inadvertently giving the oldest item the largest weight.  Keep that path
        # reproducible under model name ``pec``; ``pec_fixed`` uses the intended
        # recent-first exponential decay (newest exponent 0, hence weight 1).
        if self.corrected_recency:
            exponent = torch.arange(self.M - 1, -1, -1, device=dev, dtype=x.dtype)
        else:
            exponent = torch.arange(self.M, device=dev, dtype=x.dtype)
        w = self.decay ** exponent
        w = w / w.sum()
        pad = torch.zeros(B, self.M - 1, D, device=dev, dtype=x.dtype)
        xpad = torch.cat([pad, x], dim=1)                            # left pad only -> causal
        windows = xpad.unfold(1, self.M, 1).permute(0, 1, 3, 2)       # [B,T,M,D]
        recency_raw = torch.einsum("m,btmd->btd", w, windows)
        h_recency = F.relu(self.recency_proj(recency_raw))

        return self.head(torch.cat([h_seq, h_self, h_recency], dim=-1)).squeeze(-1)


class CausalPseudoFuture(nn.Module):
    """Two-stage: a causal encoder first regresses a predicted embedding of utterance n+1 from
    history x_<=n. The real x_{n+1} is used only as a regression target during training, never
    as a model input at train or test time -- the same relationship the shift label itself has
    to future data. The model then classifies shift from the causal hidden state concatenated
    with its own predicted future embedding; at inference, no real future data enters the
    input path at any point."""
    def __init__(self, d_in: int, hidden: int = 128, dropout: float = 0.1,
                aux_weight: float = 0.1, mask_aux_padding: bool = False):
        super().__init__()
        self.enc = nn.GRU(d_in, hidden, batch_first=True)
        self.predict_head = nn.Linear(hidden, d_in)
        self.cls_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden + d_in, 1))
        self.aux_weight = aux_weight
        self.mask_aux_padding = mask_aux_padding
        self._last = None   # (pred, target) cached for aux_loss(), consumed by train.py

    def forward(self, x: torch.Tensor, spk=None) -> torch.Tensor:
        h, _ = self.enc(x)                     # [B,T,H], causal
        x_hat = self.predict_head(h)           # [B,T,D] predicted embedding of t+1, from h_t
        logits = self.cls_head(torch.cat([h, x_hat], dim=-1)).squeeze(-1)
        B, T, D = x.shape
        if T >= 2:
            self._last = (x_hat[:, :-1, :], x[:, 1:, :].detach())   # (pred, real-future target)
        else:
            self._last = None
        return logits

    def aux_loss(self, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Auxiliary regression loss: MSE between the predicted and real embedding of t+1.
        The real embedding is used only as a label here, never fed into the forward pass."""
        if self._last is None:
            return torch.tensor(0.0)
        pred, target = self._last
        if self.mask_aux_padding and valid_mask is not None:
            if not bool(valid_mask.any()):
                return pred.sum() * 0.0
            pred = pred[valid_mask]
            target = target[valid_mask]
        return self.aux_weight * F.mse_loss(pred, target)
