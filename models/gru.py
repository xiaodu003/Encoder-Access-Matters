"""Text-only, concatenation and rate-gated recurrent predictors."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data.features import MODALITIES

HIDDEN_DIM = 64


DROPOUT = 0.1


GATE_HIDDEN = 32


class MaskedGRU(nn.Module):
    def __init__(self, d, h=HIDDEN_DIM, drop=DROPOUT):
        super().__init__()
        # Published plan: project features before appending observed-position flag.
        self.proj = nn.Linear(d, h)
        self.drop = nn.Dropout(drop)
        self.gru = nn.GRU(h + 1, h, batch_first=True)

    def forward(self, x, keep, plen):
        z = torch.relu(self.proj(x * keep.unsqueeze(-1)))
        z = torch.cat([z, keep.unsqueeze(-1)], dim=-1)
        lens = plen.clamp(min=1, max=x.shape[1]).detach().cpu()
        packed = pack_padded_sequence(z, lens, batch_first=True, enforce_sorted=False)
        po, _ = self.gru(packed)
        out, _ = pad_packed_sequence(po, batch_first=True, total_length=x.shape[1])
        w = keep.unsqueeze(-1)
        pooled = (out * w).sum(1) / w.sum(1).clamp_min(1.0)
        return self.drop(pooled * (w.sum(1) > 0).float())


class MSARegressor(nn.Module):
    def __init__(self, name, dims, fallback_value=0.0):
        super().__init__()
        self.name = name
        self.fallback = float(fallback_value)
        mods = (
            ("text",)
            if name == "B0_TEXT"
            else MODALITIES
        )
        self.enc = nn.ModuleDict({m: MaskedGRU(dims[m]) for m in mods})
        if name == "B0_TEXT":
            self.head = nn.Sequential(
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM, 1),
            )
        elif name == "B1_FUSION":
            self.head = nn.Sequential(
                nn.Linear(3 * HIDDEN_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM, 1),
            )
        elif name == "B2_RATE":
            self.gate = nn.Sequential(
                nn.Linear(3 * HIDDEN_DIM + 9, GATE_HIDDEN),
                nn.ReLU(),
                nn.Linear(GATE_HIDDEN, 3),
            )
            self.head = nn.Sequential(
                nn.Linear(3 * HIDDEN_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM, 1),
            )
        else:
            raise ValueError(name)

    def forward(self, b, return_aux=False):
        h = {
            m: self.enc[m](b["x_" + m], b["keep_" + m], b["plen_" + m])
            for m in self.enc
        }
        if self.name == "B0_TEXT":
            out = self.head(h["text"]).squeeze(-1)
            unavail = (
                b["keep_text"].sum(1) == 0
            )  # doc 5.3/7.2: text unavailable -> constant output
            if bool(unavail.any()):
                out = torch.where(unavail, torch.full_like(out, self.fallback), out)
            if return_aux:
                return out, torch.full((len(out), 3), float("nan"), device=out.device)
            return out
        hs = torch.stack([h[m] for m in MODALITIES], dim=1)
        if self.name == "B1_FUSION":
            out = self.head(hs.reshape(len(hs), -1)).squeeze(-1)
            missing = torch.stack(
                [b["keep_" + m].sum(1) == 0 for m in MODALITIES], dim=1
            ).all(1)
            out = torch.where(missing, torch.full_like(out, self.fallback), out)
            if return_aux:
                return out, torch.full((len(out), 3), float("nan"), device=out.device)
            return out
        rates, available = [], []
        for m in MODALITIES:
            k = b["keep_" + m]
            v = k.sum(1) / b["vlen_" + m].clamp_min(1.0)
            rates.append(v)
            available.append(k.sum(1) > 0)
        v = torch.stack(rates, dim=1)
        stats = torch.stack(
            [v, torch.zeros_like(v), torch.zeros_like(v)], dim=-1
        ).reshape(
            len(v), -1
        )  # v/0/0
        logits = self.gate(torch.cat([hs.reshape(len(hs), -1), stats], dim=1))
        avail = torch.stack(available, dim=1)
        logits = logits.masked_fill(~avail, -1e4)
        alpha = torch.softmax(logits, dim=1)
        out = self.head((hs * alpha.unsqueeze(-1)).reshape(len(hs), -1)).squeeze(-1)
        all_unavail = (~avail).all(dim=1)
        if bool(all_unavail.any()):
            out = torch.where(all_unavail, torch.full_like(out, self.fallback), out)
        if return_aux:
            alpha = torch.where(
                all_unavail[:, None], torch.full_like(alpha, float("nan")), alpha
            )
            return out, alpha
        return out
