"""Build the four model families used in the audit."""

from types import SimpleNamespace

import torch
from torch import nn

from models import gru as core
from models.almt import ALMT


class FP32MaskedGRU(core.MaskedGRU):
    def forward(self, x, keep, plen):
        # cuDNN RNNs and packed recurrent gradients stay outside fp16 autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            return super().forward(x.float(), keep.float(), plen)


def model_config(dims, lengths):
    return SimpleNamespace(model=SimpleNamespace(
        bert_pretrained='cached-bert', l_input_dim=dims['text'],
        a_input_dim=dims['audio'], v_input_dim=dims['vision'],
        l_proj_dst_dim=128, a_proj_dst_dim=128, v_proj_dst_dim=128,
        token_len=8, token_dim=128, l_input_length=lengths['text'],
        a_input_length=lengths['audio'], v_input_length=lengths['vision'],
        proj_input_dim=128, proj_depth=1, proj_heads=8, proj_mlp_dim=128,
        token_length=8, l_enc_heads=8, l_enc_mlp_dim=128,
        AHL_depth=3, ahl_heads=8, ahl_dim_head=16, ahl_droup=0.,
        fusion_heads=8, fusion_mlp_dim=128, fusion_layer_depth=4))


class ALMTFeature(nn.Module):
    def __init__(self, dims, lengths, fallback):
        super().__init__()
        self.net = ALMT(model_config(dims, lengths))
        self.fallback = float(fallback)

    def forward(self, batch):
        net = self.net
        summaries = {}
        available = {}
        for m, proj in (('text', net.proj_l), ('audio', net.proj_a), ('vision', net.proj_v)):
            keep = batch[f'keep_{m}'].bool()
            available[m] = keep.any(1)
            # torch.where also excludes nonfinite placeholder values at unobserved positions.
            x = torch.where(keep.unsqueeze(-1), batch[f'x_{m}'], 0.)
            h = proj[1](proj[0](x), keep)[:, :8]
            summaries[m] = h * available[m][:, None, None]
        layers = [h * available['text'][:, None, None] for h in net.l_encoder(summaries['text'])]
        hyper = net.h_hyper.expand(len(layers[0]), -1, -1)
        hyper = net.h_hyper_layer(layers, summaries['audio'], summaries['vision'], hyper)
        feat = net.fusion_layer(hyper, layers[-1])[:, 0]
        output = net.regression_layer(feat).squeeze(-1)
        any_input = available['text'] | available['audio'] | available['vision']
        return torch.where(any_input, output, torch.full_like(output, self.fallback))


def build_model(name, dims, lengths, fallback):
    if name == 'ALMT_FEATURE':
        return ALMTFeature(dims, lengths, fallback)
    if name not in ('B0_TEXT', 'B1_FUSION', 'B2_RATE'):
        raise ValueError(name)
    model = core.MSARegressor(name, dims, fallback_value=fallback)
    for module in model.enc.values():
        module.__class__ = FP32MaskedGRU
    return model
