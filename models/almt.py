# Upstream MIT license: vendor/almt/LICENSE. Fixed source hashes in provenance.json.
'''
* @name: almt_layer.py
* @description: Basic layer of ALMT. Note: Some code references Vision Transformer (VIT, https://github.com/gupta-abhay/pytorch-vit).
'''

import torch
from torch import nn, einsum


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNormForward(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormAttention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_k = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, q, k, v, mask=None, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v, mask=mask)


class PreNormAHL(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, h_t, h_a, h_v, h_hyper):
        h_t = self.norm1(h_t)
        h_a = self.norm2(h_a)
        h_v = self.norm3(h_v)
        h_hyper = self.norm4(h_hyper)

        return self.fn(h_t, h_a, h_v, h_hyper)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v, mask=None):
        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if mask is not None:
            if mask.shape != (b, k.shape[2]) or not mask.any(dim=1).all():
                raise ValueError('attention key mask has wrong shape or no valid key')
            dots = dots.float().masked_fill(~mask[:, None, None, :], torch.finfo(torch.float32).min)

        attn = self.attend(dots.float())

        out = einsum('b h i j, b h j d -> b h i d', attn, v.float())
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class HhyperLearningLayer(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_tv = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_tv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=True),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, h_t, h_a, h_v, h_hyper):
        b, n, _, h = *h_t.shape, self.heads

        q = self.to_q(h_t)
        k_ta = self.to_k_ta(h_a)
        k_tv = self.to_k_tv(h_v)
        v_ta = self.to_v_ta(h_a)
        v_tv = self.to_v_tv(h_v)

        q, k_ta, k_tv, v_ta, v_tv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k_ta, k_tv, v_ta, v_tv))

        dots_ta = einsum('b h i d, b h j d -> b h i j', q, k_ta) * self.scale
        attn_ta = self.attend(dots_ta.float())
        out_ta = einsum('b h i j, b h j d -> b h i d', attn_ta, v_ta.float())
        out_ta = rearrange(out_ta, 'b h n d -> b n (h d)')

        dots_tv = einsum('b h i d, b h j d -> b h i j', q, k_tv) * self.scale
        attn_tv = self.attend(dots_tv.float())
        out_tv = einsum('b h i j, b h j d -> b h i d', attn_tv, v_tv.float())
        out_tv = rearrange(out_tv, 'b h n d -> b n (h d)')

        h_hyper_shift = self.to_out(out_ta + out_tv)
        h_hyper = h_hyper + h_hyper_shift

        return h_hyper


class HhyperLearningEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAHL(dim, HhyperLearningLayer(dim, heads = heads, dim_head = dim_head, dropout = dropout))
            ]))

    def forward(self, h_t_list, h_a, h_v, h_hyper):
        for i, attn in enumerate(self.layers):
            h_hyper = attn[0](h_t_list[i], h_a, h_v, h_hyper)
        return h_hyper


class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, save_hidden=False, mask=None):
        if save_hidden == True:
            hidden_list = []
            hidden_list.append(x)
            for attn, ff in self.layers:
                x = attn(x, x, x, mask=mask) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x, mask=mask) + x
                x = ff(x) + x
            return x


class CrossTransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormAttention(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNormForward(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, source_x, target_x):
        for attn, ff in self.layers:
            target_x_tmp = attn(target_x, source_x, source_x)
            target_x = target_x_tmp + target_x
            target_x = ff(target_x) + target_x
        return target_x


class Transformer(nn.Module):
    def __init__(self, *, num_frames, token_len, save_hidden, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.token_len = token_len
        self.save_hidden = save_hidden

        if token_len is not None:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, dim))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, dim))
        else:
             self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
             self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()


    def forward(self, x, mask=None):
        b, n, _ = x.shape

        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b = b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, :n+self.token_len]
            if mask is not None:
                mask = torch.cat((torch.ones((b, self.token_len), device=x.device, dtype=torch.bool), mask), dim=1)
        else:
            x = x + self.pos_embedding[:, :n]

        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden, mask=mask)

        return x


class CrossTransformer(nn.Module):
    def __init__(self, *, source_num_frames, tgt_num_frames, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames + 1, dim))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames + 1, dim))
        self.extra_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.CrossTransformerEncoder = CrossTransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

    def forward(self, source_x, target_x):
        b, n_s, _ = source_x.shape
        b, n_t, _ = target_x.shape

        extra_token = repeat(self.extra_token, '1 1 d -> b 1 d', b = b)

        source_x = torch.cat((extra_token, source_x), dim=1)
        source_x = source_x + self.pos_embedding_s[:, : n_s+1]

        target_x = torch.cat((extra_token, target_x), dim=1)
        target_x = target_x + self.pos_embedding_t[:, : n_t+1]

        source_x = self.dropout(source_x)
        target_x = self.dropout(target_x)

        x_s2t = self.CrossTransformerEncoder(source_x, target_x)

        return x_s2t


def repeat(x, pattern, *, b):
    if pattern not in ('1 n d -> b n d', '1 1 d -> b 1 d'):
        raise ValueError(pattern)
    return x.expand(b, -1, -1)

def rearrange(x, pattern, *, h=None):
    if pattern == 'b n (h d) -> b h n d':
        return x.reshape(x.shape[0], x.shape[1], h, -1).permute(0, 2, 1, 3)
    if pattern == 'b h n d -> b n (h d)':
        return x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)
    raise ValueError(pattern)

class BertTextEncoder(nn.Identity):
    # Cache enters exactly at the author's BERT output interface.
    def __init__(self, *args, **kwargs):
        super().__init__()

import torch
from torch import nn


class ALMT(nn.Module):
    def __init__(self, args):
        super(ALMT, self).__init__()

        args = args.model

        self.h_hyper = nn.Parameter(torch.ones(1, args.token_len, args.token_dim))

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args.bert_pretrained)

        self.proj_l = nn.Sequential(
            nn.Linear(args.l_input_dim, args.l_proj_dst_dim),
            Transformer(num_frames=args.l_input_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
        )
        self.proj_a = nn.Sequential(
            nn.Linear(args.a_input_dim, args.a_proj_dst_dim),
            Transformer(num_frames=args.a_input_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
        )
        self.proj_v = nn.Sequential(
            nn.Linear(args.v_input_dim, args.v_proj_dst_dim),
            Transformer(num_frames=args.v_input_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
        )

        self.l_encoder = Transformer(num_frames=args.token_length, save_hidden=True, token_len=None, dim=args.proj_input_dim, depth=args.AHL_depth-1, heads=args.l_enc_heads, mlp_dim=args.l_enc_mlp_dim)
        self.h_hyper_layer = HhyperLearningEncoder(dim=args.token_dim, depth=args.AHL_depth, heads=args.ahl_heads, dim_head=args.ahl_dim_head, dropout=args.ahl_droup)
        self.fusion_layer = CrossTransformer(source_num_frames=args.token_len, tgt_num_frames=args.token_len, dim=args.proj_input_dim, depth=args.fusion_layer_depth, heads=args.fusion_heads, mlp_dim=args.fusion_mlp_dim)

        self.regression_layer = nn.Sequential(
            nn.Linear(args.token_dim, 1)
        )

    def forward(self, x_visual, x_audio, x_text, mask_visual=None, mask_audio=None, mask_text=None):
        b = x_visual.size(0)

        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b=b)

        x_text = self.bertmodel(x_text)

        h_v = self.proj_v[1](self.proj_v[0](x_visual), mask_visual)[:, :self.h_hyper.shape[1]]
        h_a = self.proj_a[1](self.proj_a[0](x_audio), mask_audio)[:, :self.h_hyper.shape[1]]
        h_l = self.proj_l[1](self.proj_l[0](x_text), mask_text)[:, :self.h_hyper.shape[1]]

        h_t_list = self.l_encoder(h_l)
        h_hyper = self.h_hyper_layer(h_t_list, h_a, h_v, h_hyper)
        feat = self.fusion_layer(h_hyper, h_t_list[-1])[:, 0]

        output = self.regression_layer(feat)

        return output


def build_model(args):
    model = ALMT(args)

    return model


