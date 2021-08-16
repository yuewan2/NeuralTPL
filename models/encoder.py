import torch
import torch.nn as nn

import onmt
from onmt.encoders.encoder import EncoderBase
from onmt.modules.position_ffn import PositionwiseFeedForward


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout, attn):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = attn
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = onmt.modules.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        input_norm = self.layer_norm(inputs)
        context, attn = self.self_attn(input_norm, input_norm, input_norm,
                                       mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out), attn

class TransformerEncoder(EncoderBase):
    def __init__(self, num_layers, d_model, heads, d_ff,
                 dropout, embeddings, attn_modules):
        super(TransformerEncoder, self).__init__()

        self.num_layers = num_layers
        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout, attn_modules[i])
             for i in range(num_layers)])
        self.layer_norm = onmt.modules.LayerNorm(d_model)


    def forward(self, src, segments=None, masked_atom_mapping=None):
        if segments is None:
            emb = self.embeddings(src)
        else:
            emb = self.embeddings(src, masked_atom_mapping, segments)
        out = emb.transpose(0, 1).contiguous()

        words = src.transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1) \
            .expand(w_batch, w_len, w_len)

        # Run the forward pass of every layer of the transformer.
        for i in range(self.num_layers):
            out, attn = self.transformer[i](out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous()

