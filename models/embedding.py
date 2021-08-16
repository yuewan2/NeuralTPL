import math

import torch
import torch.nn as nn

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size)

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)

class AtomMappingEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(100, embed_size, padding_idx=0)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(0)].transpose(0, 1)
    

class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.1, padding_idx=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.atom_map = AtomMappingEmbedding(embed_size=self.token.embedding_dim)
        self.atom_map.require_grad = False
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.word_padding_idx = padding_idx

    def forward(self, sequence, atom_mapping=None, segment_label=None, inner_padding=False, step=None):
        x = self.token(sequence)

        # TODO, If 中间有padding:
        if inner_padding:
            positional_embedding = torch.zeros(sequence.shape[0], sequence.shape[1], 256).to(sequence.device)
            for i in range(sequence.shape[1]):
                prev_sep = False
                pad_num = 0
                for j in range(sequence.shape[0]):
                    if sequence[j, i].item() == 13:
                        positional_embedding[j, i] = self.position.pe[0, j - pad_num]
                        prev_sep = True
                        continue
                    if prev_sep and sequence[j, i].item() == 1:
                        pad_num += 1
                        continue
                    if prev_sep and sequence[j, i].item() != 1:
                        prev_sep = False

                    positional_embedding[j, i] = self.position.pe[0, j - pad_num]

            output = x + positional_embedding
        else:
            x += self.position(sequence)

            if segment_label is not None and sequence.size(0) < segment_label.size(0):
                segment_label = segment_label[:sequence.size(0)]

            if segment_label is not None:
                x += self.segment(segment_label)

            if atom_mapping is not None:
                x += self.atom_map(atom_mapping)

            output = x

        if step is None:
            return self.dropout(output)
        else:
            return self.dropout(output)[step].unsqueeze(0)