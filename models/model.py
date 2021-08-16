import torch
import torch.nn as nn
import numpy as np

import onmt
from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder
from models.embedding import Embedding
from models.module import Latent


class TemplateGenerator(nn.Module):
    def __init__(self, num_layers, d_model, heads, d_ff, dropout, 
        vocab_size_src, vocab_size_tgt, src_pad_idx, tgt_pad_idx):
        super(TemplateGenerator, self).__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.heads = heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.embedding_src = Embedding(vocab_size = vocab_size_src + 1, embed_size = d_model, padding_idx = src_pad_idx)
        self.embedding_tgt = Embedding(vocab_size = vocab_size_tgt + 1, embed_size = d_model, padding_idx = tgt_pad_idx)

        multihead_attn_modules_en = nn.ModuleList(
            [onmt.modules.MultiHeadedAttention(heads, d_model, dropout=dropout)
             for _ in range(num_layers)])

        multihead_attn_modules_de = nn.ModuleList(
            [onmt.modules.MultiHeadedAttention(heads, d_model, dropout=dropout)
             for _ in range(num_layers)])

        self.encoder = TransformerEncoder(num_layers=num_layers,
                             d_model=d_model, heads=heads,
                             d_ff=d_ff, dropout=dropout,
                             embeddings=self.embedding_src,
                             attn_modules=multihead_attn_modules_en)

        self.encoder_prior = TransformerEncoder(num_layers=num_layers,
                             d_model=d_model, heads=heads,
                             d_ff=d_ff, dropout=dropout,
                             embeddings=self.embedding_tgt,
                             attn_modules=multihead_attn_modules_de)

        self.decoder = TransformerDecoder(num_layers=num_layers,
                             d_model=d_model, heads=heads,
                             d_ff=d_ff, dropout=dropout,
                             embeddings=self.embedding_tgt,
                             self_attn_modules=multihead_attn_modules_de)

        self.latent_net = Latent(d_model)
        self.classifier = nn.Sequential(nn.Linear(d_model, 10),
                                        nn.LogSoftmax(dim=-1))
        self.generator = nn.Sequential(nn.Linear(d_model, vocab_size_tgt),
                                       nn.LogSoftmax(dim=-1))


    def forward(self, inputs, fixed_z=False, generalize=True):
        src, src_am, src_seg, tgt = inputs
        post_tgt = tgt.clone() # prior_tgt is use to train the prior net
        post_tgt[0] = 0 # index 0 by default is <CLS> token; if ver2, replace <sos> with <CLS>
        tgt = tgt[:-1]

        _, prior_encoder_out = self.encoder(src, src_seg, src_am)
        # _, prior_reverse_encoder_out = self.encoder(src_reverse, src_seg, src_am)

        _, post_encoder_out = self.encoder_prior(post_tgt) # template_embedding
        kld_loss, z = self.latent_net(prior_encoder_out[0], post_encoder_out[0], train=self.training, fixed_z=fixed_z)

        outputs, attn = self.decoder(src, tgt, prior_encoder_out, z)
        scores = self.generator(outputs)

        # outputs_reverse, _ = self.decoder(src_reverse, tgt, prior_reverse_encoder_out)
        # scores_reverse = self.generator(outputs_reverse)

        if generalize:
            rt_scores = self.classifier(post_encoder_out[0])
            return scores, rt_scores, kld_loss
        else:
            return scores, 0, kld_loss




        
            

