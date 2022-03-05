"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import layers_char_embed
import layers_qanet
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.): # added the char_vectors here just to make the init function same in both models
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs): # last two parameters are unused here at the moment
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class BiDAFWithChar(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained char vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.):
        super(BiDAFWithChar, self).__init__()

        self.char_embed_size = 100
        self.emb = layers_char_embed.BiDAFWordPlusCharEmbedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob,
                                    num_filters=self.char_embed_size)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        # to do : determine what do about the masking for the character embedding
        # answer: nothing needs to be done here

        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class QANet(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        char_vectors (torch.Tensor): Pre-trained char vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, drop_prob=0.):
        super(QANet, self).__init__()
        # self.word_embed_size = word_vectors.size(1)

        self.emb = layers_qanet.QANetEmbedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    drop_prob=drop_prob,
                                    num_filters=100)

        self.initial_embed_dim = self.emb.GetOutputEmbeddingDim()
        self.d_model = 128 # d model is the dimensionality of each word before and after it goes into the encoder layer, i
        self.num_conv_filters = 128

        # These two layers will reduce the dimensionality of the embedding of each word from (500) to (128)
        self.context_conv = layers_qanet.DepthwiseSeparableConv(
            in_channels = self.initial_embed_dim, 
            out_channels= self.d_model,
            kernel_size=5)
        
        self.question_conv = layers_qanet.DepthwiseSeparableConv(
            in_channels = self.initial_embed_dim, 
            out_channels = self.d_model,
            kernel_size=5)

        # Output of the Convolutions above will be fed into the encoder
        self.embedding_encoder_context =  layers_qanet.Encoder(d_model=self.d_model,
                                                                num_filters=self.num_conv_filters, 
                                                                kernel_size=7, 
                                                                num_conv_layers=4, 
                                                                num_heads=8, 
                                                                drop_prob=drop_prob)

        self.embedding_encoder_question =  layers_qanet.Encoder(d_model=self.d_model,
                                                                num_filters=self.num_conv_filters, 
                                                                kernel_size=7, 
                                                                num_conv_layers=4, 
                                                                num_heads=8, 
                                                                drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=self.d_model, drop_prob=drop_prob)

        self.att_conv = layers_qanet.DepthwiseSeparableConv(
            in_channels = self.d_model*4, 
            out_channels = self.d_model,
            kernel_size=5)

        self.model_encoders =  nn.ModuleList([layers_qanet.Encoder(d_model=self.d_model,
                                                                num_filters=self.num_conv_filters,
                                                                kernel_size=5,
                                                                num_conv_layers=2,
                                                                num_heads=8,
                                                                drop_prob=drop_prob) for _ in range(5)])
        

        self.out = layers_qanet.QANetOutput(d_model=self.d_model, drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        batch_size = cw_idxs.shape[0]

        # In QANet the projection is not applied and output of highway network is same size as the word+char embedding dim
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, self.initial_embed_dim)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, self.initial_embed_dim)

        # print("context embedding shape", c_emb.shape)
        # print("query embedding shape", q_emb.shape)

        c_emb = self.context_conv(c_emb.transpose(1,2)).transpose(1,2) # (batch_size, self.num_conv_filters, c_len)
        q_emb = self.question_conv(q_emb.transpose(1,2)).transpose(1,2) # (batch_size, self.num_conv_filters, q_len)

        c_enc = self.embedding_encoder_context(c_emb)
        q_enc = self.embedding_encoder_question(q_emb)

        # print("context encoding", c_enc[0][5][:10])
        # print("query encoding shape",q_enc.shape)

        # assert(c_enc.shape == (batch_size, c_len, self.hidden_size))
        # assert(q_enc.shape == (batch_size, q_len, self.hidden_size))

        # compute attention same as BiDAF
        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 4 * d_model)

        # print("context-query attention shape", att.shape)
        # assert(att.shape == (batch_size, c_len, 8 * self.d_model))
        m0 = self.att_conv(att.transpose(1,2)).transpose(1,2)
        
        print("m0", m0[0][5][:10])
        for i, enc in enumerate(self.model_encoders):
            m0 = enc(m0)
        m1 = m0

        for i, enc in enumerate(self.model_encoders):
            m0 = enc(m0)
        m2 = m0

        for i, enc in enumerate(self.model_encoders):
            m0 = enc(m0)
        m3 = m0

        out = self.out(m1, m2, m3, c_mask)

        return out

class QANetConfig:
    def __init__(self) -> None:
        pass
