"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import layers_char_embed
import layers_qanet
import layers_qanet1
import layers1
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

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
        self.d_model = 128 # d model is the dimensionality of each word before and after it goes into the encoder layer, i
        self.num_conv_filters = 128

        self.emb = layers_qanet.QANetEmbedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    drop_prob=drop_prob,
                                    d_model= self.d_model)

        # self.initial_embed_dim = self.emb.GetOutputEmbeddingDim()
        

        # These two layers will reduce the dimensionality of the embedding of each word from (500) to (128)
        # self.context_conv = layers_qanet.DepthwiseSeparableConv(
        #     in_channels = self.initial_embed_dim, 
        #     out_channels= self.d_model,
        #     kernel_size=5)
        
        # self.question_conv = layers_qanet.DepthwiseSeparableConv(
        #     in_channels = self.initial_embed_dim, 
        #     out_channels = self.d_model,
        #     kernel_size=5)

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
        c_emb = F.dropout(c_emb, self.drop_prob, self.training)
        q_emb = F.dropout(q_emb, self.drop_prob, self.training) 
        #c_emb = self.context_conv(c_emb.transpose(1,2)).transpose(1,2) # (batch_size, self.num_conv_filters, c_len)
        #q_emb = self.question_conv(q_emb.transpose(1,2)).transpose(1,2) # (batch_size, self.num_conv_filters, q_len)

        c_enc = F.relu(self.embedding_encoder_context(c_emb))
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
        
        #print("m0", m0[0][5][:10])
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
        
class QANet1(nn.Module):
    def __init__(self, word_vectors, hidden_size, char_vocab_size , char_emb_size, word_char_emb_size,  drop_prob,
                num_blocks_embd , num_conv_embd , kernel_size , num_heads , num_blocks_model , num_conv_model,
                dropout_char, dropout_word, survival_prob):
        super(QANet1, self).__init__()
        # self.word_embed_size = word_vectors.size(1)
        self.d_model = hidden_size # d model is the dimensionality of each word before and after it goes into the encoder layer, i
        self.num_conv_filters = hidden_size
        self.drop_prob = drop_prob

        self.emb = layers1.Embedding(word_vectors=word_vectors,
                                    char_vocab_size=char_vocab_size,
                                    word_emb_size= word_char_emb_size,
                                    char_emb_size= char_emb_size,
                                    drop_prob_char=dropout_char,
                                    drop_prob_word = dropout_word)
                        
        self.embedding_projection = nn.Linear(500, self.d_model)
        # self.question_conv = layers_qanet1.DepthwiseSeparableConv(500,self.d_model, 5)
        self.emb_enc = layers1.Embedding_Encoder(num_blocks=num_blocks_embd,
                                                 num_conv = num_conv_embd,
                                                  kernel_size = kernel_size,
                                                  hidden_size = hidden_size,
                                                  num_heads = num_heads,
                                                  survival_prob= survival_prob)

        # self.c_emb_enc = layers_qanet1.EncoderBlock(conv_num=4, ch_num=self.d_model, k=7)
        # self.q_emb_enc = layers_qanet1.EncoderBlock(conv_num=4, ch_num=self.d_model, k=7)
        #self.cq_att = layers_qanet1.CQAttention()
        self.att = layers.BiDAFAttention(hidden_size=self.d_model, drop_prob=drop_prob)
        self.att_resizer = nn.Linear(self.d_model * 4, self.d_model)

        self.mod = layers1.Model_Encoder(num_blocks = num_blocks_model, 
                                        num_conv = num_conv_model, 
                                        kernel_size=kernel_size, 
                                        hidden_size = self.d_model, 
                                        num_heads = num_heads, 
                                        survival_prob= survival_prob)
        
        # enc_blk = layers_qanet1.EncoderBlock(conv_num=2, ch_num=self.d_model, k=5)
        # self.model_enc_blks = nn.ModuleList([enc_blk] * 7)
        self.out = layers_qanet.QANetOutput(d_model=self.d_model)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        start_time = time.time()
        cmask = torch.zeros_like(cw_idxs) != cw_idxs
        qmask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = cmask.sum(-1), qmask.sum(-1)
        batch_size = cw_idxs.shape[0]

        # Cw, Cc = self.word_emb(cw_idxs), self.char_emb(cc_idxs)
        # Qw, Qc = self.word_emb(qw_idxs), self.char_emb(qw_idxs)
        # C, Q = self.emb(Cc, Cw), self.emb(Qc, Qw)
        # c_emb = self.context_conv(C)  
        # q_emb = self.question_conv(Q)  
        
        emb_st = time.time()
        # In QANet the projection is not applied and output of highway network is same size as the word+char embedding dim
        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, self.initial_embed_dim)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, self.initial_embed_dim)

        c_emb = F.dropout(c_emb, self.drop_prob, self.training)
        q_emb = F.dropout(q_emb, self.drop_prob, self.training)

        #print('c_emb before projection is', c_emb[0:5])
        c_emb = F.relu(self.embedding_projection(c_emb))
        q_emb = F.relu(self.embedding_projection(q_emb))
        #print('c_emb after projection is', c_emb[0:5])
        
        #c_emb = c_emb.transpose(1,2)
        #q_emb = q_emb.transpose(1,2)
        
        c_enc, q_enc = self.emb_enc(c_emb, q_emb, cmask, qmask)
        c_enc = F.dropout(c_enc, self.drop_prob, self.training)
        q_enc = F.dropout(q_enc, self.drop_prob, self.training)
        emb_end = time.time()

        #Ce = self.c_emb_enc(c_emb, cmask)
        # print('Mean of context encdoing is', torch.mean(Ce))
        #print('c_enc is', Ce.transpose(1,2)[0:5])

        #Qe = self.q_emb_enc(q_emb, qmask)
        # Ce = Ce.transpose(1,2)
        # Qe = Qe.transpose(1,2)
        att = self.att(c_enc, q_enc, cmask, qmask)

        att = F.dropout(att, self.drop_prob, self.training)
        
        att = F.relu(self.att_resizer(att))

        att_end = time.time()

        # att = att.transpose(1,2)
        mod1, mod2, mod3 = self.mod(att, cmask)

        mod_end = time.time()

        # for enc in self.model_enc_blks: M1 = enc(M1, cmask)
        # M2 = M1
        # for enc in self.model_enc_blks: M2 = enc(M2, cmask)
        # M3 = M2
        # for enc in self.model_enc_blks: M3 = enc(M3, cmask)

        #Dropout 
        mod1 = F.dropout(mod1, self.drop_prob, self.training)
        mod2 = F.dropout(mod2, self.drop_prob, self.training)
        mod3 = F.dropout(mod3, self.drop_prob, self.training)
        p1, p2 = self.out(mod1, mod2, mod3, cmask)

        out_end = time.time()
        # print("emb time: {}, att time: {}, model: {}, out: {}".format(
        #     emb_end - start_time,
        #     att_end - emb_end,
        #     mod_end - att_end,
        #     out_end - mod_end
        # ))
        return p1, p2