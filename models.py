"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import layers_char_embed
import layers_qanet
import torch
import torch.nn as nn
import args
import torch.nn.functional as F

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
    """"
    Based on two papers:
    First paper: "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).
    Second paper: Adams Wei Yu, David Dohan, Minh-Thang Luong, Rui Zhao, Kai Chen, Mohammad Norouzi,
    and Quoc V Le. Qanet: Combining local convolution with global self-attention for reading
    comprehension. arXiv preprint arXiv:1804.09541, 2018 (https://arxiv.org/pdf/1804.09541.pdf).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_mat (torch.Tensor): Pre-trained word vectors.
        char_mat (torch.Tensor): Pre-trained char vectors.
        n_encoder_blocks: number of blocks in an encoder, as mentioned in page 3 of the paper.
        n_head: The number of head of the attention mechanmism.
    """
    
    def __init__(self, word_vectors, char_vectors, number_of_encoder_blocks=7, number_of_heads=4):
        super().__init__()
        #Dimension of connectors in QANet. #same as the d_model
        self.n_model_enc_blks = number_of_encoder_blocks
        train_args = args.get_train_args()
        #         # These two layers will reduce the dimensionality of the embedding of each word from (500) to (128)
        #         self.context_conv = layers_qanet.DepthwiseSeparableConv(
        #             in_channels = self.initial_embed_dim,
        #             out_channels= self.d_model,
        #             kernel_size=5)
        #
        #         self.question_conv = layers_qanet.DepthwiseSeparableConv(
        #             in_channels = self.initial_embed_dim,
        #             out_channels = self.d_model,
        #             kernel_size=5)
        if train_args.use_pretrained_char:
            self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_vectors), freeze=True)
        else:
            self.char_emb = nn.Embedding.from_pretrained(torch.Tensor(char_vectors), freeze=False)
        #
        # self.emb = layers_qanet.QANetEmbedding(word_vectors=word_vectors,char_vectors=char_vectors, drop_prob=drop_prob, num_filters=100)
        # self.initial_embed_dim = self.emb.GetOutputEmbeddingDim()
        # self.d_model = 128 # d model is the dimensionality of each word before and after it goes into the encoder layer, i
        # self.num_conv_filters = 128
        self.emb = layers_qanet.Embedding()
        self.word_emb = nn.Embedding.from_pretrained(torch.Tensor(word_vectors), freeze=True)
        self.emb_encoding = layers_qanet.theEncoderblock(number_of_convolutions=4, number_of_characters=layers_qanet.train_args.d_model, k=7, number_of_head=number_of_heads)
        self.contextqueryattention = layers_qanet.ContextQueryAttention()
        self.contextqueryattention_change_size = layers_qanet.FeedForwardHelper(4 * layers_qanet.train_args.d_model, layers_qanet.train_args.d_model)
        #         self.model_encoders =  nn.ModuleList([layers_qanet.Encoder(d_model=self.d_model,
        #                                                                 num_filters=self.num_conv_filters,
        #                                                                 kernel_size=5,
        #                                                                 num_conv_layers=2,
        #                                                                 num_heads=8,
        #                                                                 drop_prob=drop_prob) for _ in range(5)])
        #         self.out = layers_qanet.QANetOutput(d_model=self.d_model, drop_prob=drop_prob)
        self.model_encoder_blocks = nn.ModuleList([layers_qanet.theEncoderblock(number_of_convolutions=2, number_of_characters=layers_qanet.train_args.d_model, k=5, number_of_head=number_of_heads)for index in range(number_of_encoder_blocks)])
        self.out = layers_qanet.QANetOutput(layers_qanet.train_args.d_model)

    #     def __init__(self, word_vectors, char_vectors, drop_prob=0.):
    #         super(QANet, self).__init__()
    #         # self.word_embed_size = word_vectors.size(1)
    #
    #         # Output of the Convolutions above will be fed into the encoder
    #         self.embedding_encoder_context =  layers_qanet.Encoder(d_model=self.d_model,
    #                                                                 num_filters=self.num_conv_filters,
    #                                                                 kernel_size=7,
    #                                                                 num_conv_layers=4,
    #                                                                 num_heads=8,
    #                                                                 drop_prob=drop_prob)
    #
    #         self.embedding_encoder_question =  layers_qanet.Encoder(d_model=self.d_model,
    #                                                                 num_filters=self.num_conv_filters,
    #                                                                 kernel_size=7,
    #                                                                 num_conv_layers=4,
    #                                                                 num_heads=8,
    #                                                                 drop_prob=drop_prob)
    #
    #         self.att = layers.BiDAFAttention(hidden_size=self.d_model, drop_prob=drop_prob)
    #
    #         self.att_conv = layers_qanet.DepthwiseSeparableConv(
    #             in_channels = self.d_model*4,
    #             out_channels = self.d_model,
    #             kernel_size=5)
    #



    # log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        train_args = args.get_train_args()
        # c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        # q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        # c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)
        # batch_size = cw_idxs.shape[0]
        c_mask = (torch.zeros_like(cw_idxs) != cw_idxs).float()
        q_mask = (torch.zeros_like(qw_idxs) != qw_idxs).float()
        cw_word, cc_character = self.word_emb(cw_idxs), self.char_emb(cc_idxs)
        # (bs, ctxt_len, word_emb_dim=300), (bs, ctxt_len, char_lim, char_emb_dim=64)
        qw_word, qc_character = self.word_emb(qw_idxs), self.char_emb(qc_idxs)
        # In QANet the projection is not applied and output of highway network is same size as the word+char embedding dim
        # c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, self.initial_embed_dim)
        # q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, self.initial_embed_dim)
        # (bs, ques_len, word_emb_dim=300), (bs, ques_len, char_lim, char_emb_dim=64)
        # c_emb = self.context_conv(c_emb.transpose(1,2)).transpose(1,2) # (batch_size, self.num_conv_filters, c_len)
        # q_emb = self.question_conv(q_emb.transpose(1,2)).transpose(1,2) # (batch_size, self.num_conv_filters, q_len)
        c, q = self.emb(cc_character, cw_word), self.emb(qc_character, qw_word)
        # c_enc = self.embedding_encoder_context(c_emb)
        # q_enc = self.embedding_encoder_question(q_emb)
        ce_em = self.emb_encoding(c, c_mask, 1, 1) # (bs, d_model, ctxt_len)
        qe_em = self.emb_encoding(q, q_mask, 1, 1) # (bs, d_model, ques_len)
        X = self.contextqueryattention(ce_em, qe_em, c_mask, q_mask) # (bs, 4 * d_model, ctxt_len)
        # att = self.att(c_enc, q_enc, c_mask, q_mask)    # (batch_size, c_len, 4 * d_model)
        # m0 = self.att_conv(att.transpose(1,2)).transpose(1,2)
        # for i, enc in enumerate(self.model_encoders):
        #      m0 = enc(m0)
        #      m1 = m0
        intermediate = self.contextqueryattention_change_size(X) # (bs, d_model, ctxt_len), fusion function
        intermediate = F.dropout(intermediate, p=train_args.qanet_dropout, training=self.training)
        # for i, enc in enumerate(self.model_encoders):
        #      m0 = enc(m0)
        #m2 = m0
        for i, block in enumerate(self.model_encoder_blocks):
             intermediate = block(intermediate, c_mask, i*(2+2)+1, self.n_model_enc_blks)
        intermediate1 = intermediate
        for i, block in enumerate(self.model_encoder_blocks):
             intermediate = block(intermediate, c_mask, i*(2+2)+1, self.n_model_enc_blks)
        intermediate2 = intermediate
        intermediate = F.dropout(intermediate, p=train_args.qanet_dropout, training=self.training)
        for i, block in enumerate(self.model_encoder_blocks):
             intermediate = block(intermediate, c_mask, i*(2+2)+1, self.n_model_enc_blks)
        intermediate3 = intermediate
        # for i, enc in enumerate(self.model_encoders):
        #     m0 = enc(m0)
        #     m3 = m0
        #
        #
        return self.out(intermediate1, intermediate2, intermediate3, c_mask)
        # p1, p2 = self.out(M1, M2, M3, maskC) # (bs, ctxt_len)
        # return p1, p2


class QANetConfig:
    def __init__(self) -> None:
        pass
