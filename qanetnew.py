import torch
import torch.nn as nn
import torch.nn.functional as F
import layers_qanet
import layersnew
import args

class QANetNew(nn.Module):
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
    
    def __init__(self, word_mat, char_mat, char_embed_dim, n_encoder_blocks=7, n_head=4, num_encoder_conv = 4, num_model_conv = 2,output_type= 'default'):
        super().__init__()
        #Dimension of connectors in QANet. #same as the d_model
        D = layers_qanet.D
        self.n_model_enc_blks = n_encoder_blocks
        self.embedding = layersnew.QANetEmbedding(word_mat, char_mat, D, drop_prob=layers_qanet.dropout, drop_char = layers_qanet.dropout_char, char_embed_dim=char_embed_dim)
        
        self.emb_enc = layers_qanet.EncoderBlock(
            conv_num=num_encoder_conv, ch_num=D, k=7, n_head=n_head)
        self.attention = layersnew.BiDAFAttention(hidden_size=D, drop_prob=layers_qanet.dropout)       
        self.cq_resizer = layers_qanet.Initialized_Conv1d(4 * D, D)

        self.model_enc_blks = nn.ModuleList([
            layers_qanet.EncoderBlock(conv_num=num_model_conv, ch_num=D, k=5, n_head=n_head)
            for _ in range(n_encoder_blocks)
        ])

        if output_type == 'default':
            self.out = layers_qanet.QANetOutput(D)
        elif output_type == 'conditional_attention':
            self.out = layers_qanet.QANetConditionalOutput2(D)

    def forward(self, Cwid, Qwid, Ccid, Qcid):
        train_args = args.get_train_args()
        maskC = (torch.zeros_like(Cwid) != Cwid).float()
        maskQ = (torch.zeros_like(Qwid) != Qwid).float()
        
        C = self.embedding(Cwid, Ccid)         # (batch_size, self.d_model, c_len)
        Q = self.embedding(Qwid, Qcid)         # (batch_size, self.d_model, q_len)
        
        Ce = self.emb_enc(C, maskC, 1, 1) # (bs, d_model, ctxt_len)
        Qe = self.emb_enc(Q, maskQ, 1, 1) # (bs, d_model, ques_len)
        X = self.attention(Ce, Qe, maskC, maskQ) # (bs, 4 * d_model, ctxt_len)
        # att = self.att(c_enc, q_enc, c_mask, q_mask)    # (batch_size, c_len, 4 * d_model)
        # m0 = self.att_conv(att.transpose(1,2)).transpose(1,2)
        # for i, enc in enumerate(self.model_encoders):
        #      m0 = enc(m0)
        #      m1 = m0
        M0 = self.cq_resizer(X) # (bs, d_model, ctxt_len), fusion function
        M0 = F.dropout(M0, p=train_args.qanet_dropout, training=self.training)
        # for i, enc in enumerate(self.model_encoders):
        #      m0 = enc(m0)
        #m2 = m0
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, self.n_model_enc_blks)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, self.n_model_enc_blks)
        M2 = M0
        M0 = F.dropout(M0, p=train_args.qanet_dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, self.n_model_enc_blks)
        M3 = M0
        # for i, enc in enumerate(self.model_encoders):
        #     m0 = enc(m0)
        #     m3 = m0
        #
        if train_args.output_type == 'default':
            out = self.out(M1, M2, M3, maskC)
        elif train_args.output_type == 'conditional_attention':
            out = self.out(M1, M2, M3, X, maskC)    
        return out

class QANetNew1(nn.Module):
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
    
    def __init__(self, word_mat, char_mat, char_embed_dim = 128, n_encoder_blocks=7, n_head=4, num_encoder_conv = 4, num_model_conv = 2,output_type= 'default', use_bidaf_att = True):
                
        super().__init__()
        #Dimension of connectors in QANet. #same as the d_model
        D = layers_qanet.D
        self.n_model_enc_blks = n_encoder_blocks
        self.use_bidaf_att = use_bidaf_att
        self.embedding = layersnew.QANetEmbedding(word_mat, char_mat, D, drop_prob=layers_qanet.dropout, drop_char = layers_qanet.dropout_char, char_embed_dim=char_embed_dim)
        self.output_type = output_type
        self.emb_enc = layers_qanet.EncoderBlock(
            conv_num=num_encoder_conv, ch_num=D, k=7, n_head=n_head)
        
        if use_bidaf_att:
            self.attention = layersnew.BiDAFAttention(hidden_size=D, drop_prob=layers_qanet.dropout) 
        else:
            self.cq_att = layers_qanet.CQAttention()
        
        self.cq_resizer = layers_qanet.Initialized_Conv1d(4 * D, D)

        self.model_enc_blks = nn.ModuleList([
            layers_qanet.EncoderBlock(conv_num=num_model_conv, ch_num=D, k=5, n_head=n_head)
            for _ in range(n_encoder_blocks)
        ])

        #print(self.output_type)
        if output_type == 'default':
            self.out = layers_qanet.QANetOutput(D)
        elif output_type == 'conditional_output':
            self.out = layers_qanet.QANetConditionalOutput(D)
        elif output_type == 'conditional_attention':
            self.out = layers_qanet.QANetConditionalOutput2(D)

    def forward(self, Cwid, Qwid, Ccid, Qcid):
        train_args = args.get_train_args()
        maskC = (torch.zeros_like(Cwid) != Cwid).float()
        maskQ = (torch.zeros_like(Qwid) != Qwid).float()
        
        C = self.embedding(Cwid, Ccid)         # (batch_size, self.d_model, c_len)
        Q = self.embedding(Qwid, Qcid)         # (batch_size, self.d_model, q_len)
        
        Ce = self.emb_enc(C, maskC, 1, 1) # (bs, d_model, ctxt_len)
        Qe = self.emb_enc(Q, maskQ, 1, 1) # (bs, d_model, ques_len)

        if self.use_bidaf_att:
            X = self.attention(Ce, Qe, maskC, maskQ) # (bs, 4 * d_model, ctxt_len)
        else:
            X = self.cq_att(Ce, Qe, maskC, maskQ)
        # att = self.att(c_enc, q_enc, c_mask, q_mask)    # (batch_size, c_len, 4 * d_model)
        # m0 = self.att_conv(att.transpose(1,2)).transpose(1,2)
        # for i, enc in enumerate(self.model_encoders):
        #      m0 = enc(m0)
        #      m1 = m0
        M0 = self.cq_resizer(X) # (bs, d_model, ctxt_len), fusion function
        M0 = F.dropout(M0, p=train_args.qanet_dropout, training=self.training)
        # for i, enc in enumerate(self.model_encoders):
        #      m0 = enc(m0)
        #m2 = m0
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, self.n_model_enc_blks)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, self.n_model_enc_blks)
        M2 = M0
        M0 = F.dropout(M0, p=train_args.qanet_dropout, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, maskC, i*(2+2)+1, self.n_model_enc_blks)
        M3 = M0
        # for i, enc in enumerate(self.model_encoders):
        #     m0 = enc(m0)
        #     m3 = m0
        #
        if self.output_type == 'default' or self.output_type == 'conditional_output':
            out = self.out(M1, M2, M3, maskC)
        elif self.output_type == 'conditional_attention':
            out = self.out(M1, M2, M3, X, maskC)    
        return out