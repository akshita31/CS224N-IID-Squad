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
    
    def __init__(self, word_mat, char_mat, n_encoder_blocks=7, n_head=4):
        super().__init__()
        #Dimension of connectors in QANet. #same as the d_model
        D = layers_qanet.D
        self.n_model_enc_blks = n_encoder_blocks
        train_args = args.get_train_args()
        self.embedding = layersnew.QANetEmbedding(word_mat, char_mat, D, drop_prob=layers_qanet.dropout, drop_char = layers_qanet.dropout_char, num_filters=100)
        
        self.emb_enc = layers_qanet.EncoderBlock(
            conv_num=4, ch_num=D, k=7, n_head=n_head)
        self.attention = layersnew.BiDAFAttention(hidden_size=D, drop_prob=layers_qanet.dropout)       
        self.cq_resizer = layers_qanet.Initialized_Conv1d(4 * D, D)
        #         self.model_encoders =  nn.ModuleList([layers_qanet.Encoder(d_model=self.d_model,
        #                                                                 num_filters=self.num_conv_filters,
        #                                                                 kernel_size=5,
        #                                                                 num_conv_layers=2,
        #                                                                 num_heads=8,
        #                                                                 drop_prob=drop_prob) for _ in range(5)])
        #         self.out = layers_qanet.QANetOutput(d_model=self.d_model, drop_prob=drop_prob)
        self.model_enc_blks = nn.ModuleList([
            layers_qanet.EncoderBlock(conv_num=2, ch_num=D, k=5, n_head=n_head)
            for _ in range(n_encoder_blocks)
        ])

        self.out = layers_qanet.QANetOutput(D)

    # log_p1, log_p2 = model(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        train_args = args.get_train_args()
        c_mask = (torch.zeros_like(cw_idxs) != cw_idxs).float()
        q_mask = (torch.zeros_like(qw_idxs) != qw_idxs).float()
        
        c = self.embedding(cw_idxs, cc_idxs)         # (batch_size, self.d_model, c_len)
        q = self.embedding(qw_idxs, qc_idxs)         # (batch_size, self.d_model, q_len)
        
        ce = self.emb_enc(c, c_mask, 1, 1) # (bs, d_model, ctxt_len)
        qe = self.emb_enc(q, q_mask, 1, 1) # (bs, d_model, ques_len)
        X = self.attention(ce, qe, c_mask, q_mask) # (bs, 4 * d_model, ctxt_len)
        # att = self.att(c_enc, q_enc, c_mask, q_mask)    # (batch_size, c_len, 4 * d_model)
        # m0 = self.att_conv(att.transpose(1,2)).transpose(1,2)
        # for i, enc in enumerate(self.model_encoders):
        #      m0 = enc(m0)
        #      m1 = m0
        intermediary = self.cq_resizer(X) # (bs, d_model, ctxt_len), fusion function
        intermediary = F.dropout(intermediary, p=train_args.qanet_dropout, training=self.training)
        # for i, enc in enumerate(self.model_encoders):
        #      m0 = enc(m0)
        #m2 = m0
        for index, block in enumerate(self.model_enc_blks):
             intermediary = block(intermediary, c_mask, index*(2+2)+1, self.n_model_enc_blks)
        intermediary1 = intermediary
        for index, block in enumerate(self.model_enc_blks):
             intermediary = block(intermediary, c_mask, index*(2+2)+1, self.n_model_enc_blks)
        intermediary2 = intermediary
        intermediary = F.dropout(intermediary, p=train_args.qanet_dropout, training=self.training)
        for index, block in enumerate(self.model_enc_blks):
             intermediary = block(intermediary, c_mask, index*(2+2)+1, self.n_model_enc_blks)
        intermediary3 = intermediary
        # for i, enc in enumerate(self.model_encoders):
        #     m0 = enc(m0)
        #     m3 = m0
        #
        out = self.out(intermediary1, intermediary2, intermediary3, c_mask)
        return out