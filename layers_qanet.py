from layers import HighwayEncoder
from layers_char_embed import _CharEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import masked_softmax

class QANetOutput(nn.Module):
    """Output layer used by QANet for question answering.

    As mentioned in the paper, output size of the encoding layers is (hidden_size = 128)
    They are basically the query informed context words representations
    
    Args:
        hidden_size (int): Hidden size used in the BiDAF model.
        drop_prob (float): Probability of zero-ing out activations.
    """
    def __init__(self, hidden_size, drop_prob):
        super(QANetOutput, self).__init__()
        self.start_linear = nn.Linear(2*hidden_size, 1, bias = False)
        self.end_linear = nn.Linear(2*hidden_size,1 , bias = False)

    def forward(self, m0, m1, m2, mask):
        
        (batch_size, seq_len, hidden_size) = m0.shape

        # (batch_size, seq_len, hidden_size)
        start_enc = torch.cat((m0, m1), dim =2)
        end_enc = torch.cat((m0, m2), dim = 2)

        assert(start_enc.shape == (batch_size, seq_len, 2*hidden_size))
        assert(end_enc.shape == (batch_size, seq_len, 2*hidden_size))

        # Shapes: (batch_size, seq_len, 1)
        logits_1 = self.start_linear(start_enc)
        logits_2 = self.end_linear(end_enc)

        assert(logits_1.shape == (batch_size, seq_len, 1))
        assert(logits_2.shape == (batch_size, seq_len, 1))

        # Shapes: (batch_size, seq_len)
        log_p1 = masked_softmax(logits_1.squeeze(dim=2), mask, log_softmax=True)
        log_p2 = masked_softmax(logits_2.squeeze(dim=2), mask, log_softmax=True)

        return log_p1, log_p2

class QANetEmbedding(nn.Module):
    """Combines the Word and Character embedding and then applies a transformation and highway network.
    Output of this layer will be (batch_size, seq_len, hidden_size)
    """

    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob, num_filters):
        super(QANetEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.num_filters = num_filters
        self.word_embed_size = word_vectors.size(1)
        self.hidden_size = hidden_size

        self.word_embed = nn.Embedding.from_pretrained(word_vectors)   
        self.char_embed = _CharEmbedding(char_vectors=char_vectors, drop_prob=drop_prob, num_filters = self.num_filters)

        self.hwy = HighwayEncoder(2, self.num_filters + self.word_embed_size)

    def forward(self, word_idxs, char_idxs):
        word_emb = self.word_embed(word_idxs)
        char_emb = self.char_embed(char_idxs)

        (batch_size, seq_len, _) = word_emb.shape
        assert(char_emb.shape == (batch_size, seq_len, self.num_filters))
        
        word_emb = F.dropout(word_emb, self.drop_prob, self.training)
        #word_projection = self.word_proj(word_emb)
        
        #concatenate to produce the final embedding
        emb = torch.cat((word_emb, char_emb), dim = 2)
        #emb = self.proj(emb)
        #assert(emb.shape == (batch_size, seq_len, self.hidden_size))

        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        assert(emb.shape == (batch_size, seq_len, self.word_embed_size + self.num_filters))
        return emb