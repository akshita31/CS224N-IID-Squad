import torch
import torch.nn as nn
import torch.nn.Functional as F

from layers import HighwayEncoder


class CharEmbedding(nn.Module):
    """Character Embedding layer used by BiDAF.

    It takes in an input word (or its index) and using the characters in the word, 
    transforms it to an embedding of a fixed size.

    Args:
        char_vector: Pretrained character vectors. (maybe one-hot. need to verify this)
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
        char_embed_size: dimension of the output embeddings for each word.
    """
    
    def __init__(self, char_vectors, drop_prob, char_embed_size) -> None:
        super(CharEmbedding, self).__init__()
        self.output_embed_size = char_embed_size # whatever dimension we will use for the word vector, same we will use for the char vectors (output of the Char Embed layer) 

        self.char_embed = nn.Embedding.from_pretrained(char_vectors)

        self.cnn = nn.Sequential(nn.Conv1d(in_channels = char_vectors.size(1), out_channels =  self.output_embed_size, kernel_size = 3),# check dimensions passed here
                                nn.AdaptiveMaxPool1d(self.output_embed_size))

    def forward(self, char_idxs):
        (batch_size, seq_len, _) = char_idxs.shape()
        
        emb = self.char_embed(char_idxs)
        emb = self.cnn(emb)
        
        assert(emb.shape == (batch_size, seq_len, self.output_embed_size))

        return emb

class WordPlusCharEmdedding(nn.Module):
    """Combines the Word and Character embedding and then applies a transformation and highway network.
    Output of this layer will be (batch_size, seq_len, hidden_size)
    """

    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob):
        super(WordPlusCharEmdedding, self).__init__()
        self.drop_prob = drop_prob
        self.char_embed_size = 50
        self.word_embed_size = word_vectors.size(1)
        self.hidden_size = hidden_size

        self.word_embed = nn.Embedding.from_pretrained(word_vectors)   
        self.char_embed = CharEmbedding(char_vectors=char_vectors, drop_prob=drop_prob, char_embed_size = self.char_embed_size)
        
        self.proj = nn.Linear(self.word_embed_size + self.char_embed_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, word_idxs, char_idxs):
        word_emb = self.word_embed(word_idxs)
        char_emb = self.char_embed(char_idxs)
        
        (batch_size, seq_len, _) = word_emb.shape()
        assert(char_emb.shape == (batch_size, seq_len, self.char_embed_size))

        emb = torch.cat(word_emb, char_emb, dim = 2)
        assert(emb.shape == (batch_size, seq_len, self.word_embed_size + self.char_embed_size))

        # to do: Check whether the dropout mentioned is needed ?
        # to do: how has the char and word embedding fed into the highway network ??
        emb = F.dropout(emb, self.drop_prob, self.training)
        emb = self.proj(emb)  # (batch_size, seq_len, hidden_size)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)

        assert(emb.shape == (batch_size, seq_len, self.hidden_size))
        return emb


## extra knowledge
    """ From the paper
Following Kim (2014), we obtain the character level embedding of each word using Convolutional Neural Networks (CNN). Characters are embedded into vectors, 
which can be considered as 1D inputs to the CNN, and whose size is the input channel size of the CNN. The outputs of the CNN are max-pooled over the entire width to obtain a
fixed-size vector for each word.

We use 100 1D filters for CNN char embedding, each with a width of 5.
        """