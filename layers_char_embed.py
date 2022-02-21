import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import HighwayEncoder


# This is a private class. Only the WordPlusCharEmbedding should be called by the model
class _CharEmbedding(nn.Module):
    """Character Embedding layer used by BiDAF.

    It takes in an input word (or its index) and using the characters in the word, 
    transforms it to an embedding of a fixed size.

    Args:
        char_vector: Pretrained character vectors. (maybe one-hot. need to verify this)
        hidden_size (int): Size of hidden activations.
        drop_prob (float): Probability of zero-ing out activations
        char_embed_size: dimension of the output embeddings for each word.
    """
    
    def __init__(self, char_vectors, drop_prob, char_embed_size, device) -> None:
        super(_CharEmbedding, self).__init__()
        self.output_embed_size = char_embed_size # whatever dimension we will use for the word vector, same we will use for the char vectors (output of the Char Embed layer) 
        self.device = device
        self.chars_per_word = 16 # this is from the char_idx array
        self.char_embed = nn.Embedding.from_pretrained(char_vectors) #output will be (batch_size, seq_length, chars_per_word, input_embedding_len)

        #print('Chars per word is', self.chars_per_word)
        #print('Out channel is', self.output_embed_size)
        self.cnn = nn.Sequential(nn.Conv1d(in_channels = self.chars_per_word, out_channels =  self.output_embed_size, kernel_size = 3),# check dimensions passed here
                                #nn.Dropout(p = drop_prob),
                                nn.AdaptiveMaxPool1d(1)) # output will be (batch_size*seq_length, char_embed_size (or num_filters), 1)

    def forward(self, char_idxs):
        emb = self.char_embed(char_idxs)
        (batch_size, seq_len, num_chars_per_word, input_chars_dim) = emb.shape

        emb = emb.reshape(batch_size * seq_len, num_chars_per_word, -1)
        assert(emb.shape == (batch_size*seq_len, num_chars_per_word, input_chars_dim))
        emb = self.cnn(emb)
        
        assert(emb.shape == (batch_size*seq_len, self.output_embed_size, 1))

        emb = torch.squeeze(emb, dim=2)
        emb = emb.reshape(batch_size, seq_len, self.output_embed_size)

        assert(emb.shape == (batch_size, seq_len, self.output_embed_size))

        return emb

class WordPlusCharEmbedding(nn.Module):
    """Combines the Word and Character embedding and then applies a transformation and highway network.
    Output of this layer will be (batch_size, seq_len, hidden_size)
    """

    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob, device):
        super(WordPlusCharEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.char_embed_size = 50
        self.word_embed_size = word_vectors.size(1)
        self.hidden_size = hidden_size

        self.word_embed = nn.Embedding.from_pretrained(word_vectors)   
        self.char_embed = _CharEmbedding(char_vectors=char_vectors, drop_prob=drop_prob, char_embed_size = self.char_embed_size, device = device)
        
        #self.word_proj = nn.Linear(self.word_embed_size, (int)(hidden_size/2), bias=False)

        self.proj = nn.Linear(self.word_embed_size + self.char_embed_size, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, word_idxs, char_idxs):
        word_emb = self.word_embed(word_idxs)
        char_emb = self.char_embed(char_idxs)
        
        (batch_size, seq_len, _) = word_emb.shape
        assert(char_emb.shape == (batch_size, seq_len, self.char_embed_size))
        
        word_emb = F.dropout(word_emb, self.drop_prob, self.training)
        #word_projection = self.word_proj(word_emb)
        
        #concatenate to produce the final embedding
        emb = torch.cat((word_emb, char_emb), dim = 2)
        emb = self.proj(emb)
        assert(emb.shape == (batch_size, seq_len, self.hidden_size))

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