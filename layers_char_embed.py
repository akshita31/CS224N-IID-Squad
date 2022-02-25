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
        num_filters: dimension of the output embeddings for each word.
    """
    
    def __init__(self, char_vectors, drop_prob, num_filters) -> None:
        super(_CharEmbedding, self).__init__()
        
        self.input_char_emb_size = char_vectors.size(1)
        self.num_filters = num_filters
        self.char_embed = nn.Embedding.from_pretrained(char_vectors) #output will be (batch_size, seq_length, chars_per_word, input_embedding_len)
        self.drop_prob = drop_prob

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels = self.input_char_emb_size, out_channels =  self.num_filters, kernel_size = 3),# check dimensions passed here
                                nn.ReLU(),
                                nn.BatchNorm1d(num_features = self.num_filters),
                                # nn.Dropout(p = drop_prob),
                                nn.AdaptiveMaxPool1d(1)) # output will be (batch_size*seq_length, num_filters, 1)
        
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels = self.input_char_emb_size, out_channels =  self.num_filters, kernel_size = 5),# check dimensions passed here
                                nn.ReLU(),
                                nn.BatchNorm1d(num_features = self.num_filters),
                                # nn.Dropout(p = drop_prob),
                                nn.AdaptiveMaxPool1d(1)) # output will be (batch_size*seq_length, num_filters, 1)

    def forward(self, char_idxs):

        (batch_size, seq_len, _) = char_idxs.shape
        char_idxs = char_idxs.reshape(batch_size*seq_len, -1)
        
        emb = self.char_embed(char_idxs)
        emb = F.dropout(emb, self.drop_prob, self.training)

        emb = torch.transpose(emb, 1, 2)
        
        emb1 = self.conv1(emb)
        emb1 = torch.squeeze(emb1, dim=2)
        
        emb2= self.conv2(emb)
        emb2 = torch.squeeze(emb2, dim=2)

        emb = torch.cat((emb1, emb2), dim=1)

        # assert(emb.shape == (batch_size*seq_len, self.num_filters, 1))
        #emb = torch.squeeze(emb, dim=2)
        emb = emb.reshape(batch_size, seq_len, -1)

        assert(emb.shape == (batch_size, seq_len, self.num_filters *2))

        return emb
    
    def GetCharEmbedShape(self):
        return self.num_filters *2

class BiDAFWordPlusCharEmbedding(nn.Module):
    """Combines the Word and Character embedding and then applies a transformation and highway network.
    Output of this layer will be (batch_size, seq_len, hidden_size)
    """

    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob, num_filters):
        super(BiDAFWordPlusCharEmbedding, self).__init__()
        self.drop_prob = drop_prob
        self.num_filters = num_filters
        self.word_embed_size = word_vectors.size(1)
        self.hidden_size = hidden_size

        self.word_embed = nn.Embedding.from_pretrained(word_vectors)   
        self.char_embed = _CharEmbedding(char_vectors=char_vectors, drop_prob=drop_prob, num_filters = self.num_filters)

        self.proj = nn.Linear(self.word_embed_size + self.num_filters*2, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, word_idxs, char_idxs):
        word_emb = self.word_embed(word_idxs)
        char_emb = self.char_embed(char_idxs)

        (batch_size, seq_len, _) = word_emb.shape
        assert(char_emb.shape == (batch_size, seq_len, self.num_filters * 2))
        
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