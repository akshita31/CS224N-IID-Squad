class PositionalEncoder(nn.Module):
    #Reference: https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py
    def __init__(self, in_channels):
        super(PositionalEncoder, self).__init__()

        if in_channels%2 == 0:
            self.channels = in_channels
        else:
            self.channels = in_channels + 1

        self.frequency_factor = 1.0 / (10000 ** (torch.arange(0, self.channels, 2).float() / self.channels))

    def forward(self, tensor):

        batch_size, x, orig_ch = tensor.shape
        # print("positional encoding orig shape", tensor.shape)
        pos_x = torch.arange(x, device=tensor.device).type(self.frequency_factor.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.frequency_factor)
        # print("sin_inp_x apply sincos", sin_inp_x.shape)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        #print("embx apply sincos", emb_x.shape)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        #print("emb zeros", emb.shape)
        emb[:, : self.channels] = emb_x
        #print("output", emb[None, :, :orig_ch].repeat(batch_size, 1, 1).shape)
        return emb[None, :, :orig_ch].repeat(batch_size, 1, 1)