import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

'''
Input output shapes
audio (COVAREP): (20, 74)
label: (1) -> [sentiment]
'''


class RNNEncoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers=1, dropout=0.2, bidirectional=False):
        """
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            num_layers: specify the number of layers of LSTMs.
            dropout: dropout probability
            bidirectional: specify usage of bidirectional LSTM
        Output:
            (return value in forward) a tensor of shape (batch_size, out_size)
        """
        super().__init__()
        self.bidirectional = bidirectional

        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                           batch_first=False)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear((2 if bidirectional else 1) * hidden_size, out_size)

    def forward(self, x, lengths):
        """
        x: (batch_size, sequence_len, in_size)
        """
        lengths = lengths.to(torch.int64)

        packed_sequence = pack_padded_sequence(x, lengths, enforce_sorted=False)
        _, final_states = self.rnn(packed_sequence)

        if self.bidirectional:
            h = self.dropout(torch.cat((final_states[0][0], final_states[0][1]), dim=-1))
        else:
            h = self.dropout(final_states[0].squeeze())
        y = self.linear(h)
        return y
