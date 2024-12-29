import torch.nn as nn
from rnn import *

'''
Input output shapes
audio (COVAREP): (20, 74)
label: (1) -> [sentiment]
'''


class AudioEncoder(nn.Module):
    def __init__(self, config):
        super(AudioEncoder, self).__init__()
        self.acoustic_enc = RNNEncoder(
            in_size=74,
            hidden_size=config.hidden,
            out_size=config.rnn_hidden,
            num_layers=config.rnn_num_layers,
            dropout=config.dropout_a if config.rnn_num_layers > 1 else 0.0,
            bidirectional=config.bidirectional
        )

    def forward(self, acoustic, a_len):
        acoustic = self.acoustic_enc(acoustic, a_len)
        return acoustic
