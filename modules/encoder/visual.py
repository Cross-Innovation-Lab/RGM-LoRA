import torch.nn as nn
from rnn import *
'''
Input output shapes
visual (FACET): (20, 35)
label: (1) -> [sentiment]
'''


class VisualEncoder(nn.Module):
    def __init__(self, hp):
        super(VisualEncoder, self).__init__()
        self.acoustic_enc = RNNEncoder(
            in_size=hp.d_ain,
            hidden_size=hp.d_ah,
            out_size=hp.d_aout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional
        )
        self.fc = nn.Linear(16, 1)
    def forward(self, sentences, visual, acoustic, v_len, a_len, bert_sent, bert_sent_type, bert_sent_mask, y=None, mem=None):
        print(acoustic.shape)
        print(a_len.shape)
        acoustic = self.acoustic_enc(acoustic, a_len)
        pred = self.fc(acoustic)
        return pred