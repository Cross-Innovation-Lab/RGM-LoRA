import torch
import torch.nn as nn
from torch.nn import Parameter
from modules.attention.transformer import TransformerEncoder


class BottleNeckFusion(nn.Module):
    def __init__(self, config):
        super(BottleNeckFusion, self).__init__()
        self.embed_dim = 32
        self.seq = 4
        self.audio_linear = nn.Linear(74, self.embed_dim)
        self.visual_linear = nn.Linear(35, self.embed_dim)
        self.text_linear = nn.Linear(768, self.embed_dim)
        self.bottle_neck = Parameter(torch.Tensor(config.batch_size, self.seq, self.embed_dim))
        self.audio_cls = Parameter(torch.Tensor(config.batch_size, 1, self.embed_dim))
        self.visual_cls = Parameter(torch.Tensor(config.batch_size, 1, self.embed_dim))
        self.text_cls = Parameter(torch.Tensor(config.batch_size, 1, self.embed_dim))
        self.reset_parameter()
        self.audio_transformer = TransformerEncoder(embed_dim=self.embed_dim, num_heads=8, layers=4, attn_mask=False)
        self.visual_transformer = TransformerEncoder(embed_dim=self.embed_dim, num_heads=8, layers=4, attn_mask=False)
        self.text_transformer = TransformerEncoder(embed_dim=self.embed_dim, num_heads=8, layers=4, attn_mask=False)
        self.fc = nn.Linear(self.embed_dim, 1)

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.bottle_neck)
        nn.init.xavier_uniform_(self.audio_cls)
        nn.init.xavier_uniform_(self.visual_cls)
        nn.init.xavier_uniform_(self.text_cls)

    def forward(self, audio, visual, text):
        a, v, t = self.audio_linear(audio), self.visual_linear(visual), self.text_linear(text)
        seq_a, seq_v, seq_t = a.shape[1], v.shape[1], t.shape[1]
        a_in, v_in, t_in = torch.concat((self.audio_cls, a), dim=1), torch.concat((self.visual_cls, v), dim=1), \
                           torch.concat((self.text_cls, t), dim=1)
        # [batch_size,1+seq_len+seq_bottle,emb]
        a_out = self.audio_transformer(torch.concat((a_in, self.bottle_neck), dim=1))
        v_out = self.visual_transformer(torch.concat((v_in, self.bottle_neck), dim=1))
        t_out = self.text_transformer(torch.concat((t_in, self.bottle_neck), dim=1))
        # [batch_size, emb]
        a_cls, v_cls, t_cls = torch.squeeze(a_out[:, 0, :], dim=1), torch.squeeze(v_out[:, 0, :], dim=1), \
                              torch.squeeze(t_out[:, 0, :], dim=1)
        # [batch_size, seq_len, emb]
        a_fea, v_fea, t_fea = a_out[:, 1:seq_a+1, :], v_out[:, 1:seq_v+1, :], t_out[:, 1:seq_t+1, :]
        result = self.fc(a_cls) + self.fc(v_cls) + self.fc(t_cls)
        result_a = torch.sum(self.fc(a_fea)) + torch.mean(self.fc(a_fea)) + torch.std(self.fc(a_fea))
        result_v = torch.sum(self.fc(v_fea)) + torch.mean(self.fc(v_fea)) + torch.std(self.fc(v_fea))
        result_t = torch.sum(self.fc(t_fea)) + torch.mean(self.fc(t_fea)) + torch.std(self.fc(t_fea))
        return result_a, result_v, result_t, result
