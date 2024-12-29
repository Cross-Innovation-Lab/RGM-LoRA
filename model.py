from modules.encoder.text import TextEncoder
from modules.fusion import BottleNeckFusion
from torch import nn


class Multi_models(nn.Module):
    def __init__(self, config):
        super(Multi_models, self).__init__()
        # self.audio_enc = nn.Linear(74, 74)
        # self.visual_enc = nn.Linear(35, 35)
        self.text_enc = TextEncoder(config)
        self.BottleNeckFuse = BottleNeckFusion(config)

    def forward(self, audio, visual, bert_sentences, bert_sentence_types, bert_sentence_att_mask):
        # a_in = self.audio_enc(audio)
        # v_in = self.visual_enc(visual)
        t_in = self.text_enc(bert_sentences, bert_sentence_types, bert_sentence_att_mask)
        a_result, v_result, t_result, result = self.BottleNeckFuse(audio, visual, t_in)
        return a_result, v_result, t_result, result


