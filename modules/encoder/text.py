from transformers import BertModel, BertConfig
from peft import get_peft_model, LoraConfig, TaskType
from torch import nn


class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        self.config = config
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.model = BertModel.from_pretrained('bert-base-uncased', config=self.bert_config)
        if self.config.use_lora:
            print("-" * 50)
            print("building lora model ...")
            peft_config = LoraConfig(
                task_type=TaskType.TOKEN_CLS, inference_mode=self.config.inference_mode,
                r=self.config.r, lora_alpha=self.config.lora_alpha, lora_dropout=self.config.lora_dropout
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
            print("-" * 50)

    def forward(self, bert_sentences, bert_sentence_types, bert_sentence_att_mask):
        # [batch_size, seq_len, dimension]
        if self.config.use_lora:
            return self.model.base_model.forward(input_ids=bert_sentences,
                                                 token_type_ids=bert_sentence_types,
                                                 attention_mask=bert_sentence_att_mask)
        else:
            return self.model.forward(input_ids=bert_sentences,
                                      token_type_ids=bert_sentence_types,
                                      attention_mask=bert_sentence_att_mask)
