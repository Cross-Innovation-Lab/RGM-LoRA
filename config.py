import os
import argparse
from datetime import datetime
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import pprint
from torch import optim
import torch.nn as nn


def get_args():
    parser = argparse.ArgumentParser(description='MOSI-and-MOSEI Sentiment Analysis')
    parser.add_argument('-f', default='', type=str)

    # Tasks
    parser.add_argument('--dataset', type=str, default='mosei', choices=['mosi', 'mosei'],
                        help='dataset to use (default: mosei)')
    parser.add_argument('--data_path', type=str, default='datasets',
                        help='path for storing the dataset')

    # use PEFT lora
    parser.add_argument('--PEFT', type=bool, default=True,
                        help='use lora')

    # Dropouts
    parser.add_argument('--dropout_a', type=float, default=0.1,
                        help='dropout of acoustic LSTM out layer')
    parser.add_argument('--dropout_v', type=float, default=0.1,
                        help='dropout of visual LSTM out layer')
    parser.add_argument('--dropout_prj', type=float, default=0.1,
                        help='dropout of projection layer')

    # Architecture
    parser.add_argument('--multiseed', action='store_true', help='training using multiple seed')
    parser.add_argument('--contrast', action='store_true', help='using contrast learning')
    parser.add_argument('--add_va', action='store_true', help='if add va MMILB module')
    parser.add_argument('--n_layer', type=int, default=1,
                        help='number of layers in LSTM encoders (default: 1)')
    parser.add_argument('--cpc_layers', type=int, default=1,
                        help='number of layers in CPC NCE estimator (default: 1)')
    parser.add_argument('--d_vh', type=int, default=16,
                        help='hidden size in visual rnn')
    parser.add_argument('--d_ah', type=int, default=16,
                        help='hidden size in acoustic rnn')
    parser.add_argument('--d_vout', type=int, default=16,
                        help='output size in visual rnn')
    parser.add_argument('--d_aout', type=int, default=16,
                        help='output size in acoustic rnn')
    parser.add_argument('--bidirectional', action='store_true', help='Whether to use bidirectional rnn')
    parser.add_argument('--d_prjh', type=int, default=128,
                        help='hidden size in projection network')
    parser.add_argument('--pretrain_emb', type=int, default=768,
                        help='dimension of pretrained model output')

    # Training Setting
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size (default: 32)')
    parser.add_argument('--clip', type=float, default=1.0,
                        help='gradient clip value (default: 0.8)')
    parser.add_argument('--lr_main', type=float, default=1e-3,
                        help='initial learning rate for main model parameters (default: 1e-3)')
    parser.add_argument('--lr_bert', type=float, default=5e-5,
                        help='initial learning rate for bert parameters (default: 5e-5)')
    parser.add_argument('--lr_mmilb', type=float, default=1e-3,
                        help='initial learning rate for mmilb parameters (default: 1e-3)')
    parser.add_argument('--alpha', type=float, default=0.1, help='weight for CPC NCE estimation item (default: 0.1)')
    parser.add_argument('--beta', type=float, default=0.1, help='weight for lld item (default: 0.1)')

    parser.add_argument('--weight_decay_main', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_bert', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')
    parser.add_argument('--weight_decay_club', type=float, default=1e-4,
                        help='L2 penalty factor of the main Adam optimizer')

    parser.add_argument('--optim', type=str, default='Adam',
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='number of epochs (default: 40)')
    parser.add_argument('--when', type=int, default=20,
                        help='when to decay learning rate (default: 20)')
    parser.add_argument('--patience', type=int, default=10,
                        help='when to stop training if best never change')
    parser.add_argument('--update_batch', type=int, default=1,
                        help='update batch interval')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of result logging (default: 100)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    args = parser.parse_args()
    return args


class Config(object):
    def __init__(self):
        super(Config, self).__init__()
        """
        basic config
        """
        self.dataset = 'MOSEI'
        self.data_path = '/home/lab/fuziwang/OGM/dataset/' + self.dataset
        self.mode = 'train'
        self.n_train = 0
        self.n_valid = 0
        self.n_test = 0
        self.batch_size = 32
        self.num_epochs = 40
        self.lr = 1e-5
        self.momentum = 0.9
        self.weight_decay = 1e-4

        self.a_rnn_hidden = 16
        self.a_rnn_num_layers = 1
        self.a_rnn_dropout = 0.1
        self.a_rnn_bidirectional = False

        """
        OGM config
        """
        self.alpha = 0.3

        """
        lora config reference on LoraConfig
        This is the configuration class to store the configuration of a [`~peft.Lora`].

        Args:
            use_lora (`bool`):Whether use peft method
            r (`int`): Lora attention dimension
            target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
            lora_alpha (`float`): The alpha parameter for Lora scaling.
            lora_dropout (`float`): The dropout probability for Lora layers.
            merge_weights (`bool`):
                Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
            fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
            bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
            modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
                and saved in the final checkpoint.
        """
        self.use_lora = True
        self.r = 8
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.inference_mode = True


def get_config(dataset, mode, batch_size):
    config = Config()
    config.dataset = dataset
    config.mode = mode
    config.batch_size = batch_size
    return config
