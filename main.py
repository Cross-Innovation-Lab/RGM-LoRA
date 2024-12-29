"""
This is the Three or more models using OGM-GE strategy
"""
import time
import torch
from torch import nn
from dataset.data_loader import get_loader
from config import Config, get_config
from model import Multi_models
from utils import *


def train(model, train_loader, optimizer, criterion, scheduler, config):
    model.train()
    epoch_loss = 0 # TODO each epoch loss
    for idx, batch_data in enumerate(train_loader):
        text, visual, vlens, audio, alens, y, l, bert_sent, bert_sent_type, bert_sent_mask, ids = batch_data
        optimizer.zero_grad()
        with torch.cuda.device(0):
            text, visual, audio, y, l, bert_sent, bert_sent_type, bert_sent_mask = \
                text.cuda(), visual.cuda(), audio.cuda(), y.cuda(), l.cuda(), bert_sent.cuda(), \
                bert_sent_type.cuda(), bert_sent_mask.cuda()

        pred_a, pred_v, pred_t, pred = model(visual, audio, bert_sent, bert_sent_type, bert_sent_mask)
        y_loss = criterion(pred, y)
        loss = y_loss
        # TODO get params grad
        loss.backward()
        # TODO OGM-GE
        score_a = torch.exp(pred_a)
        score_v = torch.exp(pred_v)
        score_t = torch.exp(pred_t)
        score_all = {'audio': score_a, 'visual': score_v, 'text': score_t}
        sorted(score_all)
        score_list = list(score_all.values())
        coeff_a, coeff_v, coeff_t = 0, 0, 0
        for index, key in enumerate(score_all):
            if index == 2:
                if key == 'audio':
                    coeff_a = 1
                elif key == 'visual':
                    coeff_v = 1
                elif key == 'text':
                    coeff_t = 1

            ratio = score_all[key] / score_list[-1]
            if key == 'audio':
                coeff_a = 1 - torch.tanh(config.alpha * torch.relu(ratio))
            elif key == 'visual':
                coeff_v = 1 - torch.tanh(config.alpha * torch.relu(ratio))
            elif key == 'text':
                coeff_t = 1 - torch.tanh(config.alpha * torch.relu(ratio))

        for name, parms in model.named_parameters():
            layer = str(name).split('_')[0]
            if 'audio' == layer and len(parms.grad.size()) != 1:
                parms.grad = parms.grad * coeff_a + \
                             torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
            if 'visual' == layer and len(parms.grad.size()) != 1:
                parms.grad = parms.grad * coeff_v + \
                             torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
            if 'text' == layer and len(parms.grad.size()) != 1:
                parms.grad = parms.grad * coeff_t + \
                             torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
        optimizer.step()
        epoch_loss += loss.item() * config.batch_size
    scheduler.step()
    return epoch_loss / config.n_train


def evaluate(model, loader, criterion, config, test=True):
    model.eval()
    total_loss = 0.0
    result, result_a, result_v, result_t, truth = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            _, visual, _, audio, _, y, _, bert_sent, bert_sent_type, bert_sent_mask, _ = batch

            with torch.cuda.device(0):
                text, audio, vision, y = text.cuda(), audio.cuda(), vision.cuda(), y.cuda()
                lengths = lengths.cuda()
                bert_sent, bert_sent_type, bert_sent_mask = bert_sent.cuda(), bert_sent_type.cuda(), bert_sent_mask.cuda()

            batch_size = lengths.size(0)  # bert_sent in size (bs, seq_len, emb_size)

            # we don't need lld and bound anymore
            pred_a, pred_v, pred_t, pred = model(visual, audio, bert_sent, bert_sent_type, bert_sent_mask)

            total_loss += criterion(pred, y).item() * batch_size

            # Collect the results into ntest if test else self.hp.n_valid)
            result.append(pred), result_a.append(pred_a), result_v.append(pred_v), result_t.append(pred_t), truth.append(y)

    avg_loss = total_loss / (config.n_test if test else config.n_valid)
    result, result_a, result_v, result_t, truth = torch.cat(result), torch.cat(result_a), torch.cat(result_v),\
                                                  torch.cat(result_t), torch.cat(truth)
    return avg_loss, result, result_a, result_v, result_t, truth



if __name__ == '__main__':
    config = Config()
    dataset = str.lower(config.dataset)
    print("Start loading the data....")
    train_config = get_config(dataset, mode='train', batch_size=config.batch_size)
    valid_config = get_config(dataset, mode='valid', batch_size=config.batch_size)
    test_config = get_config(dataset, mode='test', batch_size=config.batch_size)

    # pretrained_emb saved in train_config here
    train_loader = get_loader(train_config, shuffle=True)
    print('Training data loaded!')
    valid_loader = get_loader(valid_config, shuffle=False)
    print('Validation data loaded!')
    test_loader = get_loader(test_config, shuffle=False)
    print('Test data loaded!')
    print('Finish loading the data....')
    print('Start building model!')
    model = Multi_models(config)
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)
    criterion = nn.L1Loss(reduction="mean")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.1)
    # TODO train and eval
    for epoch in range(1, config.num_epochs + 1):
        start_time = time.time()
        loss = train(model, train_loader, optimizer, criterion, scheduler, config)
        train_time = time.time() - start_time
        # TODO valid
        valid_loss, _, _, _, _, _ = evaluate(model, valid_loader, criterion, config, test=False)
        # TODO test
        test_loss, result, result_a, result_v, result_t, truth = evaluate(model, valid_loader, criterion, config, test=True)
        all_time = time.time() - start_time
        print('-' * 50)
        print('Epoch {:2d} | Train Time {:5.4f} sec | All Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.
              format(epoch, train_time, all_time, valid_loss, test_loss))
        if dataset == 'mosei':
            print('-----All-----')
            eval_mosei_senti(result, truth, True)
            print('---Audio-----')
            eval_mosei_senti(result_a, truth, True)
            print('---Visual----')
            eval_mosei_senti(result_v, truth, True)
            print('----Text-----')
            eval_mosei_senti(result_t, truth, True)
        if dataset == 'mosi':
            print('-----All-----')
            eval_mosi(result, truth, True)
            print('---Audio-----')
            eval_mosi(result_a, truth, True)
            print('---Visual----')
            eval_mosi(result_v, truth, True)
            print('----Text-----')
            eval_mosi(result_t, truth, True)

    print('-----program end------')