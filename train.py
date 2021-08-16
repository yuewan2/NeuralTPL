from utils.build_utils import build_iterator, build_model, load_checkpoint
from utils.model_utils import validate

import re
import os
import copy
import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device GPU/CPU')
parser.add_argument('--batch_size_trn', type=int, default=8, help='train batch size')
parser.add_argument('--batch_size_val', type=int, default=2, help='val/test batch size')

parser.add_argument('--data_dir', type=str, default='./data/template', help='base directory')
parser.add_argument('--intermediate_dir', type=str, default='./intermediate', help='intermediate directory')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='checkpoint directory')
parser.add_argument('--checkpoint', type=str, default='', help='checkpoint model file')
parser.add_argument('--num_layers', type=int, default=12, help='number of layers of transformer')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model representation')
parser.add_argument('--heads', type=int, default=8, help='number of heads of multi-head attention')
parser.add_argument('--d_ff', type=int, default=2048, help='')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--allow_latent', type=str, default='True', choices = ['True', 'False'], help='whether to allow latent variable')
parser.add_argument('--generalize', type=str, default='True', choices = ['True', 'False'], help='')

parser.add_argument('--max_epoch', type=int, default=5000, help='maximum epoch')
parser.add_argument('--max_step', type=int, default=30000, help='maximum steps')
parser.add_argument('--report_per_step', type=int, default=200, help='train loss reporting steps frequency')
parser.add_argument('--save_per_step', type=int, default=1000, help='checkpoint saving steps frequency')
parser.add_argument('--val_per_step', type=int, default=1000, help='validation steps frequency')

args = parser.parse_args()


def train(args, model, optimizer, train_iter, val_iter, step, init_epoch):
    generalize = args.generalize=='True'

    criterion_tokens = nn.NLLLoss(ignore_index=model.embedding_tgt.word_padding_idx, reduction='sum')
    criterion_react_class = nn.NLLLoss(reduction='sum')
    running_loss = []
    running_loss_nll_token, running_loss_nll_react_class = [], []
    running_loss_kld = []

    for epoch in range(init_epoch, args.max_epoch):
        for batch in train_iter:
            model.train()

            gtruth_token = batch.tgt[1:].view(-1)
            gtruth_react_class = batch.rt_label
            inputs = (batch.src, batch.src_am, batch.src_seg, batch.tgt)
            scores, rt_scores, kld_loss = model(inputs, generalize=generalize)
            scores = scores.view(-1, scores.size(2))
            # if not isinstance(scores_reverse, int):
            #     scores_reverse = scores_reverse.view(-1, scores_reverse.size(2))

            optimizer.zero_grad()
            nll_loss_tokens = criterion_tokens(scores, gtruth_token)

            if generalize:
                nll_loss_rc = criterion_react_class(rt_scores, gtruth_react_class)
                # nll_loss_tokens_reverse = 0.01 * criterion_tokens(scores_reverse, gtruth_token)
            else:
                nll_loss_rc = 0
                # nll_loss_tokens_reverse = 0
            loss = nll_loss_tokens + nll_loss_rc + \
                        kld_loss * min(500, max(10, np.exp(step/2000)))

            # Update loss:
            loss.backward()
            optimizer.step()

            # Report loss:
            running_loss.append(loss.item())
            running_loss_nll_token.append(nll_loss_tokens.item())
            running_loss_nll_react_class.append(nll_loss_rc.item() if generalize else 0)
            running_loss_kld.append(kld_loss.item())

            if step % args.report_per_step == 0:
                print_line = "[Epoch {} Iter {}] Loss {} NLL-Loss {} Class-Loss {} KL-Loss {}".format(epoch, step,
                                                                                 round(np.mean(running_loss), 4),
                                                                                 round(np.mean(running_loss_nll_token), 4),
                                                                                 round(np.mean(running_loss_nll_react_class), 4),
                                                                                 round(np.mean(running_loss_kld), 4))
                print(print_line)
                running_loss, running_loss_kld = [], []
                running_loss_nll_token, running_loss_nll_react_class = [], []
                
            if step % args.save_per_step == 0:
                checkpoint_path = args.checkpoint_dir + '/model_{}_wz.pt'.format(step)
                torch.save({'model': model, 'step': step, 'optim': optimizer}, checkpoint_path)
                print('Checkpoint saved to {}'.format(checkpoint_path))
                
            if step % args.val_per_step == 0:
                accuracy_token, accuracy_rc = validate(val_iter, model, generalize=generalize)
                print('Validation accuracy: {} - {}'.format(round(accuracy_token, 4),
                                                            round(accuracy_rc, 4)))
                # accuracy_token, accuracy_rc = validate(train_iter, model, generalize=generalize)
                # print('Train accuracy: {} - {}'.format(round(accuracy_token, 4),
                #                                        round(accuracy_rc, 4)))

            step += 1
        if step > args.max_step:
            break


def main(args):
    # Build Data Iterator:
    train_iter, val_iter, test_iter, vocab_itos_src, vocab_itos_tgt = build_iterator(args)
    # Build Base Model:
    model = build_model(args, vocab_itos_src, vocab_itos_tgt)

    # Load Checkpoint:
    step = 1
    init_epoch = 0
    optimizer = optim.Adam(model.parameters(), lr=0.0001, eps=1e-9)
    if args.checkpoint:
        step, optimizer, model = load_checkpoint(args)
        init_epoch = step//len(train_iter)
        model = model.to(args.device)

    # Begin Training:
    train(args, model, optimizer, train_iter, val_iter, step, init_epoch)
    print('finished')
    return


if __name__ == "__main__":
    generalize = args.generalize == 'True'

    if generalize:
        args.checkpoint_dir = args.checkpoint_dir + '/generalize'
    else:
        args.checkpoint_dir = args.checkpoint_dir + '/leaving_group'

    print(args)
    with open('args.pk', 'wb') as f:
        pickle.dump(args, f)
    main(args)
