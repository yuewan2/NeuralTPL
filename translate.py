import re
import os
import copy
import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from utils.build_utils import build_iterator, build_model, load_checkpoint
from utils.model_utils import validate
from utils.translate_utils import translate_batch, explain_batch
from utils.data_utils import extract_mapping, reconstruct_mapping
from models.model import TemplateGenerator

import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device GPU/CPU')
parser.add_argument('--batch_size_val', type=int, default=4, help='batch size')
parser.add_argument('--batch_size_trn', type=int, default=4, help='batch size')
parser.add_argument('--beam_size', type=int, default=10, help='beam size')
parser.add_argument('--data_split', type=str, default='tst', choices=['trn', 'val', 'tst'], help='which data split to translate on')
parser.add_argument('--full_version', type=str, default='False', choices=['True', 'False'])
parser.add_argument('--constraint', type=str, default='True', choices=['True', 'False'])

parser.add_argument('--num_layers', type=int, default=12, help='number of layers of transformer')
parser.add_argument('--d_model', type=int, default=256, help='dimension of model representation')
parser.add_argument('--heads', type=int, default=8, help='number of heads of multi-head attention')
parser.add_argument('--d_ff', type=int, default=2048, help='')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--generalize', type=str, default='True', choices = ['True', 'False'], help='')

parser.add_argument('--data_dir', type=str, default='./data/template', help='base directory')
parser.add_argument('--intermediate_dir', type=str, default='./intermediate', help='intermediate directory')
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='checkpoint directory')
parser.add_argument('--model_selection', type=str, default='False', choices = ['True', 'False'])
parser.add_argument('--start_step', type=int, default=0, help = 'start step of model selection checkpoint')
parser.add_argument('--checkpoint', type=str, help='checkpoint model file')

args = parser.parse_args()

def translate(args, iterator, model, vocab_itos_src, vocab_itos_tgt, generalize):
    # Different mask_pos for untyped vs typed
    # mask_pos=-1
    # if '<MASK>' in vocab_itos_src:
    #     mask_pos = np.argwhere(np.array(vocab_itos_src) == '<MASK>')[0][0]

    gtruth, masked_src, react_class, hypos = [], [], [], []
    target_mask_num = None

    for batch in tqdm(iterator):
        inputs = (batch.src.to(args.device), batch.src_am.to(args.device),
                  batch.src_seg.to(args.device), batch.tgt.to(args.device))
        if generalize:
            max_length = 200
        else:
            # inputs = parse_batch(batch)
            # original_src, masked_src = inputs[0]
            # masked_part = inputs[1]
            # original_segments, masked_atom_mapping, original_atom_mapping, original_length, masked_length = inputs[2]
            # masked_segments, center_segments = original_segments, None
            # if args.full_version == 'False' and args.constraint == 'True':
            #     target_mask_num = masked_src.eq(mask_pos).sum(dim=0)
            max_length = 50

        pred_tokens, pred_scores = translate_batch(model, inputs,
                                                   sos_idx=np.argwhere(np.array(vocab_itos_tgt) == '<sos>')[0][0],
                                                   eos_idx=np.argwhere(np.array(vocab_itos_tgt) == '<eos>')[0][0],
                                                   sep_idx=np.argwhere(np.array(vocab_itos_tgt) == '>>')[0][0] if generalize else -1,
                                                   generalize=generalize, max_length=max_length,
                                                   fixed_z=False)

        rt_batch, mask_rxn_batch, gtruth_batch, hypos_batch, score_batch = \
            explain_batch(batch.src, batch.src_am, batch.tgt, pred_tokens, pred_scores, vocab_itos_src, vocab_itos_tgt, return_score=True)


        gtruth += gtruth_batch  # gt center
        hypos += hypos_batch  # infered center
        masked_src += mask_rxn_batch  # gt mask_rxn
        react_class += rt_batch


    return masked_src, react_class, gtruth, hypos





def main(args):
    # Build Data Iterator:
    trn_iter, val_iter, tst_iter, vocab_itos_src, vocab_itos_tgt = build_iterator(args)

    # Build Base Model:
    model = build_model(args, vocab_itos_src, vocab_itos_tgt)
    # Load Checkpoint:
    if args.checkpoint:
        _, _, model = load_checkpoint(args)

    iterator = eval('{}_iter'.format(args.data_split))

    # Begin Translating:
    generalize = args.generalize == 'True'
    if generalize:
        file_name = 'translate_result_generalize_from_{}.pk'.format(args.data_split)
    else:
        file_name = 'translate_result_lvgp_from_{}.pk'.format(args.data_split)

    masked_src, react_class, gtruth, hypos = \
        translate(args, iterator, model, vocab_itos_src, vocab_itos_tgt, generalize)

    accuracy_matrix = np.zeros((len(gtruth), args.beam_size))
    for i in range(len(gtruth)):
        for j in range(args.beam_size):
            if gtruth[i] in hypos[i][:j + 1]:
                accuracy_matrix[i][j] = 1
    for j in range(args.beam_size):
        print('Top-{}: {}'.format(j + 1, round(np.mean(accuracy_matrix[:, j]), 4)))


    with open(os.path.join(args.intermediate_dir, file_name), 'wb') as f:
        pickle.dump([masked_src, react_class, gtruth, hypos], f)

    return

def select_model(args):
    print('Model Selection:')
    generalize = args.generalize=='True'
    trn_iter, val_iter, tst_iter, vocab_itos_src, vocab_itos_tgt = build_iterator(args)
    # model = build_model(args, vocab_itos_src, vocab_itos_tgt)
    print(args.checkpoint_dir)
    best_checkpoint, best_acc = None, -1
    for checkpoint_step in np.arange(args.start_step, 300001, 1000):
        checkpoint_file = 'model_{}_wz.pt'.format(checkpoint_step)
        if checkpoint_file not in os.listdir(args.checkpoint_dir):
            continue
        args.checkpoint = checkpoint_file
        _, _, model = load_checkpoint(args)
        print('Start Validating...')
        accuracy_val, _ = validate(tst_iter, model, generalize=generalize, verbose=True)
        if accuracy_val > best_acc:
            best_acc = accuracy_val
            best_checkpoint = checkpoint_file
        print('{}: {}'.format(checkpoint_file, accuracy_val))

    print('BEST: {} - {}'.format(best_checkpoint, best_acc))
    

if __name__ == "__main__":
    print(args)
    if args.generalize == 'True':
        args.checkpoint_dir = args.checkpoint_dir + '/generalize'
        args.full_version = 'True'
    else:
        args.checkpoint_dir = args.checkpoint_dir + '/leaving_group'

    if args.model_selection == 'True':
        select_model(args)
    else:
        main(args)



