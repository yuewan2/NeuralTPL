import torch
from torch.utils.data import Dataset

import os
import copy
import pandas as pd
import numpy as np
import pickle
from rdkit import Chem
from collections import namedtuple

from utils.data_utils import extract_mapping, get_masked_mol_smarts, convert_to_group_mask
from utils.tpl_utils.test_mask_tpl import smi_tokenizer


input_names = ['src', 'src_am', 'src_seg', 'tgt', 'rt_label']
MainData = namedtuple('data', input_names)

class LvgpTplGenDataset(Dataset):
    def __init__(self, mode, data_folder='data/template', intermediate_folder='./intermediate', dictionary=None, allow_noise=True, subset=False):
        assert mode in ['train', 'test', 'val']
        self.mode = mode
        self.subset = subset
        self.allow_noise = allow_noise
        if mode != 'train':
            assert dictionary is not None
            self.src_itos, self.tgt_itos = dictionary
            self.data = pd.read_csv(os.path.join(data_folder, 'lvgp_story_{}.csv'.format(mode)))
        else:
            train_data = pd.read_csv(os.path.join(data_folder, 'lvgp_story_train.csv'))
            val_data = pd.read_csv(os.path.join(data_folder, 'lvgp_story_val.csv'))
            if 'lvgp_vocab.pk' not in os.listdir(intermediate_folder):
                print('Building vocab...')
                self.src_itos, self.tgt_itos = set(), set()
                data = pd.concat([train_data, val_data])
                data.reset_index(inplace=True, drop=True)
                for i in range(data.shape[0]):
                    backbone_tokens, _ = extract_mapping(data.iloc[i]['backbone'])
                    lvgp_tokens = data.iloc[i]['lvgp']
                    self.src_itos.update(backbone_tokens.split(' '))
                    self.tgt_itos.update(lvgp_tokens.split(' '))

                self.tgt_itos.update(['<RX_{}>'.format(i) for i in range(1, 11)])
                self.src_itos = ['<CLS>', '<pad>', '<unk>'] + sorted(
                    list(self.src_itos))  # <CLS> at the beginning index for future use convenience
                self.tgt_itos = ['<CLS>', '<eos>', '<sos>', '<pad>', '<unk>'] + sorted(list(self.tgt_itos))
                with open(os.path.join(intermediate_folder, 'lvgp_vocab.pk'), 'wb') as f:
                    pickle.dump([self.src_itos, self.tgt_itos], f)
            else:
                with open(os.path.join(intermediate_folder, 'lvgp_vocab.pk'), 'rb') as f:
                    self.src_itos, self.tgt_itos = pickle.load(f)
            self.data = eval('{}_data'.format(mode))

        self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
        self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}
        self.rt_token_dict = {'<RX_{}>'.format(i): (i - 1) for i in range(1, 11)}

        if self.subset:
            self.data = self.data.sample(n=1000)
        self.data.reset_index(inplace=True, drop=True)

        return

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        instance = self.data.iloc[idx]
        backbone = instance['backbone']
        rt_token = instance['rxn_class']
        rt_label = self.rt_token_dict[rt_token]

        backbone_token = ['<CLS>'] + backbone.split(' ')
        p = np.random.rand()
        if not self.allow_noise or p > 0.2:
            lvgp_token = [rt_token] + instance['lvgp'].split(' ') + ['<eos>']
        else:
            lvgp_token = ['<sos>'] + instance['lvgp'].split(' ') + ['<eos>']

        src, src_am = extract_mapping(' '.join(backbone_token))
        tgt = ' '.join(lvgp_token)

        src = np.array([self.src_stoi.get(t, self.src_stoi['<unk>']) for t in src.split(' ')])
        src_am = np.array(src_am.split(' '), dtype=int)
        tgt = np.array([self.tgt_stoi.get(t, self.tgt_stoi['<unk>']) for t in tgt.split(' ')])

        # Reassign Atom-mapping:
        reassignment = {}
        for am in src_am:
            if am != 0 and am not in reassignment:
                reassignment[am] = len(reassignment) + 1
        src_am = np.array([reassignment.get(am, 0) for am in src_am])

        return src, src_am, tgt, rt_label


class MainTplGenDataset(Dataset):
    def __init__(self, mode, data_folder='./data/template', intermediate_folder='./intermediate', dictionary=None, split_am=True, allow_noise=True, subset=False):
        '''
        :param mode: ['train', 'test', 'val']
        :param data_folder: root path of main template data
        :param dictionary: additional dictionary built by train_dataset
        :param split_am: if separate atom mapping from atoms
        :param allow_noise: if allow noise (may only occur in val/test setting)
        :param subset: for debugging only
        '''
        assert mode in ['train', 'test', 'val']
        self.mode = mode
        self.split_am = split_am
        self.allow_noise = allow_noise
        self.subset = subset
        if mode != 'train':
            assert dictionary is not None
            self.src_itos, self.tgt_itos = dictionary
            self.data = pd.read_csv(os.path.join(data_folder, 'main_story_{}.csv'.format(mode)))
        else:
            train_data = pd.read_csv(os.path.join(data_folder, 'main_story_train.csv'))
            val_data = pd.read_csv(os.path.join(data_folder, 'main_story_val.csv'))
            if 'main_vocab.pk' not in os.listdir(intermediate_folder):
                print('Building main vocab')
                self.src_itos, self.tgt_itos = set(), set()
                data = pd.concat([train_data, val_data])
                data.reset_index(inplace=True, drop=True)
                for i in range(data.shape[0]):
                    if self.split_am:
                        masked_rxn, _ = extract_mapping(data.iloc[i]['masked_rxn'])
                    else:
                        masked_rxn = data.iloc[i]['masked_rxn']
                    self.src_itos.update(masked_rxn.split(' '))
                    self.tgt_itos.update(data.iloc[i]['templates'].split(' '))

                # -----------------------------version2-------------------------------#
                # self.src_itos.update(['<RX_{}>'.format(i) for i in range(0, 11)])
                # ------------------------------------------------------------#
                self.tgt_itos.update(['<RX_{}>'.format(i) for i in range(1,11)])
                # -----------------------------version1-------------------------------#
                self.src_itos = ['<CLS>', '<pad>', '<unk>', '<MASK>', '<NOISE>'] + sorted(
                    list(self.src_itos))  # <CLS> at the beginning index for future use convenience
                self.tgt_itos = ['<CLS>', '<eos>', '<sos>', '<pad>', '<unk>'] + sorted(list(self.tgt_itos))

                with open(os.path.join(intermediate_folder, 'main_vocab.pk'), 'wb') as f:
                    pickle.dump([self.src_itos, self.tgt_itos], f)
            else:
                with open(os.path.join(intermediate_folder, 'main_vocab.pk'), 'rb') as f:
                    self.src_itos, self.tgt_itos = pickle.load(f)

            self.data = eval('{}_data'.format(mode))
            #print('Key Names:', input_names)
        self.src_stoi = {self.src_itos[i]: i for i in range(len(self.src_itos))}
        self.tgt_stoi = {self.tgt_itos[i]: i for i in range(len(self.tgt_itos))}
        self.rt_token_dict = {'<RX_{}>'.format(i):(i-1) for i in range(1,11)}
        if self.subset:
            self.data = self.data.sample(n=1000)
            self.data.reset_index(inplace=True, drop=True)

        with open(os.path.join(intermediate_folder, 'rxn2centers.pk'), 'rb') as f:
            self.rxn2centers = pickle.load(f)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        instance = self.data.iloc[idx]
        rt_token = instance['rxn_class']
        rt_label = self.rt_token_dict[rt_token]
        template = instance['templates']
        prod_patt, react_patt = ''.join(template.split(' ')).split('>>')

        rxn = instance['raw_rxn']
        react_smiles, prod_smiles = rxn.split('>>')

        # Add Random Noise (from other potential reaction centers) with 50% probability
        p = np.random.rand()
        if rxn not in self.rxn2centers or not self.allow_noise or p > 0.8 or not len(self.rxn2centers[rxn]):
            prod, masked_prod, atoms_to_use_guidance = \
                get_masked_mol_smarts(prod_smiles, prod_patt, return_atom_mapping=True)
            # react, masked_react = get_masked_mol_smarts(react_smiles, react_patt,
            #                                             atoms_to_use_guidance=atoms_to_use_guidance)
            merged_masked_rxn = ' '.join(smi_tokenizer('{}>>{}'.format(prod, masked_prod)))
        else:
            potential_reaction_centers = self.rxn2centers[rxn]
            potential_rc = np.random.choice(potential_reaction_centers)
            try:
                prod, masked_prod, atoms_to_use_guidance = \
                    get_masked_mol_smarts(prod_smiles, potential_rc, return_atom_mapping=True, noise_only=True)
                _, pseudo_react, _ = \
                    get_masked_mol_smarts(prod_smiles, prod_patt, potential_rc, return_atom_mapping=True, noise_only=False)
                # _, masked_react = \
                #     get_masked_mol_smarts(react_smiles, react_patt, potential_rc,
                #                           atoms_to_use_guidance=atoms_to_use_guidance)
                merged_masked_rxn = ' '.join(smi_tokenizer('{}>>{}'.format(masked_prod, pseudo_react)))
            except:
                prod, masked_prod, atoms_to_use_guidance = \
                    get_masked_mol_smarts(prod_smiles, prod_patt, return_atom_mapping=True)
                # react, masked_react = get_masked_mol_smarts(react_smiles, react_patt,
                #                                             atoms_to_use_guidance=atoms_to_use_guidance)
                merged_masked_rxn = ' '.join(smi_tokenizer('{}>>{}'.format(prod, masked_prod)))
        
        # print(merged_masked_rxn, '\n')
        merged_masked_rxn_final = convert_to_group_mask(merged_masked_rxn, '[#0:11111]', '•', '<MASK>')

        # print(merged_masked_rxn_final, '\n')
        if rxn in self.rxn2centers and len(self.rxn2centers[rxn]):
            merged_masked_rxn_final = convert_to_group_mask(merged_masked_rxn_final, '[#0:99999]', '≈', '<NOISE>')
        # print(merged_masked_rxn_final, '\n')

        # -------------------------------------version2----------------------------------------#
        # p = np.random.rand()
        # if not self.allow_noise or p > 0.15:
        #     merged_masked_rxn_final = '<CLS> {} '.format(rt_token) + merged_masked_rxn_final
        # else:
        #     merged_masked_rxn_final = '<CLS> {} '.format('<RX_0>') + merged_masked_rxn_final
        # template_tokens = instance['templates'].split(' ')
        # template_tokens = ['<sos>'] + template_tokens + ['<eos>']
        # -------------------------------------version1----------------------------------------#
        merged_masked_rxn_final = '<CLS> ' + merged_masked_rxn_final
        template_tokens = instance['templates'].split(' ')
        p = np.random.rand()
        if not self.allow_noise or p > 0.2:
            template_tokens = [rt_token] + template_tokens + ['<eos>']
        else:
            template_tokens = ['<sos>'] + template_tokens + ['<eos>']
        # -----------------------------------------------------------------------------#

        # Build Structured Data based on Token Index
        if self.split_am:
            src, src_am = extract_mapping(merged_masked_rxn_final)
        else:
            src = merged_masked_rxn_final
            src_am = ' '.join(['0'] * len(src.split(' ')))

        src = np.array([self.src_stoi.get(t, self.src_stoi['<unk>']) for t in src.split(' ')])
        # src_reverse = np.array([self.src_stoi.get(t, self.src_stoi['<unk>']) for t in src_reverse.split(' ')])
        src_am = np.array(src_am.split(' '), dtype=int)
        tgt = np.array([self.tgt_stoi.get(t, self.tgt_stoi['<unk>']) for t in template_tokens])

        # Reassign Atom-mapping:
        reassignment = {}
        for am in src_am:
            if am != 0 and am not in reassignment:
                reassignment[am] = len(reassignment) + 1
        src_am = np.array([reassignment.get(am, 0) for am in src_am])

        return src, src_am, tgt, rt_label


def collate_fn(data, sep, pads, device='cuda', inference=False):
    '''Build mini-batch tensors:
    :param sep: (int) index of src seperator
    :param pads: (tuple) index of src and tgt padding
    '''
    src_pad, tgt_pad = pads
    # Sort a data list by caption length
    # data.sort(key=lambda x: len(x[0]), reverse=True)
    if inference:
        src, src_am, tgt, rt_label = data[0]
    else:
        src, src_am, tgt, rt_label = zip(*data)

    max_src_length = max([len(s) for s in src])
    max_tgt_length = max([len(t) for t in tgt])

    # Pad_sequence
    new_src = torch.zeros((max_src_length, len(data))).fill_(src_pad).long()
    new_src_am = torch.zeros((max_src_length, len(data))).fill_(0).long()
    new_src_seg = torch.zeros((max_src_length, len(data))).fill_(0).long()
    new_tgt = torch.zeros((max_tgt_length, len(data))).fill_(tgt_pad).long()

    new_src_seg[0] = 2
    for i in range(len(data)):
        new_src[:, i][:len(src[i])] = torch.from_numpy(src[i])
        new_src_am[:, i][:len(src_am[i])] = torch.from_numpy(src_am[i])
        new_tgt[:, i][:len(tgt[i])] = torch.from_numpy(tgt[i])

        sep_pos = np.argwhere(src[i] == sep)[0][0]
        new_src_seg[1:sep_pos, i] = 1
        new_src_seg[sep_pos, i] = 2

    data = MainData(new_src.to(device), new_src_am.to(device), new_src_seg.to(device), new_tgt.to(device), torch.LongTensor(rt_label).to(device))
    return data
