import warnings
warnings.filterwarnings("ignore")

from tpl_rxnmapper import RXNMapper
import pickle
import pandas as pd
import numpy as np
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from rdkit.Chem import AllChem
from multiprocessing.pool import Pool
from functools import partial

from utils.data_utils import *
from rdchiral.template_extractor import extract_from_reaction
from rdchiral.main import rdchiralRunText
from utils.data_utils import extract_mapping, reconstruct_mapping

rxn_mapper = RXNMapper()

def convert_to_smiles(rxn_smarts):
    prod, react = rxn_smarts.split('>>')
    prod = Chem.MolToSmiles(Chem.MolFromSmarts(prod))
    react = Chem.MolToSmiles(Chem.MolFromSmarts(react))
    return '{}>>{}'.format(prod,react)

def convert_to_smarts(rxn_smiles):
    prod, react = rxn_smiles.split('>>')
    prod = Chem.MolToSmarts(Chem.MolFromSmiles(prod))
    react = Chem.MolToSmarts(Chem.MolFromSmiles(react))
    return '{}>>{}'.format(prod,react)

def postprocess_am(mapped_rxn_smiles):
    prod, react = mapped_rxn_smiles.split('>>')
    prod_mol, react_mol = Chem.MolFromSmarts(prod), Chem.MolFromSmarts(react)
    prod_atom_maps, react_atom_maps = [], []
    for atom in prod_mol.GetAtoms():
        if 'molAtomMapNumber' in atom.GetPropsAsDict():
            prod_atom_maps.append(atom.GetProp('molAtomMapNumber'))
    for atom in react_mol.GetAtoms():
        if 'molAtomMapNumber' in atom.GetPropsAsDict():
            react_atom_maps.append(atom.GetProp('molAtomMapNumber'))
    share_atom_maps = sorted(list(set(prod_atom_maps) & set(react_atom_maps)))
    for mol in [prod_mol, react_mol]:
        for atom in mol.GetAtoms():
            if 'molAtomMapNumber' in atom.GetPropsAsDict():
                if atom.GetProp('molAtomMapNumber') not in share_atom_maps:
                    atom.ClearProp('molAtomMapNumber')

    return '{}>>{}'.format(Chem.MolToSmarts(prod_mol), Chem.MolToSmarts(react_mol))

def assign_am(rxn_center, postprocess=False, sanitize_token=True):
    results = rxn_mapper.get_attention_guided_atom_maps([rxn_center],
                                                        canonicalize_rxns=False,
                                                        sanitize_token=sanitize_token)
    mapped_rxn_smiles = results[0]['mapped_rxn']
    mapped_rxn_smiles = mapped_rxn_smiles.replace('-,:', '')
    mapped_rxn_smiles = mapped_rxn_smiles.replace(':&:', ':')
    mapped_rxn_smiles = mapped_rxn_smiles.replace('-&-', '-')
    mapped_rxn_smiles = mapped_rxn_smiles.replace('/&/', '/')
    mapped_rxn_smiles = mapped_rxn_smiles.replace('-&#', '-')
    mapped_rxn_smiles = mapped_rxn_smiles.replace('-&=', '-')
    
    if postprocess:
        mapped_rxn_smiles = postprocess_am(mapped_rxn_smiles)
        mapped_rxn_smiles = mapped_rxn_smiles.replace('-,:', '')
        mapped_rxn_smiles = mapped_rxn_smiles.replace(':&:', ':')
        mapped_rxn_smiles = mapped_rxn_smiles.replace('-&-', '-')
        mapped_rxn_smiles = mapped_rxn_smiles.replace('/&/', '/')
        mapped_rxn_smiles = mapped_rxn_smiles.replace('-&#', '-')
        mapped_rxn_smiles = mapped_rxn_smiles.replace('-&=', '-')
    return mapped_rxn_smiles

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|<MASK>|<unk>|>>|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    if smi != ''.join(tokens):
        print('ERROR tokenizing {}'.format(smi))
        assert smi == ''.join(tokens)
    return tokens

def clear_atom_map(mol_smiles):
    mol = Chem.MolFromSmiles(mol_smiles)
    for atom in mol.GetAtoms():
        atom.ClearProp('molAtomMapNumber')
    return Chem.MolToSmiles(mol)

# Assign atom mapping for format-3 template:
def get_mapped_center(center, postprocess=False, sanitize_token=True):
    target_flag = []
    prod_f, react_f = center.split('>>')

    for i, token in enumerate(smi_tokenizer(react_f)):
        if token[0] == '[' and token[-1] == ']' or token == '.':
            target_flag.append(1)
        elif re.match('[a-zA-Z]', token):
            target_flag.append(-1)
        else:
            target_flag.append(0)

    selected_indices = []
    i = 0
    while i < len(target_flag):
        if target_flag[i] == 1:
            start_idx = i
            while i < len(target_flag) and target_flag[i] != -1:
                i += 1
            end_idx = i

            if i < len(target_flag) and target_flag[i] == -1:
                while True:
                    if target_flag[i] == 1:  # or react_f.split(' ')[i] in ['-', '(', ')']:
                        break
                    i -= 1
            selected_indices.append((start_idx, i + 1))
            i = end_idx + 1
        else:
            i += 1

    new_react_f = []
    for start, end in selected_indices:
        new_react_f += react_f.split(' ')[start:end]

    new_center = ' '.join(prod_f.split(' ') + ['>>'] + new_react_f)

    react_f_token = react_f.split(' ')
    assigned_center = assign_am(new_center, postprocess=postprocess, sanitize_token=sanitize_token)
    assigned_prod_f, assigned_react_f = assigned_center.split('>>')
    assigned_react_f_token = smi_tokenizer(assigned_react_f)
    for start, end in selected_indices:
        react_f_token[start:end] = assigned_react_f_token[:end - start]
        assigned_react_f_token = assigned_react_f_token[end - start:]
    

    new_center_token = smi_tokenizer(assigned_prod_f) + ['>>'] + react_f_token
    if postprocess:
        
        for i in range(len(new_center_token)):
            new_center_token[i] = re.sub('&H1', 'H', new_center_token[i])
            new_center_token[i] = re.sub('&H', 'H', new_center_token[i])
            new_center_token[i] = re.sub('&', '', new_center_token[i])
    return ''.join(new_center_token)


def get_generalized_mapped_center(center, postprocess=False, sanitize_token=True):
    new_center = assign_am(center, postprocess=postprocess, sanitize_token=sanitize_token)
    new_center_token = smi_tokenizer(new_center)
    if postprocess:
        for i in range(len(new_center_token)):
            new_center_token[i] = re.sub('&H1', 'H', new_center_token[i])
            new_center_token[i] = re.sub('&H', 'H', new_center_token[i])
            new_center_token[i] = re.sub('&', '', new_center_token[i])
    return ''.join(new_center_token)


def get_mapped_center_main(center, postprocess=False, sanitize_token=True):
    '''
    Main function of tagging Atom-mapping for retro-template SMARTS; basic functionality is based on RXNMapper (https://github.com/rxn4chemistry/rxnmapper)
    :param center: retro template SMARTS (without space separated)
    :param postprocess: whether enable SMARTS token postprocessing
    :param sanitize_token: whether to sanitize token before the atom-mapping
    :return:
    '''
    if sanitize_token:
        try:
            c, ab = center.split('>>')
            c = Chem.MolToSmarts(Chem.MolFromSmarts(c))
            ab = Chem.MolToSmarts(Chem.MolFromSmarts(ab))
            new_sample = '{}>>{}'.format(ab, c)
            new_sample = new_sample.replace('-,:', '')
            new_sample = new_sample.replace(':&:', ':')
            new_sample = new_sample.replace('-&-', '-')
            new_sample = new_sample.replace('/&/', '/')
            new_sample = new_sample.replace('-&#', '-')
            new_sample = new_sample.replace('-&=', '-')

            new_center_token = smi_tokenizer(new_sample)
            for i in range(len(new_center_token)):
                new_center_token[i] = re.sub('&H1', 'H', new_center_token[i])
                new_center_token[i] = re.sub('&H', 'H', new_center_token[i])
                new_center_token[i] = re.sub('&', '', new_center_token[i])
            new_center = ''.join(new_center_token)
        except:
            new_center = '>>'.join(center.split('>>')[::-1])
    else:
        new_center = '>>'.join(center.split('>>')[::-1])

    try:
        gen_center = get_generalized_mapped_center(new_center, postprocess=postprocess, sanitize_token=sanitize_token)
        gen_center = '>>'.join(gen_center.split('>>')[::-1])
    except:
        gen_center = ''
    return gen_center

def reassign_am(tpl):
    tpl_token = smi_tokenizer(tpl)
    tpl_naked, atom_mapping = extract_mapping(' '.join(tpl_token))
    new_mapping = {'0':'0'}
    for am in atom_mapping.split(' '):
        if am != '0' and am not in new_mapping:
            new_mapping[am] = str(len(new_mapping))

    new_atom_mapping = ' '.join([new_mapping[am] for am in atom_mapping.split(' ')])
    new_tpl = reconstruct_mapping(tpl_naked, new_atom_mapping)
    return new_tpl.replace(' ','')

