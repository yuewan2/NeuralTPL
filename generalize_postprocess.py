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
    #print('input:', ''.join(rxn_center.split(' ')))
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
    :param center: retro template smarts (without space separated)
    :param postprocess:
    :param sanitize_token:
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

def retrieve_positive_product(gen_center, raw_reaction_list):
    prod_patt = gen_center.split('>>')[0]
    prod_candidates = [rxn.split('>>')[1] for rxn in raw_reaction_list]

    patt = Chem.MolFromSmarts(prod_patt)
    prod_selected = []
    for prod_smiles in prod_candidates:
        mol = Chem.MolFromSmiles(prod_smiles)
        if mol.GetSubstructMatches(patt):
            prod_selected.append(prod_smiles)
    return prod_selected

def reassign_center(gen_center):
    gen_prod_frag_token = smi_tokenizer(gen_center.split('>>')[0])
    gen_react_frag_token = smi_tokenizer(gen_center.split('>>')[1])

    atom2am_prod = {}
    atom2pos_prod = {}
    for i, token in enumerate(gen_prod_frag_token):
        if token[0] == '[' and token[-1] == ']':
            token_atom = re.match('\[(.*):([0-9]+)\]', token).group(1)
            token_tag = re.match('\[(.*):([0-9]+)\]', token).group(2)
            token_atom = re.sub('H[1-9]?', '', token_atom).lower()
            atom2am_prod[token_atom] = atom2am_prod.get(token_atom, []) + [token_tag]
            atom2pos_prod[token_atom] = atom2pos_prod.get(token_atom, []) + [i]

    atom2am_react = {}
    atom2pos_react = {}
    for token in gen_react_frag_token:
        if token[0] == '[' and token[-1] == ']':
            token_atom = re.match('\[(.*):([0-9]+)\]', token).group(1)
            token_tag = re.match('\[(.*):([0-9]+)\]', token).group(2)
            token_atom = re.sub('H[1-9]?', '', token_atom).lower()
            atom2am_react[token_atom] = atom2am_react.get(token_atom, []) + [token_tag]
            atom2pos_react[token_atom] = atom2pos_react.get(token_atom, []) + [i]

    # 找出unique token:
    repeated_atom = []
    anchor_pos = -1

    for atom in atom2am_prod:
        if len(atom2am_prod[atom]) == 1:
            if atom2am_prod[atom][0] != atom2am_react[atom][0]:
                anchor_pos = atom2pos_prod[atom][0]
        else:
            repeated_atom.append(atom)

    if anchor_pos == -1:
        return gen_center

    nearest_pos = -1
    length = float('inf')
    for atom in repeated_atom:
        for pos in atom2pos_prod[atom]:
            if abs(anchor_pos-pos) < length:
                nearest_pos = pos
                length = abs(anchor_pos-pos)

    nearest_token = gen_prod_frag_token[nearest_pos]
    anchor_token = gen_prod_frag_token[anchor_pos]

    switched_am = re.match('\[(.*):([0-9]+)\]', nearest_token).group(2)
    gen_prod_frag_token[anchor_pos] = re.sub(':[1-9]+', ':{}'.format(switched_am), anchor_token)

    switched_am = re.match('\[(.*):([0-9]+)\]', anchor_token).group(2)
    gen_prod_frag_token[nearest_pos] = re.sub(':[1-9]+', ':{}'.format(switched_am), nearest_token)

    new_gen_center = '{}>>{}'.format(''.join(gen_prod_frag_token), ''.join(gen_react_frag_token))
    return new_gen_center


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

def transform_template(gen_center, data, verbose=False):
    #gen_center = get_mapped_center_main(center, postprocess=True)
    if verbose:
        print(gen_center)
    try:
        gen_center = reassign_center(gen_center)
    except:
        pass
    if verbose:
        print(gen_center)

    # Step 1: SubstructMatch the pattern with existing reaction/product
    prod_selected = retrieve_positive_product(gen_center, data['raw_rxn'])

    if verbose:
        print(len(prod_selected))
    # Step 2: For the matched product, build up inferred reaction
    rxn_candidates = []
    for prod in prod_selected:
        try:
            result = rdchiralRunText(gen_center, prod, keep_mapnums=True)
            if result:
                for inferred_react in result:
                    inferred_rxn = '{}>>{}'.format(inferred_react, prod)
                    rxn_candidates.append(inferred_rxn)
        except:
            pass

    if verbose:
        print(len(rxn_candidates))

    # Step 3: extract template from inferred reaction and select the most generalized one
    tpl_candidates = []
    for i in range(len(rxn_candidates)):
        reaction = {'_id': 0, 'reactants': rxn_candidates[i].split('>>')[0],
                    'products': rxn_candidates[i].split('>>')[1]}
        result = extract_from_reaction(reaction)
        if 'reaction_smarts' in result:
            template = result['reaction_smarts']
            tpl_candidates.append(template)

    if not tpl_candidates:
        return None
    simplest_tpl = tpl_candidates[0]
    for tpl in set(tpl_candidates):
        if len(smi_tokenizer(tpl)) < len(smi_tokenizer(simplest_tpl)):
            simplest_tpl = tpl

    return simplest_tpl

def multiprocess_center(gen_tpl, test_prods):
    # tpl = transform_template(gen_center, data)
    try:
        gen_tpl = reassign_center(gen_tpl)
    except:
        pass

    success_cases = []
    try:
        prod_f = gen_tpl.split('>>')[0]
        mol = Chem.MolFromSmarts(prod_f)
        for atom in mol.GetAtoms():
            if not atom.HasProp('molAtomMapNumber'):
                return []
    except:
        return []

    prod_pat = Chem.MolFromSmarts(gen_tpl.split('>>')[0])
    for prod_smiles in test_prods:
        prod_mol = Chem.MolFromSmiles(prod_smiles)
        if prod_mol.HasSubstructMatch(prod_pat):
            try:
                react_list = rdchiralRunText(gen_tpl, prod_smiles)
                if react_list:
                    for react_smiles in react_list:
                        rxn = '{}>>{}'.format(react_smiles, prod_smiles)
                        success_cases.append((rxn, gen_tpl))
            except:
                continue

    return success_cases

def main(process=True):
    # test_data = pd.read_csv('data/template/generalized_story_tst.csv')
    # test_prods = []
    # for rxn in test_data['raw_rxn']:
    #     reacts, prod = rxn.split('>>')
    #     test_prods.append(clear_atom_map(prod))
    # test_prods = list(set(test_prods))

    train_data = pd.read_csv('data/template/generalized_story_trn.csv')
    train_prods = []
    for rxn in train_data['raw_rxn']:
        reacts, prod = rxn.split('>>')
        train_prods.append(clear_atom_map(prod))
    train_prods = list(set(train_prods))


    #gen_data = pd.read_csv('result/test_generalize_generated_template_model_200000_wz.pt_bsz_10.csv')
    gen_data = pd.read_csv('result/train_generalize_generated_template_model_200000_wz.pt_bsz_10.csv')
    src2tgt = {}
    for i in range(gen_data.shape[0]):
        src = gen_data.iloc[i]['masked_rxn']
        if src not in src2tgt:
            src2tgt[src] = [gen_data.iloc[i]['generated_center']]
            # src2tgt[src] = [gen_data.iloc[i]['gt_center'], gen_data.iloc[i]['generated_center']]
        else:
            src2tgt[src].append(gen_data.iloc[i]['generated_center'])
            src2tgt[src] = list(set(src2tgt[src]))


    with open('novel_candidates.pk', 'rb') as f:
        novel_candidates = pickle.load(f)
        print('Number of Novel Candidates:', len(novel_candidates))

    src2tpl = {}
    # with open('gen_templates.pk', 'rb') as f:
    #     src2tpl = pickle.load(f)
    gen_centers = []
    for i, sample_key in tqdm(enumerate(src2tgt), total=len(src2tgt)):
        # if i < len(src2tpl)+42:
        #     continue
        if sample_key in src2tpl:
            continue
        for center in src2tgt[sample_key]:
            if center in novel_candidates:
                try:
                    gen_center = get_mapped_center_main(center, postprocess=True)
                    gen_centers.append(gen_center)
                    #gen_centers.append(center)
                except:
                    continue


    if process:
        print('Length of gen centers:', len(gen_centers))

        with Pool() as pool:
            results = process_map(partial(multiprocess_center, test_prods=train_prods),
                                  gen_centers)

        gen_templates = []
        for res in results:
            gen_templates += res

        with open('gen_templates_trn.pk', 'wb') as f:
            pickle.dump(gen_templates, f)

    else:
        print('Length of gen centers:', len(gen_centers))
        with open('gen_templates_trn_raw.pk', 'wb') as f:
            pickle.dump(gen_centers, f)




if __name__ == '__main__':
    main(process=False)
