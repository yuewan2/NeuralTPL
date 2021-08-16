import re
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

from utils.data_utils import extract_mapping
from rdchiral.template_extractor import extract_from_reaction
from utils.tpl_utils.test_mask_tpl import smi_tokenizer

def convert_atom_to_wildcard(atom, super_general=False):
    assert super_general

    label = re.search('\:[0-9]+\]', atom.GetSmarts())
    if label:
        return '[*{}'.format(label.group())
    else:
        return '[*]'

def get_masked_mol_smarts_helper():
    return

def get_masked_mol_smarts(mol_smiles, patt_smarts, patt_smarts_extra=None,
                          atoms_to_use_guidance=None, return_atom_mapping=False, noise_only=False):
    '''
    extra_patt_smarts must be pre-verified that can match to mol_smiles!
    '''

    atoms_to_use_atom_mapping, atoms_to_use_atom_mapping_extra = [], []
    if atoms_to_use_guidance is not None:
        atoms_to_use_atom_mapping, atoms_to_use_atom_mapping_extra = atoms_to_use_guidance

    replace_symbol, replace_bond = '[#0:11111]', '•'
    replace_symbol_extra, replace_bond_extra = '[#0:99999]', '≈'
    if noise_only:
        replace_symbol, replace_bond = '[#0:99999]', '≈'

    mol = Chem.MolFromSmiles(mol_smiles)

    matched_atoms_list = mol.GetSubstructMatches(Chem.MolFromSmarts(patt_smarts))
    matched_atoms_list_extra = []
    if patt_smarts_extra is not None:
        matched_atoms_list_extra = mol.GetSubstructMatches(Chem.MolFromSmarts(patt_smarts_extra))
        if not len(matched_atoms_list_extra) and atoms_to_use_guidance is None:
            print('ERROR: fail to extract the proposed reaction center (without guidance)')
            return []
        elif not len(matched_atoms_list_extra):
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() in atoms_to_use_atom_mapping_extra:
                    matched_atoms_list_extra.append(atom.GetIdx())
            matched_atoms_list_extra = [matched_atoms_list_extra]

    if not len(matched_atoms_list) and atoms_to_use_guidance is None:
        print(mol_smiles, patt_smarts)
        print('ERROR: fail to extract the original reaction center (without guidance)')
        return []
    elif not len(matched_atoms_list):
        print('WARNING: using product masking to guide reactants masking')
        matched_atoms_list=[]
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in atoms_to_use_atom_mapping:
                matched_atoms_list.append(atom.GetIdx())
        matched_atoms_list = [matched_atoms_list]

    matched_atoms = matched_atoms_list[0]
    matched_atoms_extra = matched_atoms_list_extra[0] if len(matched_atoms_list_extra) else []

    # Modify molecule smiles based on matched atoms:
    bonds_to_replace, bonds_to_replace_extra = set(), set()
    atoms_to_use = []
    atom_symbols = []
    for atom in mol.GetAtoms():
        # Match Exact Reaction Center
        if atom.GetIdx() in matched_atoms: # Atoms corresponding to the <MASK> tag
            atom_symbols.append(replace_symbol)
            atoms_to_use_atom_mapping.append(atom.GetAtomMapNum())
            for neighbor_atom in atom.GetNeighbors():
                if neighbor_atom.GetIdx() in matched_atoms:
                    bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor_atom.GetIdx())
                    bonds_to_replace.update([bond.GetIdx()])
        elif atom.GetIdx() in matched_atoms_extra: # Atoms corresponding to the <NOISE> tag
            atom_symbols.append(replace_symbol_extra)
            atoms_to_use_atom_mapping_extra.append(atom.GetAtomMapNum())
            for neighbor_atom in atom.GetNeighbors():
                if neighbor_atom.GetIdx() in matched_atoms_extra:
                    bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor_atom.GetIdx())
                    bonds_to_replace_extra.update([bond.GetIdx()])
        else:
            atom_symbols.append(atom.GetSmarts())
        atoms_to_use.append(atom.GetIdx())

    bond_symbols = []
    for bond in mol.GetBonds():
        if bond.GetIdx() in bonds_to_replace:
            bond_symbols.append(replace_bond)
        elif bond.GetIdx() in bonds_to_replace_extra:
            bond_symbols.append(replace_bond_extra)
        else:
            bond_symbols.append(bond.GetSmarts())



    # Full Molecule Smarts
    full_mol = Chem.rdmolfiles.MolFragmentToSmiles(mol, atoms_to_use, allHsExplicit=True,
                                           isomericSmiles=True, allBondsExplicit=True)

    # Mask Molecule Smarts based on (potential) Reaction Center
    masked_mol = Chem.rdmolfiles.MolFragmentToSmiles(mol, atoms_to_use,
                                             atomSymbols=atom_symbols, allHsExplicit=True,
                                             isomericSmiles=True, allBondsExplicit=True,
                                             bondSymbols=bond_symbols)

    if return_atom_mapping:
        return full_mol, masked_mol, [atoms_to_use_atom_mapping, atoms_to_use_atom_mapping_extra]
    else:
        return full_mol, masked_mol

def convert_to_group_mask(masked_rxn, atom_wildcard, bond_wildcard, target_mask_token):
    '''masked_rxn: space seperated reaction (assume has reaction class token)'''
    mapped_rxn_tokens = masked_rxn.split(' ')
    wildcard_indices = []
    for i, token in enumerate(mapped_rxn_tokens):
        if token == atom_wildcard or token == bond_wildcard:
            if len(wildcard_indices) and i < wildcard_indices[-1] + 5:  # Relax the Wildcard indices range
                if '.' not in mapped_rxn_tokens[wildcard_indices[-1]:i] and '>>' not in mapped_rxn_tokens[wildcard_indices[-1]:i]:  # If has seperate, force it split
                    for j in range(wildcard_indices[-1] + 1, i):
                        wildcard_indices.append(j)
            wildcard_indices.append(i)
    new_mapped_rxn_tokens = []
    for i in range(len(mapped_rxn_tokens)):
        if i in wildcard_indices:
            if not len(new_mapped_rxn_tokens) or new_mapped_rxn_tokens[-1] != target_mask_token:
                new_mapped_rxn_tokens.append(target_mask_token)
            continue
        new_mapped_rxn_tokens.append(mapped_rxn_tokens[i])
    return ' '.join(new_mapped_rxn_tokens)


def main(data_folder):
    raw_data = pd.read_csv(data_folder+'/data/reaction/uspto-50k_tpl_modified.csv')

    mode = ['train', 'val', 'test']
    for m in mode:
        data = raw_data[raw_data['dataset'] == m]
        data.reset_index(inplace=True, drop=True)
        raw_rxn, full_rxn, masked_rxn, templates, rxn_tokens = [], [], [], [], []
        for i in tqdm(range(data.shape[0])):
            rt_token = '<RX_{}>'.format(data.iloc[i]['class'])
            rxn = data.iloc[i]['rxn_smiles']
            react_smiles, prod_smiles = rxn.split('>>')

            reaction = {
                'reactants': react_smiles,
                'products': prod_smiles,
                '_id': '0'
            }
            res = extract_from_reaction(reaction)
            if 'reaction_smarts' not in res or not res['reaction_smarts']:
                print('WARNING: fail to retrieve template')
                continue
            tpl = res['reaction_smarts']


            prod_smarts = tpl.split('>>')[0]
            prod_smarts, _ = extract_mapping(' '.join(smi_tokenizer(prod_smarts)))
            prod_smarts = ''.join(prod_smarts.split(' '))
            print(prod_smarts)
            # Canonical smarts:
            mol = Chem.MolFromSmiles(prod_smiles)
            selected_rcs_idx = sorted(set(mol.GetSubstructMatch(Chem.MolFromSmarts(prod_smarts))))
            atom_symbols = []
            for atom in mol.GetAtoms():
                if atom.HasProp('molAtomMapNumber'):
                    smarts = atom.GetSmarts()
                    am = re.match('.*(:[0-9]+).*', smarts).group(1)
                    atom_symbols.append(smarts.replace(am, ''))
            prod_smarts = Chem.MolFragmentToSmiles(mol, atomsToUse=selected_rcs_idx,
                                                   atomSymbols=atom_symbols, allBondsExplicit=True, allHsExplicit=True)
            
            
            
            
            template = ' '.join(smi_tokenizer(tpl))
            template, _ = extract_mapping(template)
            prod_patt, react_patt = ''.join(template.split(' ')).split('>>')
            if not Chem.MolFromSmiles(prod_smiles).HasSubstructMatch(Chem.MolFromSmarts(prod_patt)):
                print('WARNING: template not matched')
                continue

            prod, masked_prod, atoms_to_use_guidance = \
                get_masked_mol_smarts(prod_smiles, prod_patt, return_atom_mapping=True)
            react, masked_react = get_masked_mol_smarts(react_smiles, react_patt, atoms_to_use_guidance=atoms_to_use_guidance)

            merged_full_rxn = '{} >> {}'.format(' '.join(smi_tokenizer(prod)),
                                                ' '.join(smi_tokenizer(react)))
            merged_masked_rxn = '{} >> {}'.format(' '.join(smi_tokenizer(masked_prod)),
                                                  ' '.join(smi_tokenizer(masked_react)))

            raw_rxn.append(rxn)
            full_rxn.append(merged_full_rxn)
            masked_rxn.append(merged_masked_rxn)
            templates.append(template)
            rxn_tokens.append(rt_token)

        result = pd.DataFrame({'rxn_class': rxn_tokens,
                               'raw_rxn': raw_rxn,
                               'templates': templates,
                               'full_rxn': full_rxn,
                               'masked_rxn': masked_rxn})

        result.to_csv(data_folder+'/data/template/main_story_{}.csv'.format(m), index=False)





if __name__ == '__main__':
    data_folder = '/apdcephfs/private_yuewan/template_synthesis_dataset'
    main(data_folder)
