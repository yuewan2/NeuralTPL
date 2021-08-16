import re
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from utils.tpl_utils.test_mask_tpl import smi_tokenizer

def remove_mapnum_rxn(raw_rxn):
    raw_rxn_unmapped_token, raw_rxn_am = extract_mapping(' '.join(smi_tokenizer(raw_rxn)))
    blank_rxn_am = []
    for am in raw_rxn_am.split(' '):
        if int(am) > 0:
            blank_rxn_am.append('1')
        else:
            blank_rxn_am.append('0')
    raw_rxn = reconstruct_mapping(raw_rxn_unmapped_token, ' '.join(blank_rxn_am))
    return raw_rxn.replace(' ', '')


def get_fragment_smarts(mol, selected_rcs_idx):
    atom_symbols = []
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            smarts = atom.GetSmarts()
            atom_symbols.append(smarts)

    [x.ClearProp('molAtomMapNumber') for x in mol.GetAtoms()]
    frag = AllChem.MolFragmentToSmiles(mol, atomsToUse=sorted(selected_rcs_idx),
                                       atomSymbols=atom_symbols, allBondsExplicit=True,
                                       allHsExplicit=True, isomericSmiles=False)
    return extract_mapping(' '.join(smi_tokenizer(frag)))[0].replace(' ', '')


def clear_map_number(smi, ver='smarts'):
    if ver == 'smarts':
        mol = Chem.MolFromSmarts(smi)
    else:
        mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.ClearProp('molAtomMapNumber')
    return Chem.MolToSmiles(mol)

def extract_mapping(sample_tpl):
    modified_sample_tpl = []
    mapping_sequence = []

    for i, token in enumerate(sample_tpl.split(' ')):
        if re.match('.*(:\d{1,})', token):
            modified_token = re.sub(':\d{1,}', '', token)
            modified_sample_tpl.append(modified_token)
            mapping_sequence.append(re.match('.*(:\d{1,})', token).group(1)[1:])
        else:
            modified_sample_tpl.append(token)
            mapping_sequence.append('0')
            
    return ' '.join(modified_sample_tpl), ' '.join(mapping_sequence)

def reconstruct_mapping(modified_sample_tpl, mapping_sequence):
    modified_token = modified_sample_tpl.split(' ')
    mapping_sequence = mapping_sequence.split(' ')
    assert len(modified_token) == len(mapping_sequence)
    sample_tpl = []
    for i, token in enumerate(modified_token):
        mapping_number = mapping_sequence[i]
        if int(mapping_number) > 0:
            sample_tpl.append(token[:-1] + ':{}]'.format(mapping_number))
        else:
            sample_tpl.append(token)
    return ' '.join(sample_tpl)


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

