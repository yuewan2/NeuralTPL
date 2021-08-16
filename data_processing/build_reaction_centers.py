import sys
sys.path.append('../')
import re
import os
import pandas as pd
from rdkit import Chem
from functools import partial
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import pickle
from rdchiral.template_extractor import extract_from_reaction
from utils.tpl_utils.test_mask_tpl import smi_tokenizer
from utils.data_utils import extract_mapping

BASE='/apdcephfs/private_yuewan/template_synthesis_dataset'

def find_center(inputs, prod_centers, inference=False):
    rxn, tpl = inputs
    if inference:
        assert tpl is None
    react_smiles, prod_smiles = rxn.split('>>')
    if tpl is not None:
        prod_patt, react_patt = tpl.split('>>')
        gt_match_atoms = prod_mol.GetSubstructMatch(Chem.MolFromSmarts(prod_patt))
        assert isinstance(gt_match_atoms, tuple)
        gt_match_atoms = sorted(set(gt_match_atoms))

    prod_mol, react_mol = Chem.MolFromSmiles(prod_smiles), Chem.MolFromSmiles(react_smiles)
    potential_reaction_centers = []
    for pc in prod_centers:
        patt = Chem.MolFromSmarts(pc)
        if inference:
            flag = prod_mol.HasSubstructMatch(patt)
        else:
            flag = prod_mol.HasSubstructMatch(patt) and react_mol.HasSubstructMatch(patt)
        if flag:
            pot_match_atoms = prod_mol.GetSubstructMatch(patt)
            assert isinstance(pot_match_atoms, tuple)
            if tpl is not None:
                overlap = False
                for atom in pot_match_atoms:
                    if atom in gt_match_atoms:
                        overlap = True
                        break
            if inference or not overlap:
                potential_reaction_centers.append(pc)

    return rxn, potential_reaction_centers

def main():
    raw_data = pd.read_csv(BASE + '/data/reaction/uspto-50k_tpl_modified.csv')
    raw_data = raw_data[raw_data['dataset'] != 'test']

    tasks = []
    prod_centers = set()
    rt2prod_centers = {i: set() for i in range(1, 11)}
    for i in tqdm(range(raw_data.shape[0])):
        rxn = raw_data.iloc[i]['rxn_smiles']
        react_smiles, prod_smiles = rxn.split('>>')
        reaction = {
            'reactants': react_smiles,
            'products': prod_smiles,
            '_id': '0'
        }
        try:
            tpl = extract_from_reaction(reaction)['reaction_smarts']
        except:
            continue

        rt = int(raw_data.iloc[i]['class'])
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
        print(prod_smarts)
        a=dsad
        
        
        prod_centers.add(prod_smarts)
        rt2prod_centers[rt].add(prod_smarts)

        tasks.append((rxn, tpl))

    print('Total Number of Product Centers:', len(prod_centers))
    with open(BASE + '/intermediate/prod_centers.pk', 'wb') as f:
        pickle.dump(prod_centers, f)
    with open(BASE + '/intermediate/rt2prod_centers.pk', 'wb') as f:
        pickle.dump(rt2prod_centers, f)
    results = process_map(partial(find_center, prod_centers=prod_centers), tasks)
    rxn2centers = {}
    for rxn, potential_reaction_centers in results:
        rxn2centers[rxn] = potential_reaction_centers
    with open(BASE + '/intermediate/rxn2centers.pk', 'wb') as f:
        pickle.dump(rxn2centers, f)


if __name__ == '__main__':
    main()