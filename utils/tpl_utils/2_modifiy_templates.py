import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdchiral.main import rdchiralRunText
from tqdm import tqdm
import numpy as np
import re


def get_begain(mol):
    not_have_lvgp = True
    for atom in mol.GetAtoms():
        if not atom.HasProp('molAtomMapNumber'):
            not_have_lvgp = False
            for nei in atom.GetNeighbors():
                if nei.HasProp('molAtomMapNumber'):
                    return atom.GetIdx()
    if not_have_lvgp:
        return -1


def get_resort_tpl_rt(smarts, remove_leaving_group=False):
    mol = Chem.MolFromSmarts(smarts)
    smarts_begain_index = get_begain(mol)
    atomSymbols = [a.GetSmarts() for a in mol.GetAtoms()]
    if remove_leaving_group:
        atomsToUse = [a.GetIdx() for a in mol.GetAtoms() if a.HasProp('molAtomMapNumber')]
        smarts_begain_index = -1
    else:
        atomsToUse = [a.GetIdx() for a in mol.GetAtoms()]
    return AllChem.MolFragmentToSmiles(mol,
                                       atomsToUse=atomsToUse,
                                       allHsExplicit=False,
                                       isomericSmiles=True,
                                       atomSymbols=atomSymbols,
                                       allBondsExplicit=True,
                                       rootedAtAtom=smarts_begain_index)


def remove_leaving_group_and_resort_template(tpl, remove_leaving_group=False):
    if tpl is not np.nan:
        not_change_symbol = [
            '@', '#34'
        ]
        change_flag = True
        for s in not_change_symbol:
            if s in tpl:
                change_flag = False
        if change_flag:
            tpl_pd, tpl_rt = tpl.split('>>')
            tpl_rt_list = tpl_rt.split('.')
            resort_tpl_rt_list = [get_resort_tpl_rt(smarts, remove_leaving_group=remove_leaving_group) for smarts in
                                  tpl_rt_list]

            resort_tpl_rt = '.'.join(resort_tpl_rt_list)
            return '{}>>{}'.format(tpl_pd, resort_tpl_rt.replace('&', ';'))
        else:
            return tpl
    else:
        return


uspto_50k = pd.read_csv('../dataset_gln/uspto_50k_gln_w_tpl.csv')
templates = uspto_50k['templates'].tolist()
#
# test_tpl = '[C:5](-[#7:4])(=[O;D1;H0:6])-[C:7]-[N;H0;D3;+0:8](-[C:9])-[C;H0;D3;+0:1](-[Cl;D1;H0:2])=[O;D1;H0:3]>>[C;H0;D3;+0:1](-Cl)(-[Cl;D1;H0:2])=[O;D1;H0:3].[#7:4]-[C:5](=[O;D1;H0:6])-[C:7]-[NH;D2;+0:8]-[C:9]'
#
# # remove_leaving_group_and_resort_template(test_tpl, remove_leaving_group=True)
# # print(remove_leaving_group_and_resort_template(test_tpl, remove_leaving_group=True))

resort_template = [remove_leaving_group_and_resort_template(tpl, remove_leaving_group=False) for tpl in tqdm(templates)]
# remove_lvgp_template = [remove_leaving_group_and_resort_template(tpl, remove_leaving_group=True) for tpl in
                        # tqdm(templates)]

uspto_50k['resort_template'] = resort_template
# uspto_50k['remove_lvgp_template'] = remove_lvgp_template

uspto_50k.to_csv('../dataset_gln/uspto-50k_tpl_modified.csv', index=False)
