import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from rdchiral.template_extractor import extract_from_reaction

from functools import partial


def multi_run1(instance):
    tpl = None
    rxn = instance['rxn_smiles']
    react_smiles, prod_smiles = rxn.split('>>')
    reaction = {'reactants': react_smiles,
                'products': prod_smiles,
                '_id': '0'}
    result = extract_from_reaction(reaction)
    if 'reaction_smarts' in result:
        tpl = result['reaction_smarts']
    return tpl

def multi_run2(instance, tpls):
    rxn = instance['rxn_smiles']
    react_smiles, prod_smiles = rxn.split('>>')
    mol = Chem.MolFromSmiles(prod_smiles)
    count = 0
    for tpl in tpls:
        prod_patt, react_patt = tpl.split('>>')

        patt = Chem.MolFromSmarts(prod_patt)
        patt.UpdatePropertyCache(strict=False)
        if mol.HasSubstructMatch(patt):
            count += 1
    return count

raw_data = pd.read_csv('../data/reaction/uspto-50k_tpl_modified.csv')
tasks = []
tpls = []
for i in tqdm(range(raw_data.shape[0])):
    tasks.append(raw_data.iloc[i])
results = process_map(multi_run1, tasks, chunksize=100)
for res in results:
    if res is not None:
        tpls.append(res)
print(tpls[0])
tpls = list(set(tpls))
print(len(tpls))

counts = process_map(partial(multi_run2, tpls=tpls), tasks[:10])
print(np.mean(counts))

#
# prod2mtplcount = {}
# for i in tqdm(range(raw_data.shape[0])):
#     rxn = raw_data.iloc[i]['rxn_smiles']
#     react_smiles, prod_smiles = rxn.split('>>')
#     mol = Chem.MolFromSmiles(prod_smiles)
#     for tpl in tpls:
#         prod_patt, react_patt = tpl.split('>>')
#         patt = Chem.MolFromSmarts(prod_patt)
#         if mol.HasSubstructMatch(patt):
#             prod2mtplcount[prod_smiles] = prod2mtplcount.get(prod_smiles, 0) + 1
#
