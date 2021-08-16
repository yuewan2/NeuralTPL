import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdchiral.main import rdchiralRunText
from tqdm import tqdm
import multiprocessing
import numpy as np

uspto_50k = pd.read_csv('../dataset_gln/uspto-50k_tpl_modified.csv')
index_list = uspto_50k['templates'].index.tolist()
prod_smiles = uspto_50k['prod_smiles'].tolist()
templates = uspto_50k['templates'].tolist()
resort_templates = uspto_50k['resort_template'].tolist()
task_zip = list(zip(index_list, prod_smiles, templates, resort_templates))


def check_resort_result(task):
    index, prod, tpl, resort_tpl = task
    try:
        tpl = '(' + tpl.replace('>>', ')>>')
        org_result = rdchiralRunText(tpl, prod)
        org_result.sort()
        resort_tpl = '(' + resort_tpl.replace('>>', ')>>')
        modified_result = rdchiralRunText(resort_tpl, prod)
        modified_result.sort()
        if org_result != modified_result:
            return index, False, (org_result, modified_result)
        else:
            return index, True, (org_result, modified_result)
    except Exception as e:
        print(e)
        print(tpl)
        return




pool = multiprocessing.Pool(6)

check_results = []
for results in tqdm(pool.imap_unordered(check_resort_result, task_zip), total=len(task_zip)):
    if results is not None:
        check_results.append(results)
check_results.sort(key=lambda x: x[0])
change_list = [x for x in check_results if x[1] == 0]
