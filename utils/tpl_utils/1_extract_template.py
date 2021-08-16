import pandas as pd
from rdkit import Chem
from tqdm import tqdm
from multiprocessing import Pool
from rdchiral.template_extractor import extract_from_reaction


def get_tpl(task):
    idx, row_idx, rxn_smiles = task
    react, reagent, prod = rxn_smiles.split('>')
    reaction = {'_id': row_idx, 'reactants': react, 'products': prod}
    template = extract_from_reaction(reaction)
    return idx, template


if __name__ == '__main__':
    dataset = pd.read_csv('../dataset_gln/uspto_50k_gln.csv')
    index = dataset.index.tolist()
    row_index = dataset['id'].tolist()
    rxn_smiles = dataset['rxn_smiles'].tolist()
    tasks = list(zip(index, row_index, rxn_smiles))

    pool = Pool(6)

    all_results = []
    for results in tqdm(pool.imap_unordered(get_tpl, tasks), total=len(tasks)):
        idx, template = results
        all_results.append(results)

    all_results.sort(key=lambda x: x[0])

    templates = [x[1]['reaction_smarts'] for x in all_results]

    dataset['templates'] = templates

    dataset.to_csv('../dataset_gln/uspto_50k_gln_w_tpl.csv', index=False)
