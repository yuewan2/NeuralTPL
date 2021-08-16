import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool

use_multi_core = True
core = 6


def get_lvgp(tpl):
    tpl_pd, tpl_rt = tpl.split('>>')
    mol_tpl_rt = Chem.MolFromSmarts(tpl_rt)
    rt_atomSymbols = [a.GetSmarts() for a in mol_tpl_rt.GetAtoms()]

    lvgp_index = [a.GetIdx() for a in mol_tpl_rt.GetAtoms() if not a.HasProp('molAtomMapNumber')]
    if not lvgp_index: return None
    return AllChem.MolFragmentToSmiles(mol_tpl_rt,
                                       atomsToUse=lvgp_index,
                                       allHsExplicit=False,
                                       isomericSmiles=True,
                                       atomSymbols=rt_atomSymbols,
                                       allBondsExplicit=True,
                                       canonical=True)


def get_cutoff_lvgp(all_items, cutoff=2):
    lvgp_cutoff_list = [x[0] for x in all_items if x[1] >= cutoff]
    CNTs_cutoff = [x[1] for x in all_items if x[1] >= cutoff]
    uspto_all_lvgp_cutoff_df = pd.DataFrame({'Leaving Group': lvgp_cutoff_list,
                                             'Count': CNTs_cutoff})
    print('{} Leaving Group Count >= {}'.format(len(lvgp_cutoff_list), cutoff))
    uspto_all_lvgp_cutoff_df.to_csv('../dataset_uspto_all/uspto_all_lvgp_cnt_cutoff{}.csv'.format(cutoff), index=False)


if __name__ == '__main__':
    data_uspto_all = pd.read_csv('../dataset_uspto_all/uspto_all.csv')
    retro_templates = data_uspto_all['retro_templates'].tolist()
    lvgp_dic = defaultdict(int)
    if use_multi_core:
        pool = Pool(core)
        for results in tqdm(pool.imap_unordered(get_lvgp, retro_templates), total=len(retro_templates)):
            lvgp = results
            lvgp_dic[lvgp] += 1
    else:
        for tpl in tqdm(retro_templates):
            lvgp = get_lvgp(tpl)
            lvgp_dic[lvgp] += 1

    lvgp_dic_items = list(sorted(lvgp_dic.items(), key=lambda x: x[1], reverse=True))
    get_cutoff_lvgp(lvgp_dic_items, 0)
    get_cutoff_lvgp(lvgp_dic_items, 50)
    get_cutoff_lvgp(lvgp_dic_items, 10)
    get_cutoff_lvgp(lvgp_dic_items, 2)
