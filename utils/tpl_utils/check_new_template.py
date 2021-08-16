from rdchiral.main import rdchiralRunText
from rdkit import Chem
import pickle
import pandas as pd

if __name__ == '__main__':
    with open('../dataset/new_generate_tpl.pkl', 'rb') as f:
        new_generate_tpl = pickle.load(f)
    # new_generate_tpl_idx2tpl = {}
    # for key in new_generate_tpl.keys():
    #     new_generate_tpl_idx2tpl[key[0]] = [key[1],new_generate_tpl[key]]
    uspto_50k = pd.read_csv('../dataset/uspto-50k_tpl_modified.csv')

    uspto_50k_testset = uspto_50k.loc[uspto_50k['dataset'] == 'test']

    resort_template_test = uspto_50k_testset['resort_template'].tolist()
    mask_template_test = uspto_50k_testset['mask_template'].tolist()
    with open('../dataset/tpl_test_dic.pkl', 'rb') as f:
        tpl_test_dic = pickle.load(f)

    # check run rdrxn
    test_keys = sorted(list(new_generate_tpl.keys()), key=lambda x: x[0])
    for key in test_keys:
        mask_tpl = key[1]
        prod_list = new_generate_tpl[mask_tpl]
