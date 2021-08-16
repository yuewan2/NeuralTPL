import pandas as pd
import numpy as np
import pickle
from collections import defaultdict


def save_data(dic, fname):
    with open(fname, 'wb') as f:
        pickle.dump(dic, f)


def get_tpl_data_dic(mask_templates, prod_smiles, lvgp_templates):
    tpl_data_dic = {}
    for mask_tpl, prod, lvgp_tpl in list(zip(mask_templates, prod_smiles, lvgp_templates)):
        if mask_tpl not in tpl_data_dic.keys():
            tpl_data_dic[mask_tpl] = [[], []]
            if prod not in tpl_data_dic[mask_tpl][0]:
                tpl_data_dic[mask_tpl][0].append(prod)
            if lvgp_tpl not in tpl_data_dic[mask_tpl][1]:
                tpl_data_dic[mask_tpl][1].append(lvgp_tpl)
        else:
            if prod not in tpl_data_dic[mask_tpl][0]:
                tpl_data_dic[mask_tpl][0].append(prod)
            if lvgp_tpl not in tpl_data_dic[mask_tpl][1]:
                tpl_data_dic[mask_tpl][1].append(lvgp_tpl)
    return tpl_data_dic


if __name__ == '__main__':
    uspto_50k = pd.read_csv('../dataset/uspto-50k_tpl_modified.csv')
    train_data = uspto_50k.loc[uspto_50k['dataset'] == 'train']
    val_data = uspto_50k.loc[uspto_50k['dataset'] == 'val']
    test_data = uspto_50k.loc[uspto_50k['dataset'] == 'test']

    tpl_train_dic = get_tpl_data_dic(train_data['mask_template'].tolist(), train_data['prod_smiles'].tolist(),
                                     train_data['mask_template_only_leaving_group'].tolist())
    tpl_val_dic = get_tpl_data_dic(val_data['mask_template'].tolist(), val_data['prod_smiles'].tolist(),
                                   val_data['mask_template_only_leaving_group'].tolist())
    tpl_test_dic = get_tpl_data_dic(test_data['mask_template'].tolist(), test_data['prod_smiles'].tolist(),
                                    test_data['mask_template_only_leaving_group'].tolist())

    for key in tpl_train_dic.keys():
        if key in tpl_test_dic:
            tpl_test_dic.pop(key)
        if key in tpl_val_dic:
            tpl_val_dic.pop(key)
    tpl_train_dic.pop(np.nan)
    print(len(tpl_train_dic))
    print(len(tpl_val_dic))
    print(len(tpl_test_dic))
    save_data(tpl_train_dic,'../dataset/tpl_train_dic.pkl')
    save_data(tpl_val_dic, '../dataset/tpl_val_dic.pkl')
    save_data(tpl_test_dic, '../dataset/tpl_test_dic.pkl')