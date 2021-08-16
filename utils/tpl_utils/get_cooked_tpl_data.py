import re
import pickle
from tqdm import tqdm


def tpl_tokenizer(tpl, remove_map=True):
    pattern = "(\>\>|\<MASK\>|\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(tpl)]

    assert tpl == ''.join(tokens)
    if remove_map:
        tokens = [re.sub('\:([0-9]+)\]', ']', token) for token in tokens]
    return ' '.join(tokens)


def load_data(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)


def get_cooked_data(data_dic, cooked_data_path_list, remove_map=True):
    cooked_data_src_f = open(cooked_data_path_list[0], 'w', encoding='utf-8')
    cooked_data_tgt_f = open(cooked_data_path_list[1], 'w', encoding='utf-8')
    for mask_tpl, (prod_list, tpl_lvgp_list) in tqdm(list(data_dic.items())):
        src = tpl_tokenizer(mask_tpl, remove_map=remove_map)
        cooked_data_src_f.write(src + '\n')
        tgt_tokenlizer_list = [tpl_tokenizer(tpl_lvgp, remove_map=remove_map) for tpl_lvgp in tpl_lvgp_list]
        tgt = ' <SPLIT> '.join(tgt_tokenlizer_list)
        cooked_data_tgt_f.write(tgt + '\n')
    cooked_data_src_f.close()
    cooked_data_tgt_f.close()

if __name__ == "__main__":
    cooked_data_dir = '../transformer_model/data/USPTO-50K-tpl/'
    tpl_train_dic = load_data('../dataset/tpl_train_dic.pkl')
    tpl_val_dic = load_data('../dataset/tpl_val_dic.pkl')
    tpl_test_dic = load_data('../dataset/tpl_test_dic.pkl')

    get_cooked_data(tpl_train_dic, [cooked_data_dir + 'src-train.txt', cooked_data_dir + 'tgt-train.txt'])
    get_cooked_data(tpl_val_dic, [cooked_data_dir + 'src-val.txt', cooked_data_dir + 'tgt-val.txt'])
    get_cooked_data(tpl_test_dic, [cooked_data_dir + 'src-test_nomap.txt', cooked_data_dir + 'tgt-test_nomap.txt'])
    get_cooked_data(tpl_test_dic, [cooked_data_dir + 'src-test_map.txt', cooked_data_dir + 'tgt-test_map.txt'],
                    remove_map=False)
