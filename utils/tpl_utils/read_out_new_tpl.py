import re
import pickle


def read_source_and_prediction_data(src_fname, pre_fname, beam_size):
    # source data have atom map
    with open(src_fname, 'r', encoding='utf-8') as f:
        src_data = [''.join(x.strip().split(' ')) for x in f.readlines()]
    with open(pre_fname, 'r', encoding='utf-8') as f:
        pre_data = [''.join(x.strip().split(' ')) for x in f.readlines()]

    assert len(src_data) * beam_size == len(pre_data)

    new_tpl_dic = {(i, src): [] for i, src in enumerate(src_data)}

    for (i, src) in new_tpl_dic.keys():
        new_tpl_dic[(i, src)].extend(pre_data[beam_size * i:beam_size * (i + 1)])

    return new_tpl_dic


def get_new_tpl(mask_tpl, lvgp):
    assert '*' in lvgp
    # assert '<MASK>' in mask_tpl
    tpl_pd = mask_tpl.split('>>')[0]
    tpl_rt_center_list = re.sub('\<MASK\>', '', mask_tpl.split('>>')[1]).split('.')
    lvgp_list = lvgp.split('.')
    new_tpl_rt = ''
    star_cnt = 0
    for i, str_ in enumerate(lvgp):

        if str_ == '*':
            new_tpl_rt += tpl_rt_center_list[star_cnt]
            star_cnt += 1
        else:
            new_tpl_rt += str_
    return '{}>>{}'.format(tpl_pd, new_tpl_rt)


if __name__ == "__main__":
    new_tpl_dic = read_source_and_prediction_data('../transformer_model/data/USPTO-50K-tpl/src-test_map.txt',
                                                  '../transformer_model/experiments/results/predictions_USPTO-50K_model_step_150000.pt_on_USPTO-50K-tpl_beam10.txt',
                                                  10)

    new_tpl_suplit_token_dic = {}
    for (i, src) in new_tpl_dic.keys():
        new_tpl_suplit_token_dic[(i, src)] = []
        for pre in new_tpl_dic[(i, src)]:
            new_tpl_suplit_token_dic[(i, src)].extend(pre.split('<SPLIT>'))

    new_tpl_lvgp_dic = {}
    for (i, src) in new_tpl_suplit_token_dic.keys():
        new_tpl_lvgp_dic[(i, src)] = []
        for pre in new_tpl_suplit_token_dic[(i, src)]:
            if '>>' in pre:
                lvgp = pre.split('>>')[-1]
                if not re.findall('\:([0-9]+)\]', lvgp):
                    new_tpl_lvgp_dic[(i, src)].append(lvgp)
                else:
                    continue
            else:
                continue
        new_tpl_lvgp_dic[(i, src)] = list(set(new_tpl_lvgp_dic[(i, src)]))

    new_tpl_check_lvgp_dic = {}
    for (i, src) in new_tpl_lvgp_dic.keys():
        new_tpl_check_lvgp_dic[(i, src)] = []
        src_rt = src.split('>>')[-1]
        src_rt_split = src_rt.split('.')
        for lvgp in new_tpl_lvgp_dic[(i, src)]:
            if len(re.findall('\*', lvgp)) == len(src_rt_split):
                new_tpl_check_lvgp_dic[(i, src)].append(get_new_tpl(src, lvgp))
            else:
                continue

    with open('../dataset/new_generate_tpl.pkl', 'wb') as f:
        pickle.dump(new_tpl_check_lvgp_dic, f)
