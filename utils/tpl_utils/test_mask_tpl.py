from rdkit import Chem
from rdkit.Chem import Draw
import re


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    #pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|<MASK>|>>|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|<MASK>|<unk>|>>|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|≈|•|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"

    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]

    assert smi == ''.join(tokens)
    return tokens


def mask_group(tpl, mask_type='leaving group'):
    tpl_pd, tpl_rt = tpl.split('>>')
    tokenizer_tpl_rt = smi_tokenizer(tpl_rt)
    bone_atom_index = []
    lvgp_atom_index = []
    all_index = []
    for index, token in enumerate(tokenizer_tpl_rt):
        all_index.append(index)
        if re.findall('\:([0-9]+)', token):
            bone_atom_index.append(index)
        elif re.findall('([a-zA-Z]+)', token):
            if index not in bone_atom_index:
                lvgp_atom_index.append(index)
        else:
            pass
    mask_index = []
    mask_index.extend(all_index[:min(bone_atom_index)])
    end_mask_flag = False
    for index, token in enumerate(tokenizer_tpl_rt[max(bone_atom_index) + 1:]):
        if re.findall('([a-zA-Z]+)', token):
            end_mask_flag = True
    if end_mask_flag:
        mask_index.extend(all_index[max(bone_atom_index) + 1:])

    for lvgp_idx in lvgp_atom_index:
        if min(bone_atom_index) < lvgp_idx < max(bone_atom_index):
            cur_idx_diff_list = [lvgp_idx - bone_idx for bone_idx in bone_atom_index]
            left = cur_idx_diff_list.index(min([i for i in cur_idx_diff_list if i > 0]))
            right = cur_idx_diff_list.index(max([i for i in cur_idx_diff_list if i < 0]))
            left_idx = bone_atom_index[left]
            right_idx = bone_atom_index[right]
            mask_index.extend(all_index[left_idx + 1:right_idx])
    mask_index = list(set(mask_index))
    mask_index.sort()
    mask_tokenizer_tpl_rt = []
    if mask_type == 'leaving group':
        for index, token in enumerate(tokenizer_tpl_rt):

            if index in mask_index:
                if mask_tokenizer_tpl_rt:
                    if mask_tokenizer_tpl_rt[-1] is not '<MASK>':
                        if token != '.':
                            mask_tokenizer_tpl_rt.append('<MASK>')
                        else:
                            mask_tokenizer_tpl_rt.append(token)
                    else:
                        if token != '.':
                            pass
                        else:
                            mask_tokenizer_tpl_rt.append(token)
                else:
                    if token != '.':
                        mask_tokenizer_tpl_rt.append('<MASK>')
                    else:
                        mask_tokenizer_tpl_rt.append(token)
            else:
                mask_tokenizer_tpl_rt.append(token)

        mask_tpl_rt = ''.join(mask_tokenizer_tpl_rt)

        map_pd = list(sorted(re.findall('\:([0-9]+)\]', tpl_pd)))
        map_rt = list(sorted(re.findall('\:([0-9]+)\]', mask_tpl_rt)))
        assert map_pd == map_rt
    elif mask_type == 'maped group':
        for index, token in enumerate(tokenizer_tpl_rt):
            if index not in mask_index:
                if mask_tokenizer_tpl_rt:
                    if mask_tokenizer_tpl_rt[-1] is not '*':
                        if token != '.':
                            mask_tokenizer_tpl_rt.append('*')
                        else:
                            mask_tokenizer_tpl_rt.append(token)
                    else:
                        if token != '.':
                            pass
                        else:
                            mask_tokenizer_tpl_rt.append(token)
                else:
                    if token != '.':
                        mask_tokenizer_tpl_rt.append('*')
                    else:
                        mask_tokenizer_tpl_rt.append(token)
            else:
                mask_tokenizer_tpl_rt.append(token)
        mask_tpl_rt = ''.join(mask_tokenizer_tpl_rt)
    else:
        raise ImportError('mask_type is "leaving group" or "maped group"')
    mask_tpl = '{}>>{}'.format(tpl_pd, mask_tpl_rt)

    return mask_tpl


if __name__ == '__main__':
    tpl = '[C;D1;H2:3]=[C:2]-[CH2;D2;+0:1]-[n;H0;D3;+0:7]1:[c:6]:[c:5]:[#7;a:4]:[c:8]:1>>Br-[C;H2;D2;+0:1]-[C:2]=[C;D1;H2:3].[n:4]1:[c:5]:[c:6]:[n;H1;D2;+0:7]:[c:8]:1'
    tpl_lvgp = mask_group(tpl, mask_type='leaving group')
    print(tpl_lvgp)

    # print(smi_tokenizer(tpl_lvgp.split('>>')[1]))
    print('1')
