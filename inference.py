import pickle
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem

#from data_processing.build_generalize_data import get_masked_mol_smarts, convert_to_group_mask
from data_processing.build_reaction_centers import find_center
from rdchiral.main import rdchiralRunText
from rdchiral.template_extractor import extract_from_reaction
from utils.tpl_utils.generate_retro_templates import clear_mapnum
from utils.dataset import MainTplGenDataset, collate_fn
from utils.data_utils import extract_mapping, get_masked_mol_smarts, convert_to_group_mask
from utils.tpl_utils.test_mask_tpl import smi_tokenizer
from utils.translate_utils import translate_batch, explain_batch
from models.model import TemplateGenerator

BASE='/apdcephfs/private_yuewan/template_synthesis_dataset/'

device='cuda'
checkpoint_path = BASE+'./checkpoint_wnoise/generalize/model_52000_wz.pt'
print('Loading from {}'.format(checkpoint_path))
checkpoint = torch.load(checkpoint_path, map_location=device)
model = checkpoint['model'].to(device)
dataset_train = MainTplGenDataset(mode='train',
                                  data_folder=BASE+'./data/template',
                                  intermediate_folder=BASE+'./intermediate')
raw_data = pd.read_csv(BASE + 'data/reaction/uspto-50k_tpl_modified.csv')
products_train = raw_data['prod_smiles'][raw_data['dataset']=='train'].to_list()
with open(BASE+'./intermediate/rxn2centers.pk', 'rb') as f:
    rxn2centers = pickle.load(f)
with open(BASE+'./intermediate/prod_centers.pk', 'rb') as f:
    prod_centers = pickle.load(f)
with open(BASE+'./intermediate/rt2prod_centers.pk', 'rb') as f:
    rt2prod_centers = pickle.load(f)

cutoff_file = 'utils/tpl_utils/uspto_all_lvgp_cnt_cutoff50.csv'
cutoff_lvgps = set(pd.read_csv(cutoff_file)['Leaving Group'])

def whether_in_scoped(mapped_rxn, gt_templates):
    '''
    Check if the corresponding template exist in ground truth database
    '''
    reacts, prod = mapped_rxn.split('>>')
    reacts_mol = Chem.MolFromSmiles(reacts)
    clear_mapnum(reacts_mol)
    reacts_0 = Chem.MolToSmiles(reacts_mol, isomericSmiles=True)
    reacts_1 = Chem.MolToSmiles(reacts_mol, isomericSmiles=False)
    mol = Chem.MolFromSmiles(prod)
    for template in (gt_templates):
        prod_frag = template.split('>>')[0]
        patt = Chem.MolFromSmarts(prod_frag)
        if mol.HasSubstructMatch(patt):
            result = rdchiralRunText(template, prod)
            if reacts_0 in result or reacts_1 in result:
                return True
    return False

def lvgp_validity_check(gen_tpl, lvgp, lvgp_check):
    '''
    :param sample: generated template that passed basic sanity check
    :param lvgp_check: lvgp sanity check level (0: without any rules; 1: rule 1; 2: rule 1+2; 3: rule 1+2+3)
    :return:
    '''
    check_1, check_2, check_3 = False, False, False
    for prod in tqdm(products_train):
        pred_reacts = rdchiralRunText(gen_tpl, prod)
        if pred_reacts:
            if lvgp_check == 1:
                return True
            for pred_react in pred_reacts:
                if '[CH]' not in pred_react and lvgp_check == 2:
                    return True
                check_1, check_2 = True, True
                break
        if check_1 and check_2:
            break
    check_3 = lvgp in cutoff_lvgps
    return check_1 and check_2 and check_3







def basic_validity_check(raw_sample):
    '''
    Check whether the atom in product fragment exists in reactants fragment
    '''
    if len(raw_sample.split('>>')) != 2:
        return False
    prod_f, react_f = raw_sample.split('>>')
    try:
        prod_f, react_f = Chem.MolFromSmarts(prod_f), Chem.MolFromSmarts(react_f)
        prod_f_atoms = {}
        for atom in prod_f.GetAtoms():
            symbol = atom.GetSymbol()
            prod_f_atoms[symbol] = prod_f_atoms.get(symbol, 0) + 1
        react_f_atoms = {}
        for atom in react_f.GetAtoms():
            symbol = atom.GetSymbol()
            react_f_atoms[symbol] = react_f_atoms.get(symbol, 0) + 1
        for atom in prod_f_atoms:
            if atom not in react_f_atoms:
                return False
            if react_f_atoms[atom] < prod_f_atoms[atom]:
                return False
        return True
    except:
        return False

def run_inference(raw_rxn, rt_token=None, potential_rc=None, prod_frag_constraint=False,
                  verbose=False, beam_size=10, device='cpu', force_eos_token=None, regularize=False,
                  add_noise=True):
    '''
    Main function for running sample inference; only support one sample at a time
    :param raw_rxn: raw reaction SMILES, atom-mapping is not required unless regularize = False
    :param rt_token: reaction class token
    :param potential_rc: molecule fragment SMARTS; if None, randomly select one by find_center functions
    :param prod_frag_constraint: if force the product reaction center to be exactly the same as selected
    :param verbose: verbose level
    :param beam_size: number of candidate returned
    :param device: cpu/cuda
    :param force_eos_token: if change the eos token (mainly for debug purpose)
    :param regularize: if to reconstruct the initial template or to avoid the initial template
    :return:
    '''
    vocab_src_stoi = dataset_train.src_stoi
    vocab_tgt_stoi = dataset_train.tgt_stoi

    # Stage-1: make existed reaction center <NOISE> and target reaction center <MASK>
    # Step-1.1: extract template from existing atom-mapped reaction
    reaction = {'reactants': raw_rxn.split('>>')[0],
                'products': raw_rxn.split('>>')[1],
                '_id': '0'}
    prod_smiles = reaction['products']
    tpl = extract_from_reaction(reaction)['reaction_smarts']
    # Step-1.2: build mask reaction with token replacement corresponding to <NOISE>
    prod_patt, react_patt = tpl.split('>>')

    # Step-1.3: extract target reaction center if not given, add product fragment constraint to beam search
    prefix_sequence = None
    if potential_rc is None:
        if raw_rxn in rxn2centers:
            potential_reaction_centers = rxn2centers[raw_rxn]
        else:
            _, potential_reaction_centers = find_center((raw_rxn, None), prod_centers, inference=True)
        if verbose:
            print('Num potential reaction centers: ', len(potential_reaction_centers))
        potential_rc = np.random.choice(potential_reaction_centers)
    if verbose:
        print('\nProposed reaction center:')
        print(potential_rc)

    if prod_frag_constraint:
        if regularize:
            potential_rc_simplify = extract_mapping(' '.join(smi_tokenizer(prod_patt)))[0].split(' ') + ['>>']
        else:
            potential_rc_simplify = extract_mapping(' '.join(smi_tokenizer(potential_rc)))[0].split(' ') + ['>>']
        prefix_sequence = (''.join(potential_rc_simplify[:-1]), [[vocab_tgt_stoi.get(t, vocab_tgt_stoi['<unk>']) for t in potential_rc_simplify]])

    # Step-1.4: replacing the corresponding token with customized wildcard (NOTE: potential_rc and prod_patt are flipped)
    if regularize:
        prod, masked_prod, atoms_to_use_guidance = \
            get_masked_mol_smarts(prod_smiles, potential_rc, return_atom_mapping=True, noise_only=True)
        if add_noise:
            _, pseudo_react, _ = \
                get_masked_mol_smarts(prod_smiles, prod_patt, potential_rc, return_atom_mapping=True, noise_only=False)
        else:
            _, pseudo_react, _ = \
                get_masked_mol_smarts(prod_smiles, prod_patt, return_atom_mapping=True, noise_only=False)
            masked_prod = prod
    else:
        prod, masked_prod, atoms_to_use_guidance = \
            get_masked_mol_smarts(prod_smiles, prod_patt, return_atom_mapping=True, noise_only=True)
        if add_noise:
            _, pseudo_react, _ = \
                get_masked_mol_smarts(prod_smiles, prod_patt, potential_rc, return_atom_mapping=True, noise_only=False)
        else:
            #print('WARNING: regularize=False with add_noise=False is not recommended!')
            _, pseudo_react, _ = \
                get_masked_mol_smarts(prod_smiles, potential_rc, return_atom_mapping=True, noise_only=False)
            masked_prod = prod
    merged_masked_rxn = ' '.join(smi_tokenizer('{}>>{}'.format(masked_prod, pseudo_react)))

    # Step-1.5: group the <MASK> and <NOISE> tag
    merged_masked_rxn_final = convert_to_group_mask(merged_masked_rxn, '[#0:11111]', '•', '<MASK>')
    merged_masked_rxn_final = convert_to_group_mask(merged_masked_rxn_final, '[#0:99999]', '≈', '<NOISE>')

    if verbose:
        print('\nFinal mask input:')
        print(''.join(merged_masked_rxn_final.split(' ')))

    # ---------------------------------version2---------------------------------
    # merged_masked_rxn_final = '<CLS> {} '.format(rt_token) + merged_masked_rxn_final
    # template_tokens = ['<sos>']
    # ---------------------------------version1---------------------------------
    merged_masked_rxn_final = '<CLS> ' + merged_masked_rxn_final
    if rt_token is not None:
        template_tokens = [rt_token]
    else:
        template_tokens = ['<sos>']

    # Stage-2: prepare model inputs
    src, src_am = extract_mapping(merged_masked_rxn_final)
    src = np.array([vocab_src_stoi.get(t, vocab_src_stoi['<unk>']) for t in src.split(' ')])
    src_am = np.array(src_am.split(' '), dtype=int)
    tgt = np.array([vocab_tgt_stoi.get(t, vocab_tgt_stoi['<unk>']) for t in template_tokens])
    reassignment = {}
    for am in src_am:
        if am != 0 and am not in reassignment:
            reassignment[am] = len(reassignment) + 1
    src_am = np.array([reassignment.get(am, 0) for am in src_am])
    pads = dataset_train.src_stoi['<pad>'], dataset_train.tgt_stoi['<pad>']
    sep = dataset_train.src_stoi['>>']
    data = [[[src], [src_am], [tgt], [-1]]] # the last one is not Important
    src, src_am, src_seg, tgt, rt_label = collate_fn(data, sep, pads, device=device, inference=True)
    inputs = src.to(device), src_am.to(device), src_seg.to(device), tgt.to(device) # model inputs

    # Stage-3: run inference:
    model.to(device)
    eos_token = '<eos>' if force_eos_token is None else force_eos_token
    pred_tokens, pred_scores = translate_batch(model, inputs,
                                               eos_idx=dataset_train.tgt_stoi[eos_token],
                                               sos_idx=dataset_train.tgt_stoi['<sos>'],
                                               sep_idx=dataset_train.tgt_stoi['>>'], generalize=True,
                                               prefix_sequence=prefix_sequence,
                                               inference=True, multiclass=True,
                                               beam_size=beam_size,
                                               fixed_z = True)
    rt_batch, mask_rxn_batch, gtruth_batch, hypos_batch, score_batch = \
        explain_batch(src, tgt, pred_tokens, pred_scores,
                      dataset_train.src_itos, dataset_train.tgt_itos, return_score=True)

    return hypos_batch[0], score_batch[0]


def retrieve_embeddings(templates, model=model, tgt_stoi=dataset_train.tgt_stoi, batch_capacity=10, device='cpu'):
    '''
    Retrieve template embedding and predicted reaction class from model's prior_encoder
    :param templates: list of template (not space separated)
    '''
    assert isinstance(templates, list)
    max_length = 0
    prior_tgt_list = []
    for template in templates:
        template_tokens = smi_tokenizer(template)
        template_tokens = ['<CLS>'] + template_tokens + ['<eos>']
        prior_tgt = torch.LongTensor([tgt_stoi.get(t, tgt_stoi['<unk>']) for t in template_tokens])
        prior_tgt_list.append(prior_tgt)
        max_length = max(max_length, len(prior_tgt))
    prior_tgts = torch.zeros((max_length, len(templates))).fill_(tgt_stoi['<pad>']).long()
    for i in range(len(templates)):
        prior_tgts[:len(prior_tgt_list[i]), i] = prior_tgt_list[i]

    embeddings = None
    predicted_reaction_class = None
    model.to(device)
    model.eval()
    for i in tqdm(range(len(templates) // batch_capacity + 1)):
        prior_tgt_batch = prior_tgts[:, i * batch_capacity:(i + 1) * batch_capacity]
        if not prior_tgt_batch.shape[1]:
            break
        with torch.no_grad():
            _, post_encoder_out = model.encoder_prior(prior_tgt_batch.to(device))
            rt_scores = model.classifier(post_encoder_out[0])
        _, pred_rc = rt_scores.topk(1, dim=-1)

        if embeddings is None:
            embeddings = post_encoder_out[0].cpu().numpy()
            predicted_reaction_class = pred_rc.squeeze(1).cpu().numpy()
        else:
            embeddings = np.concatenate([embeddings, post_encoder_out[0].cpu().numpy()])
            predicted_reaction_class = np.concatenate([predicted_reaction_class, pred_rc.squeeze(1).cpu().numpy()])

    predicted_reaction_class = ['<RX_{}>'.format(i+1) for i in predicted_reaction_class]
    return embeddings, predicted_reaction_class

def generate(prod_frag_constraint, verbose=False, filter=True, reaction_class=None, k=10):
    '''
    :param prod_frag_constraint:
    :param verbose:
    :param filter:
    :param reaction_class: e.g. (<RX_1>, <RX_2>, <RX_3>, ..., <RX_10>)
    :return:
    '''
    tst_data = pd.read_csv('data/template/main_story_test.csv')
    raw_rxn_list = list(set(tst_data['raw_rxn']))
    generated_templates, rt_prod_centers = [], []

    for raw_rxn in tqdm(raw_rxn_list):
        reaction = {'reactants': raw_rxn.split('>>')[0],
                    'products': raw_rxn.split('>>')[1],
                    '_id': '0'}
        tpl = extract_from_reaction(reaction)['reaction_smarts']
        if raw_rxn in rxn2centers:
            potential_reaction_centers = rxn2centers[raw_rxn]
        else:
            _, potential_reaction_centers = find_center((raw_rxn, tpl), prod_centers)

        for rt in [1,2,3,4,5,6,7,8,9,10]:
            if reaction_class is not None:
                if rt != int(reaction_class[4:-1]):
                    continue
            rt_prod_centers = rt2prod_centers[rt]
            potential_rcs = []
            for potential_rc in potential_reaction_centers:
                if potential_rc in rt_prod_centers:
                    potential_rcs.append(potential_rc)

            if not len(potential_rcs):
                continue

            rt_token = '<RX_{}>'.format(rt)
            potential_rcs = np.random.choice(potential_rcs,
                                             size=min(len(potential_rcs), 3), replace=False)

            for potential_rc in potential_rcs:
                inferred_templates = run_inference(raw_rxn, rt_token,
                                                    potential_rc=potential_rc,
                                                    prod_frag_constraint=prod_frag_constraint,
                                                    verbose=verbose, beam_size=k)[0]
                if filter:
                    for gen_tpl in set(inferred_templates):
                        if basic_validity_check(gen_tpl):
                            generated_templates.append('{} {}'.format(rt_token, gen_tpl))
                else:
                    generated_templates += ['{} {}'.format(rt_token, gen_tpl) for gen_tpl in set(inferred_templates)]

        generated_templates = list(set(generated_templates))

    if reaction_class is None:
        file_name = 'result/generated_templates_v3.txt'
    else:
        file_name = 'result/generated_templates_RX{}.txt'.format(reaction_class)

    with open(file_name, 'w') as f:
        for gen_tpl in generated_templates:
            f.write(gen_tpl)
            f.write('\n')
    # print(generated_templates)
    return

def build_from_lvgps(input_path, output_path='', lvgp_check=0):
    '''

    :param input_path: file path for the generated lvgp results pickle file
    :param output_path: file path for the output generated txt file
    :param lvgp_check: lvgp sanity check level (0: without any rules; 1: rule 1; 2: rule 1+2; 3: rule 1+2+3)
    :return:
    '''
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    masked_src, react_class, gtruth, hypos = data

    # Build templates from generated leaving groups:
    reconstructed_novel_templates = set()
    for i in range(len(masked_src)):
        masked_src_i = masked_src[i]
        hypos_i = hypos[i]
        gt_i = gtruth[i]
        backbone_token = smi_tokenizer(masked_src_i)
        num_mask = sum(np.array(backbone_token) == '<MASK>')
        reconstruct_hypos_i = []
        for hypo in hypos_i:
            if hypo == gt_i:
                continue
            new_token_list = []
            gen_lvgps = hypo.split('.')
            if num_mask != len(gen_lvgps):
                continue
            for token in backbone_token:
                if token != '<MASK>':
                    new_token_list.append(token)
                else:
                    new_token_list += smi_tokenizer(gen_lvgps.pop(0))

            gen_tpl = ''.join(new_token_list)
            if basic_validity_check(gen_tpl):
                if lvgp_check == 0 or lvgp_validity_check(gen_tpl, hypo, lvgp_check):
                    reconstruct_hypos_i.append(gen_tpl)
        reconstructed_novel_templates.update(reconstruct_hypos_i)
    reconstructed_novel_templates = list(reconstructed_novel_templates)

    if not output_path:
        output_path = 'result/generated_templates_from_lvgps.txt'

    with open(output_path, 'w') as f:
        for gen_tpl in reconstructed_novel_templates:
            f.write(gen_tpl)
            f.write('\n')

if __name__ == '__main__':
    # Load raw data and retrieve test reactions
    raw_rxn_atom_mapped = \
        '[CH:1]#[C:2][c:3]1[cH:4][cH:5][cH:6][c:7]([NH2:8])[cH:29]1.[c:9]1([Cl:30])[n:10][cH:11][n:12][c:13]2[cH:14][c:15]3[c:16]([cH:17][c:18]12)[O:19][CH2:20][CH2:21][O:22][CH2:23][CH2:24][O:25][CH2:26][CH2:27][O:28]3>>[CH:1]#[C:2][c:3]1[cH:4][cH:5][cH:6][c:7]([NH:8][c:9]2[n:10][cH:11][n:12][c:13]3[cH:14][c:15]4[c:16]([cH:17][c:18]23)[O:19][CH2:20][CH2:21][O:22][CH2:23][CH2:24][O:25][CH2:26][CH2:27][O:28]4)[cH:29]1.[ClH:30]'

    potential_rc = '[CH]#[C]-[c](:[cH]):[cH]'
    rt_token = '<RX_3>'
    print(rt_token)
    generated_templates = run_inference(raw_rxn_atom_mapped, rt_token,
                                        potential_rc=None, prod_frag_constraint=True, verbose=True)
    print('Generated templates:')
    for gen_tpl in generated_templates[0]:
        print(gen_tpl)

    # generate(prod_frag_constraint=True, filter=True, reaction_class=None, k=5)
