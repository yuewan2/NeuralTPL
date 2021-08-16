import math
import copy
import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from utils.data_utils import reconstruct_mapping

def var(a):
    return a.clone().detach()
    #return torch.tensor(a, requires_grad=False)

def rvar(a, beam_size=10):
    if len(a.size()) == 3:
        return var(a.repeat(1, beam_size, 1))
    else:
        return var(a.repeat(1, beam_size))

def translate_batch(model, inputs, eos_idx, sos_idx, sep_idx=-1, beam_size=10, max_length=300, invalid_token_indices=[],
                    fixed_z=False, generalize=False, seed=None, target_mask_num=None, prefix_sequence=None,
                    inference=False, multiclass=False):
    '''
    :param inputs: tuple of (src, src_am, src_seg, tgt), tgt is only used to retrieve conditional reaction class token
    :param fixed_z: latent variable flag
    :param seed: latent variable flag
    :param target_mask_num: available only when generalize=False; constraint the amount of generated fragment = num of <MASK>
    :param sep_idx: target seperator '>>' index, only use when generalize=True; constraint the beam search from getting seperator too early
    :param prefix_sequence: list of prefix tokens, only use in customized template generation stage
    :return:
    '''

    src, src_am, src_seg, tgt = inputs

    batch_size = inputs[0].shape[1]

    model.eval()

    start_step = 0
    constraint_step = 2 # Avoid token that end earily
    pred_tokens = torch.ones((batch_size, beam_size, max_length + 1)).long()

    # ---------------------------------version2---------------------------------
    # pred_tokens[:, :, 0] = sos_idx
    # ---------------------------------version1---------------------------------
    pred_tokens[:, :, 0] = tgt[0].repeat(beam_size, 1).transpose(0, 1)
    # ------------------------------------------------------------------

    if prefix_sequence is not None:
        prefix_rc, prefix_sequence = prefix_sequence
        # Only support one sample at a time
        assert len(prefix_sequence) == src.shape[1] == 1
        pred_tokens[:, :, 1:len(prefix_sequence[0]) + 1] = torch.LongTensor(prefix_sequence[0])
        start_step = len(prefix_sequence[0])

        # Calculate the number of atoms and bonds in prefix_rc
        patt = Chem.MolFromSmarts(prefix_rc)
        constraint_step = start_step + patt.GetNumBonds() + patt.GetNumAtoms() + 1

    pred_tokens = pred_tokens.to(src.device)
    pred_scores = torch.zeros((batch_size, beam_size)).to(src.device)
    batch2finish = {i: False for i in range(batch_size)}

    with torch.no_grad():
        _, prior_encoder_out = model.encoder(src, src_seg, src_am)
        _, z_repeat = model.latent_net(var(prior_encoder_out[0].repeat(beam_size, 1)),
                                       None, train=False, fixed_z=fixed_z, seed=seed)

    src_repeat = rvar(src.data, beam_size=beam_size)
    memory_bank_repeat = rvar(prior_encoder_out.data, beam_size=beam_size)

    state_cache = {}
    for step in range(start_step, max_length):
        torch.cuda.empty_cache()
        inp = pred_tokens.transpose(0, 1).contiguous().view(-1, pred_tokens.size(2))[:, :step + 1].transpose(0, 1).to(src.device)

        with torch.no_grad():
            outputs, attn = model.decoder(src_repeat, inp, memory_bank_repeat, z_repeat, state_cache=state_cache, step=step)
            scores = model.generator(outputs[-1])

        unbottle_scores = scores.view(beam_size, batch_size, -1)

        # Avoid invalid token:
        unbottle_scores[:, :, invalid_token_indices] = -1e25

        # Avoid token that end earily
        constraint_step = start_step + 2
        if step < constraint_step:
            unbottle_scores[:, :, eos_idx] = -1e25
            if generalize and sep_idx != -1:
                unbottle_scores[:, :, sep_idx] = -1e25


        # Beam Search:
        selected_indices = []
        for j in range(batch_size):
            prev_score = copy.deepcopy(pred_scores[j])
            batch_score = unbottle_scores[:, j]
            num_words = batch_score.size(1)
            # Get previous token to identify <eos>
            prev_token = pred_tokens[j, :, step]
            eos_index = prev_token.eq(eos_idx)
            # Prevent <eos> sequence to have children
            prev_score[eos_index] = -1e20

            if beam_size == eos_index.sum():  # all beam has finished
                pred_tokens[j, :, step + 1] = eos_idx
                batch2finish[j] = True
                selected_indices.append(torch.LongTensor(np.arange(beam_size)).to(src.device))
            else:
                # Avoid end early or late based on num_mask
                if target_mask_num is not None:
                    # if step != 0 and target_mask_num is not None:
                    #     target_mn = target_mask_num[j]
                    #     current_mns = pred_tokens[j].eq(13).sum(dim=1)
                    #
                    #     for jk, mn in enumerate(current_mns):
                    #         # If num mask not reached or the last token is '.', cannot stop
                    #         if mn < target_mn - 1 or pred_tokens[j, jk, step] == 13:
                    #             batch_score[jk][3] = -1e25
                    #         if mn == target_mn - 1:
                    #             batch_score[jk][13] = -1e20
                    pass

                beam_scores = batch_score + prev_score.unsqueeze(1).expand_as(batch_score)

                if step == start_step:
                    flat_beam_scores = beam_scores[0].view(-1)
                else:
                    flat_beam_scores = beam_scores.view(-1)

                # Select the top-k highest accumulative scores
                k = beam_size - eos_index.sum()
                best_scores, best_scores_id = flat_beam_scores.topk(k, 0, True, True)

                # Freeze the tokens which has already finished
                frozed_tokens = pred_tokens[j][eos_index]
                if frozed_tokens.shape[0] > 0:
                    frozed_tokens[:, step + 1] = eos_idx
                frozed_scores = pred_scores[j][eos_index]

                # Update the rest of tokens
                origin_tokens = pred_tokens[j][best_scores_id // num_words]
                # origin_tokens = pred_tokens[j][torch.div(best_scores_id, num_words)]
                origin_tokens[:, step + 1] = best_scores_id % num_words

                updated_scores = torch.cat([best_scores, frozed_scores])
                updated_tokens = torch.cat([origin_tokens, frozed_tokens])

                pred_tokens[j] = updated_tokens
                pred_scores[j] = updated_scores

                if eos_index.sum() > 0:
                    tmp_indices = torch.zeros(beam_size).long().to(src.device)
                    tmp_indices[:len(best_scores_id // num_words)] = (best_scores_id // num_words)
                    selected_indices.append(tmp_indices)
                else:
                    selected_indices.append((best_scores_id // num_words))

        if selected_indices:
            reorder_state_cache(state_cache, selected_indices)

        if sum(batch2finish.values()) == len(batch2finish):
            break

    # (Sorting is done in explain_batch)
    return pred_tokens, pred_scores


# Reorder state_cache:
def reorder_state_cache(state_cache, selected_indices):
    '''
    params state_cache: list of indices
    params selected_indices: size (batch_size x beam_size)
    '''
    batch_size, beam_size = len(selected_indices), len(selected_indices[0])
    indices_mapping = torch.arange(batch_size * beam_size).reshape(beam_size, batch_size).transpose(0, 1).to(
        selected_indices[0].device)
    reorder_indices = []
    for batch_i, indices in enumerate(selected_indices):
        reorder_indices.append(indices_mapping[batch_i, indices])
    reorder_indices = torch.stack(reorder_indices, dim=1).view(-1)

    new_state_cache = []
    for key in state_cache:
        if isinstance(state_cache[key], dict):
            for subkey in state_cache[key]:
                state_cache[key][subkey] = state_cache[key][subkey][reorder_indices]

        elif isinstance(state_cache[key], torch.Tensor):
            state_cache[key] = state_cache[key][reorder_indices]
        else:
            raise Exception


def explain(seq_i, vocab_itos, atommapping=None):
    '''
    :param seq_i: a 1-dimension predicted sequence
    '''
    if atommapping is not None:
        atommapping = atommapping[1:]
        explained_token = ' '.join([vocab_itos[t.item()] for t in seq_i[1:] if vocab_itos[t.item()] not in ['<eos>', '<pad>']])
        atommapping = ' '.join([str(a.item()) for a in atommapping[:len(explained_token.split(' '))]])
        explained = reconstruct_mapping(explained_token, atommapping).replace(' ', '')
        return vocab_itos[seq_i[0]], explained
    else:
        return vocab_itos[seq_i[0]], ''.join([vocab_itos[t.item()] for t in seq_i[1:] if vocab_itos[t.item()] not in ['<eos>', '<pad>']])


def explain_batch(src, src_am, tgt, pred_tokens, pred_scores, vocab_itos_src, vocab_itos_tgt, generalize=True, return_score=False):
    eos_idx = np.argwhere(np.array(vocab_itos_tgt) == '<eos>').flatten()[0]
    score_batch, gtruth_batch, hypos_batch, mask_rxn_batch, rt_batch = [], [], [], [], []
    batch_size, beam_size = pred_tokens.shape[0], pred_tokens.shape[1]
    for i in range(batch_size):
        pred_score_i = pred_scores[i]
        pred_token_i = pred_tokens[i].cpu().numpy()

        gtruth_batch.append(explain(tgt[:, i], vocab_itos_tgt)[1])
        mask_rxn_batch.append(explain(src[:, i], vocab_itos_src, atommapping=src_am[:, i])[1])
        hypos_i = []
        for j in range(beam_size):
            rc_token, hypo = explain(pred_token_i[j], vocab_itos_tgt)
            hypos_i.append(hypo)
            pred_score_i[j] = pred_score_i[j]# / (pred_token_i[j] != eos_idx).sum()

        rt_batch.append(rc_token)
        sorted_index = np.argsort(pred_score_i.cpu().numpy())[::-1]
        hypos_i = list(np.array(hypos_i)[sorted_index])
        hypos_batch.append(hypos_i)
        score_batch.append(pred_score_i.cpu().numpy()[sorted_index])

    if return_score:
        return rt_batch, list(mask_rxn_batch), list(gtruth_batch), list(hypos_batch), list(score_batch)
    else:
        return rt_batch, list(mask_rxn_batch), list(gtruth_batch), list(hypos_batch)