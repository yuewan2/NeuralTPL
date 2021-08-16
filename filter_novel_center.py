import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
from multiprocessing.pool import Pool
from functools import partial

class bbIdentifier:
    def __init__(self, gt_backbone_identifiers):
        gt_backbone_identifiers = list(set(gt_backbone_identifiers))
        self.gt_backbone_identifiers = []
        token_dict = set()
        for bb in gt_backbone_identifiers:
            # bb_token = smi_tokenizer(bb)
            bb_token = bb.split(' ')
            token_dict.update(bb_token)
        self.token_dict = {t: i for i, t in enumerate(list(token_dict))}

        gt_identifiers_prod, gt_identifiers_react = [], []
        for gt_bb in gt_backbone_identifiers:
            if '>>' not in gt_bb or len(gt_bb.split(' >> ')) != 2:
                continue
            gt_prod, gt_react = gt_bb.split(' >> ')
            gt_identifiers_prod.append(self.get_id(gt_prod))
            gt_identifiers_react.append(self.get_id(gt_react))
            self.gt_backbone_identifiers.append(gt_bb)
        self.gt_identifiers_prod = np.array(gt_identifiers_prod)
        self.gt_identifiers_react = np.array(gt_identifiers_react)

    def get_id(self, part):
        part_token = part.split(' ')
        gt_id = np.zeros(len(self.token_dict))
        for token in part_token:
            if token in self.token_dict:
                gt_id[self.token_dict[token]] += 1
        return gt_id

    def get_similar(self, bb, topk=50):
        # bb: \s seperated center
        # bb_token = smi_tokenizer(bb)
        if '>>' not in bb or len(bb.split(' >> ')) != 2:
            return None, None
        gen_prod, gen_react = bb.split(' >> ')
        gen_prod_id = self.get_id(gen_prod)
        gen_react_id = self.get_id(gen_react)
        scores_prod = cosine_similarity(gen_prod_id.reshape(1, -1), self.gt_identifiers_prod).reshape(-1)
        scores_react = cosine_similarity(gen_react_id.reshape(1, -1), self.gt_identifiers_react).reshape(-1)
        scores = scores_prod * scores_react
        sorted_idx = np.argsort(scores)[::-1][:topk]

        return np.array(self.gt_backbone_identifiers)[sorted_idx], scores[sorted_idx]


def main():
    gen_file = 'result/train_generalize_generated_template_model_200000_wz.pt_bsz_10.csv'
    gen_data = pd.read_csv(gen_file)
    src2tgt = {}
    for i in range(gen_data.shape[0]):
        src = gen_data.iloc[i]['masked_rxn']
        if src not in src2tgt:
            src2tgt[src] = [gen_data.iloc[i]['generated_center']]
        else:
            src2tgt[src].append(gen_data.iloc[i]['generated_center'])
            src2tgt[src] = list(set(src2tgt[src]))

    with open('intermediate/gt_center_identifiers.pk', 'rb') as f:
        gt_center_identifiers = pickle.load(f)
    print('Number of unique backbone identifier:', len(set(gt_center_identifiers)))


    identifier = bbIdentifier(gt_center_identifiers)

    print('Start:')

    # novel_center = []
    # for src in tqdm(src2tgt):
    #     for gen_center in src2tgt[src]:
    #         cands, scores = identifier.get_similar(gen_center)
    #         if scores is not None and scores[0] < 0.85:
    #             novel_center.append(gen_center)

    def multi_get_similar(src, identifier, src2tgt):
        novel_center=[]
        for gen_center in src2tgt[src]:
            cands, scores = identifier.get_similar(gen_center)
            if scores is not None and scores[0] < 0.85:
                novel_center.append(gen_center)
        return novel_center


    with Pool() as pool:
        results = process_map(partial(multi_get_similar, identifier=identifier, src2tgt=src2tgt),
                              list(src2tgt.keys()), chunksize=10)
    novel_center = []
    for res in results:
        novel_center += res

    with open('novel_candidates.pk', 'wb') as f:
        pickle.dump(novel_center, f)

if __name__ == '__main__':
    main()