import numpy as np
from tqdm import tqdm
import torch

def validate(val_iter, model, generalize=False, verbose=False):
    model.eval()
    predict, truth = [], []
    predict_rc, truth_rc = [], []

    progress_bar = val_iter
    if verbose:
        progress_bar = tqdm(val_iter)
    for batch in progress_bar:
        inputs = (batch.src, batch.src_am, batch.src_seg, batch.tgt)
        gtruth_token = batch.tgt[1:]
        gtruth_react_class = batch.rt_label

        with torch.no_grad():
            scores, rt_scores, kld_loss = model(inputs, fixed_z=True, generalize=generalize)
            _, pred = scores.topk(1, dim=-1)

        _, pred = scores.topk(1, dim=-1)
        pred = pred.squeeze(2)

        if generalize:
            _, pred_rc = rt_scores.topk(1, dim=-1)
            pred_rc = pred_rc.squeeze(1)
            predict_rc += list(pred_rc.cpu().numpy())
            truth_rc += list(gtruth_react_class.cpu().numpy())

        for i in range(pred.shape[1]):
            gt = gtruth_token[:, i]
            p = pred[:, i]

            for j in range(len(gt)):
                if gt[j] != 3: # <pad> index of tgt
                    predict.append(p[j].item())
                    truth.append(gt[j].item())
    return np.mean(np.array(predict) == np.array(truth)), np.mean(np.array(predict_rc) == np.array(truth_rc))
