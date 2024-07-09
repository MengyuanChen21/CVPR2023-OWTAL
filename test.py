import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from prettytable import PrettyTable
from torch.autograd import Variable

import model
import options
import proposal_methods as PM
import wsad_dataset
from eval.eval_detection import ANETdetection

torch.set_default_dtype(torch.float32)


def _get_predictions_with_label(prediction_by_label, cidx):
    """Get all predicitons of the given label. Return empty DataFrame if there
    is no predcitions with the given label.
    """
    try:
        return prediction_by_label.get_group(cidx).reset_index(drop=True)
    except:
        print("Warning: No predictions of label '%s' were provdied." % cidx)
        return pd.DataFrame()

def get_video_detections(args, tmp):
    proposal_list = []
    for i in range(tmp.shape[0]):
        tmp_proposal = {}
        tmp_proposal['label'] = args.classlist[int(tmp.loc[i]['label'])]
        tmp_proposal['score'] = float(tmp.loc[i]['score'])
        tmp_proposal['segment'] = [float(tmp.loc[i]['t-start'] / 1.5626), float(tmp.loc[i]['t-end'] / 1.5626)]
        tmp_proposal['uncertainty'] = float(tmp.loc[i]['uct'])
        tmp_proposal['actionness'] = float(tmp.loc[i]['act'])
        proposal_list.append(tmp_proposal)
    return proposal_list


@torch.no_grad()
def test(itr, dataset, args, model, device):
    model.eval()
    done = False
    if args.topk_test:
        topk_proposals_list = [[], [], [], [], [], []]
    else:
        topk_proposals_list = [[], ]
    results = defaultdict(dict)

    train_uct_list = []
    train_ori_uct_list = []
    while not done:
        features, labels, vn, done = dataset.load_data_for_threshold()
        seq_len = [features.shape[0]]
        if seq_len == 0:
            continue
        features = torch.from_numpy(features).float().to(device).unsqueeze(0)
        with torch.no_grad():
            outputs = model(Variable(features), is_training=True, seq_len=seq_len, itr=itr, opt=args, labels=None)
            this_uct = outputs['uct'][0].cpu().item()
            this_ori_uct = outputs['ori_uct'][0].cpu().item()
            train_uct_list.append(this_uct)
            train_ori_uct_list.append(this_ori_uct)
    train_uct_list = np.sort(np.array(train_uct_list), axis=0)
    train_ori_uct_list = np.sort(np.array(train_ori_uct_list), axis=0)
    thres = train_uct_list[int(0.95 * len(train_uct_list))]
    print(f"We select {thres:.4f} as the uncertainty threshold.")

    mu = train_ori_uct_list[int(0.5 * len(train_ori_uct_list))]
    mu_path = './temp/' + args.group_name + '/' + args.model_name
    if not os.path.exists(mu_path):
        os.makedirs(mu_path)
    np.save(os.path.join(mu_path, 'mu.npy'), mu)
    print(f"We select {mu:.4f} as the mean of the gaussian function.")

    n_correct = 0
    n_test_vid = 0
    test_uct_list = []
    done = False

    result_dict = {}
    while not done:
        n_test_vid += 1
        features, labels, vn, done = dataset.load_data(is_training=False)
        seq_len = [features.shape[0]]
        if seq_len == 0:
            continue
        features = torch.from_numpy(features).float().to(device).unsqueeze(0)
        with torch.no_grad():
            outputs = model(Variable(features), is_training=False, seq_len=seq_len, itr=itr, opt=args, labels=None)
            results[vn] = {'cas': outputs['cas'], 'attn': outputs['attn']}
            video_uct = outputs['uct'][0].cpu().item()
            prediction_list = getattr(PM, args.proposal_method)(vn, outputs, labels, args, thres)

            if video_uct <= thres and labels[:args.n_known_class].sum() > 0:
                n_correct += 1
            elif video_uct > thres and labels[args.n_known_class:].sum() > 0:
                n_correct += 1
            else:
                n_correct += 0

        test_uct_list.append(video_uct)
        for idx, prediction in enumerate(prediction_list):
            topk_proposals_list[idx].append(prediction)

    if not os.path.exists('temp'):
        os.mkdir('temp')
    np.save('temp/{}.npy'.format(args.model_name), results)

    if 'Thumos14' in args.dataset_name:
        iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args, verbose=True)
    else:
        iou = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args, subset='validation', verbose=True)

    # video-id, t-start, t-end, label, score
    table = PrettyTable(['k', 'split', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, '0.1-0.5', '0.3-0.7', '0.1-0.7'])

    log_all_mAP = []
    for idx, topk_proposals in enumerate(topk_proposals_list):
        proposals = pd.concat(topk_proposals).reset_index(drop=True)

        dmap_detect.prediction = proposals
        known_mAP, unknown_mAP = dmap_detect.evaluate()

        known_mAP *= 100
        unknown_mAP *= 100
        all_mAP = known_mAP * 0.5 + unknown_mAP * 0.5
        mAP_list = [known_mAP, unknown_mAP, all_mAP]
        for j, split in enumerate(['known', 'unknown', 'all']):
            # if j == 2:
            table.add_row([args.n_pred_list[idx], split] + list(np.around(mAP_list[j][:7], decimals=2)) +
                          list(np.around([mAP_list[j][:5].mean(), mAP_list[j][2:7].mean(), mAP_list[j][:7].mean()], decimals=2))
                          )

        if args.n_pred_list[idx] == 'unlimit':
            log_known_mAP, log_unknown_mAP, log_all_mAP = mAP_list

    np.set_printoptions(precision=2, suppress=True)
    print(table)
    uct_rank_acc = n_correct / n_test_vid * 100
    print(f'Accuracy of binary classification: {uct_rank_acc:.4f}%')

    return iou, log_all_mAP, uct_rank_acc


if __name__ == '__main__':
    args = options.parser.parse_args()
    device = torch.device("cuda")
    dataset = getattr(wsad_dataset, args.dataset)(args)

    model = getattr(model, args.use_model)(dataset.feature_size, dataset.num_class, opt=args).to(device)
    model.load_state_dict(torch.load(args.test_ckpt))
    iou, dmap, _ = test(-1, dataset, args, model, device)
