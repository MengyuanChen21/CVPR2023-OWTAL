import numpy as np
import pandas as pd
import torch
from scipy.signal import savgol_filter

import options
import utils.wsad_utils as utils
import math
from edl_loss import exp_evidence

args = options.parser.parse_args()

def filter_segments(segment_predict, vn):
    ambilist = args.path_dataset + '/Thumos14reduced-Annotations/Ambiguous_test.txt'
    try:
        ambilist = list(open(ambilist, "r"))
        ambilist = [a.strip("\n").split(" ") for a in ambilist]
    except:
        ambilist = []
    ind = np.zeros(np.shape(segment_predict)[0])
    for i in range(np.shape(segment_predict)[0]):
        for a in ambilist:
            if a[0] == vn:
                gt = range(
                    int(round(float(a[2]) * 25 / 16)), int(round(float(a[3]) * 25 / 16))
                )
                pd = range(int(segment_predict[i][0]), int(segment_predict[i][1]))
                IoU = float(len(set(gt).intersection(set(pd)))) / float(
                    len(set(gt).union(set(pd)))
                )
                if IoU > 0:
                    ind[i] = 1
    s = [
        segment_predict[i, :]
        for i in range(np.shape(segment_predict)[0])
        if ind[i] == 0
    ]
    return np.array(s)


def smooth(v, order=2, lens=200):
    l = min(lens, len(v))
    l = l - (1 - l % 2)
    if len(v) <= order:
        return v
    return savgol_filter(v, l, order)


def get_topk_mean(x, k, axis=0):
    return np.mean(np.sort(x, axis=axis)[-int(k):, :], axis=0)


def get_cls_score(element_cls, rat=20):
    topk_val, _ = torch.topk(element_cls, k=max(1, int(element_cls.shape[-2] // rat)), dim=-2)
    instance_logits = torch.mean(topk_val, dim=-2)
    pred_vid_score = torch.softmax(instance_logits, dim=-1)[..., :-1].squeeze().data.cpu().numpy()
    return pred_vid_score


def __vector_minmax_norm(vector, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        max_val = np.max(vector)
        min_val = np.min(vector)

    delta = max_val - min_val
    # delta[delta <= 0] = 1
    ret = (vector - min_val) / delta

    return ret


def _multiply(x, atn, dim=-1, include_min=False):
    if include_min:
        _min = x.min(dim=dim, keepdim=True)[0]
    else:
        _min = 0
    return atn * (x - _min) + _min


def sigmoid(x, thres_uct_list, max_score_class):
    # x = (x - 0.25) / (0.75 - 0.25)
    x = x - thres_uct_list[max_score_class] + thres_uct_list.mean()
    return 1 / (1 + torch.exp(-16 * (x - 0.45)))


@torch.no_grad()
def multiple_threshold_hamnet(vid_name, data_dict, labels, args, thres):
    labels = torch.tensor(labels)
    open_labels = torch.zeros(args.n_known_class + 1)
    open_labels[:args.n_known_class] = labels[:args.n_known_class]
    if labels[args.n_known_class:].sum() > 0:
        open_labels[-1] = 1

    cas = data_dict['cas']
    atn = data_dict['attn']
    video_uct = data_dict['uct'][0].cpu().item()
    # video_uct = obtain_uct(args, data_dict)

    element_logits = cas * atn

    pred_vid_score = get_cls_score(element_logits, rat=10)

    pred_vid_score = np.concatenate((pred_vid_score, np.array([video_uct])))
    cas_supp = element_logits[..., :-1]

    known_flag = True
    if video_uct <= thres:  # uct小于阈值，只有已知类
        unknown_flag = False
        pred = np.where(pred_vid_score[:-1] >= 0.2)[0]
        if len(pred) == 0:
            pred = np.array([np.argmax(pred_vid_score[:-1])])
    else:  # uct大于阈值，有未知类
        unknown_flag = True
        pred = np.where(pred_vid_score[:-1] >= 0.5)[0]
        if len(pred) == 0:
            known_flag = False
        pred = np.concatenate([pred, np.array([args.n_known_class])])
# -----------------------------------------------------------------------

    num_segments = cas.shape[1]

    cas_pred_atn = atn[0].cpu().numpy()[:, [0]]
    cas_pred_atn = np.reshape(cas_pred_atn, (num_segments, -1, 1))
    if known_flag and not unknown_flag:
        cas_pred = cas_supp[0].cpu().numpy()[:, pred]
        cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))
    elif not known_flag and unknown_flag:
        cas_pred = cas_pred_atn
    elif known_flag and unknown_flag:
        cas_pred = cas_supp[0].cpu().numpy()[:, pred[:-1]]
        cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))
        cas_pred = np.hstack((cas_pred, cas_pred_atn))
    else:
        raise "Error"

    # NOTE: threshold
    act_thresh = np.linspace(0.1, 0.9, 10)

    proposal_dict = {}

    for i in range(len(act_thresh)):
        cas_temp = cas_pred.copy()
        cas_temp_atn = cas_pred_atn.copy()
        seg_list = []
        for c in range(len(pred)):
            pos = np.where(cas_temp_atn[:, 0, 0] > act_thresh[i])
            seg_list.append(pos)

        proposals = utils.get_proposal_oic_2(seg_list,
                                             cas_temp,
                                             pred_vid_score,
                                             pred,
                                             gamma=args.gamma_oic)

        for j in range(len(proposals)):
            class_id = proposals[j][0][0]

            if class_id not in proposal_dict.keys():
                proposal_dict[class_id] = []

            proposal_dict[class_id] += proposals[j]

    final_proposals = []
    for class_id in proposal_dict.keys():
        final_proposals.append(
            utils.soft_nms(proposal_dict[class_id], 0.7, sigma=0.3))

    # [c_pred[i], c_score, t_start, t_end]
    segment_predict = []
    for i in range(len(final_proposals)):
        for j in range(len(final_proposals[i])):
            [c_pred, c_score, t_start, t_end] = final_proposals[i][j]
            segment_predict.append([t_start, t_end, c_score, c_pred])

    segment_predict = np.array(segment_predict)
    segment_predict = filter_segments(segment_predict, vid_name.decode())

    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    uct_lst, act_lst = [], []
    for i in range(np.shape(segment_predict)[0]):
        video_lst.append(vid_name.decode())
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )

    if not args.topk_test:
        args.n_pred_list = ['unlimit']
        return [prediction]
    else:
        args.n_pred_list = [5, 10, 20, 50, 100, 'unlimit']
        if prediction.empty:
            return [prediction] * 6

        topk_prediction_list = []
        for n_pred in args.n_pred_list[:-1]:
            if known_flag and not unknown_flag:  # 只有已知类
                n_known_pred = int(n_pred / pred.shape[0])
                n_unknown_pred = 0
            elif not known_flag and unknown_flag:  # 只有未知类
                n_known_pred = 0
                n_unknown_pred = n_pred
            elif known_flag and unknown_flag:  # 同时存在
                n_known_pred = int(n_pred * 0.5 / pred.shape[0])
                n_unknown_pred = n_pred - n_known_pred
            else:
                raise "Error"

            all_class_topk_proposal = []
            prediction_by_label = prediction.groupby("label")
            for i, cidx in enumerate(pred):
                one_class_prediction = _get_predictions_with_label(prediction_by_label, cidx)
                sort_idx = one_class_prediction["score"].values.argsort()[::-1]  # idx from high to low
                one_class_prediction = one_class_prediction.loc[sort_idx].reset_index(drop=True)  # value from high to low
                if cidx < args.n_known_class:
                    k = n_known_pred
                elif cidx == args.n_known_class:
                    k = n_unknown_pred
                else:
                    raise ValueError
                one_class_topk_proposal = one_class_prediction[: k]
                all_class_topk_proposal.append(one_class_topk_proposal)
            topk_prediction = pd.concat(all_class_topk_proposal).reset_index(drop=True)

            topk_prediction_list.append(topk_prediction)
        topk_prediction_list.append(prediction)

    return topk_prediction_list


def _get_predictions_with_label(prediction_by_label, cidx):
    """Get all predicitons of the given label. Return empty DataFrame if there
    is no predcitions with the given label.
    """
    return prediction_by_label.get_group(cidx).reset_index(drop=True)
