import json
import jsonlines
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import kendalltau, spearmanr
from scipy.stats import rankdata
from sklearn.metrics import ndcg_score


def get_iou(pred, gt):
    # print(pred, gt)
    intersection = 0
    for seg_a in gt:
        for seg_b in pred:
            intersection += max(0, min(seg_b[1], seg_a[1]) - max(seg_b[0], seg_a[0]))

    union = 0
    all_seg = []
    new_all_seg = []
    for seg in pred:
        all_seg.append(seg)
    for seg in gt:
        all_seg.append(seg)
    all_seg = sorted(all_seg)
    now_st = all_seg[0][0]
    now_ed = all_seg[0][1]
    for i in range(1, len(all_seg)):
        if all_seg[i][0] <= now_ed:
            now_ed = all_seg[i][1]
        else:
            new_all_seg.append([now_st, now_ed])
            now_st, now_ed = all_seg[i][0], all_seg[i][1]
    if now_st != now_ed:
        new_all_seg.append([now_st, now_ed])
    for seg in new_all_seg:
        # print(seg)
        union += seg[1] - seg[0]

    # print(intersection, union)

    iou = intersection / (union + 1e-8)
    return iou


def get_miou(predictions, groundtruths):
    ious = []
    for video in predictions.keys():
        pred = predictions[video]
        iou = get_iou(pred, groundtruths[video])
        ious.append(iou)
    miou = np.array(ious).mean()
    return miou, len(ious)


def evaluate_summary(
    result_path,
    val_summary_anno_path,
    eval_result_path,
    fps,
    sum_ratio,
    num_frames,
    mode,
):
    gt_segs = dict()
    gt_scores = dict()
    duration = dict()

    with open(val_summary_anno_path, "r") as f:
        for item in jsonlines.Reader(f):
            gt_segs[item["vname"]] = item["sum_segments"]
            gt_scores[item["vname"]] = item["saliency_scores"]
            duration[item["vname"]] = item["duration"]
    with open(result_path, "r") as f:
        result = json.load(f)
    pred_data = {}
    tau_list = []
    rho_list = []
    ndcg15_list = []
    ndcg1_list = []
    for idx in range(len(result)):
        score = torch.Tensor(result[idx]["score"])
        video = result[idx]["image_id"]
        gt_seg = gt_segs[video]
        gt_score = gt_scores[video]
        vduration = duration[video]
        # score need to romove padding here
        length = len(gt_score)
        if len(score) > length:
            ids = torch.arange(length).view(1, 1, -1).float()
            ids = F.interpolate(ids, size=num_frames, mode="nearest").long().flatten()
            score_id = [0] + [i for i in range(1, len(ids)) if ids[i] != ids[i - 1]]
            score = score[score_id]
        if mode == "2s":
            # predicted summary length
            sum_duration = vduration * sum_ratio
            score = np.array(score)
        elif mode == "frame":
            # predicted summary length
            sum_duration = len(score) / fps * sum_ratio  # (s)
            score = np.array(score)
            frame_2s = fps * 2
            last_seg = score[len(score) // frame_2s * frame_2s :]
            if len(last_seg) > 0:
                last_mean = np.mean(last_seg)
                last_len = frame_2s - len(last_seg)
                score = np.reshape(
                    np.concatenate(
                        [score, np.array([last_mean for i in range(last_len)])]
                    ),
                    (-1, frame_2s),
                )  # 2s for 16 frames
            else:
                score = np.reshape(score, (-1, frame_2s))
            score = np.mean(score, axis=1)
        else:
            raise NotImplementedError

        # unique & sort
        setsc = sorted(list(set(score)), reverse=True)

        # merge
        segments = []
        segscores = []
        seg_st = 0
        seg_ed = 0
        score_seg = {}
        for i in range(1, score.shape[0]):
            tscore = score[i]
            if tscore == score[i - 1]:
                seg_ed += 2
            else:
                score_seg.setdefault(str(score[i - 1]), [])
                score_seg[str(score[i - 1])].append([seg_st, seg_ed + 2])
                segments.append([seg_st, seg_ed + 2])
                segscores.append(score[i - 1])
                seg_st, seg_ed = seg_ed + 2, seg_ed + 2
        if seg_st != seg_ed + 2:
            score_seg.setdefault(str(score[len(score) - 1]), [])
            score_seg[str(score[len(score) - 1])].append([seg_st, seg_ed + 2])
            segments.append([seg_st, seg_ed + 2])
            segscores.append(score[len(score) - 1])

        sum_segments = []
        now_sum_duration = 0
        mxscore = setsc[0]
        for tscore in setsc:
            slen = 0
            for seg in score_seg[str(tscore)]:
                slen += seg[1] - seg[0]
            if slen + now_sum_duration <= sum_duration:
                sum_segments.extend(score_seg[str(tscore)])
                now_sum_duration += slen
            else:
                # print(id_duration)
                for seg in score_seg[str(tscore)]:
                    ratio = (seg[1] - seg[0]) / slen
                    # different segments have different lengths
                    id_duration = (sum_duration - now_sum_duration) * ratio
                    nearl, nearr = 0, 0
                    for sumseg in sum_segments:
                        if sumseg[0] == seg[1]:
                            nearr = 1
                        if sumseg[1] == seg[0]:
                            nearl = 1
                    nearcnt = nearl + nearr
                    if nearcnt == 2:
                        for sumseg in sum_segments:
                            if sumseg[0] == seg[1]:
                                sumseg[0] = sumseg[0] - id_duration / 2
                            if sumseg[1] == seg[0]:
                                sumseg[1] = sumseg[1] + id_duration / 2
                    elif nearcnt == 1:
                        for sumseg in sum_segments:
                            if sumseg[0] == seg[1]:
                                sumseg[0] = sumseg[0] - id_duration
                            if sumseg[1] == seg[0]:
                                sumseg[1] = sumseg[1] + id_duration
                    else:
                        if (
                            id_duration > 2 or tscore == mxscore
                        ):  # summary threshold per clip / maxscore
                            newseg = [
                                (seg[0] + seg[1]) / 2 - id_duration / 2,
                                (seg[0] + seg[1]) / 2 + id_duration / 2,
                            ]
                            sum_segments.append(newseg)
                break
        sum_segments = sorted(sum_segments)

        # merge segments
        new_segments, all_segments = [sum_segments[0]], sum_segments[1:]
        for seg in all_segments:
            if seg[0] == new_segments[-1][1]:
                new_segments[-1][1] = seg[1]
            else:
                new_segments.append(seg)
        pred_data[video] = new_segments
        result[idx]["summary segments"] = new_segments
        result[idx]["score"] = score.tolist()

        # tau and rho are not affected by the absolute value of gt_score, no need to divede 4.0
        # Tau
        gt_score = np.array(gt_score[: score.shape[0]])
        mscorerank, gtscorerank = rankdata(-score), rankdata(-gt_score)
        tau = kendalltau(mscorerank, gtscorerank)
        if np.isnan(tau[0]):
            pass
        else:
            tau_list.append(tau[0])

        # Rho
        rho = spearmanr(score, gt_score)
        if np.isnan(rho[0]):
            pass
        else:
            rho_list.append(rho[0])

        # and so is ndcg
        score = (score - np.min(score)) / (np.max(score) - np.min(score) + 1e-8)
        gt_score = (gt_score - np.min(gt_score)) / (
            np.max(gt_score) - np.min(gt_score) + 1e-8
        )
        ndcg15_list.append(
            ndcg_score([gt_score], [score], k=math.ceil(score.shape[0] * 0.15))
        )
        ndcg1_list.append(
            ndcg_score([gt_score], [score], k=math.ceil(score.shape[0] * 1))
        )

    # mIoU, rho, tau
    miou, _ = get_miou(pred_data, gt_segs)
    tau = np.mean(tau_list)
    rho = np.mean(rho_list)
    ndcg15 = np.mean(ndcg15_list)
    ndcg1 = np.mean(ndcg1_list)

    # F-score
    all_fscore = []
    for video in pred_data.keys():
        machine_seg = pred_data[video]
        gt_seg = gt_segs[video]
        vd_duration = math.ceil(duration[video] * fps)

        machine_summary = []
        gt_summary = []
        for i in range(vd_duration):
            flag_gt, flag_ma = 0, 0
            for seg in gt_seg:
                if i / fps >= seg[0] and i / fps <= seg[1]:
                    gt_summary.append(1)
                    flag_gt = 1
                    break
            for seg in machine_seg:
                if i / fps >= seg[0] and i / fps <= seg[1]:
                    machine_summary.append(1)
                    flag_ma = 1
                    break
            if flag_gt == 0:
                gt_summary.append(0)
            if flag_ma == 0:
                machine_summary.append(0)

        machine_summary = np.array(machine_summary)
        gt_summary = np.array(gt_summary)

        overlap_duration = (machine_summary * gt_summary).sum()
        precision = overlap_duration / (machine_summary.sum() + 1e-8)
        recall = overlap_duration / (gt_summary.sum() + 1e-8)
        if precision == 0 and recall == 0:
            f_score = 0.0
        else:
            f_score = (2 * precision * recall) / (precision + recall)
        all_fscore.append(f_score)
    f_score = np.array(all_fscore).mean()

    with open(eval_result_path, "r") as f:
        eval_result = json.load(f)
    eval_result["F-score"] = f_score
    eval_result["mIoU"] = miou
    eval_result["Tau"] = tau
    eval_result["Rho"] = rho
    eval_result["NDCG@.15"] = ndcg15
    eval_result["NDCG@.1"] = ndcg1
    with open(eval_result_path, "w") as f:
        json.dump(eval_result, f)

    with open(result_path, "w") as f:
        json.dump(result, f)
