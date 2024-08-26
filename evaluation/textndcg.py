import os
import sys
import json
from tqdm import tqdm
import torch
import numpy as np
import math
import cv2

sys.path.append("path of CLIP-main")
import clip
from PIL import Image
from sklearn.metrics import ndcg_score


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def cal_clip_sim(img, cap):
    image = preprocess(img).unsqueeze(0).to(device)
    text = clip.tokenize([cap]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = 100 * image_features @ text_features.T

    return similarity[0][0].item()


def get_gt_saliency():
    # put the directory of summary annotation including saliency here
    gt_saliency_dir = "data/summary_annotation/test.jsonl"
    gt_saliency = dict()
    with open(gt_saliency_dir, "r") as file:
        for line in file:
            video = json.loads(line)
            gt_saliency[video["vname"]] = video["saliency_scores"]
    return gt_saliency


for file in os.listdir("directory containing all files to be evaluated"):
    with open(
        os.path.join("directory containing all files to be evaluated", file), "r"
    ) as f:
        sum_info = json.load(f)

    video_base_path = "directory containing videos"
    gt_saliency = get_gt_saliency()
    # print(gt_saliency)

    all_ndcg_15 = list()
    all_ndcg_all = list()
    with tqdm(total=len(sum_info)) as pbar:
        for video_info in sum_info:
            image_id = video_info["image_id"]
            caption = video_info["caption"]
            gt_score = gt_saliency[image_id]
            sim = list()

            video_path = os.path.join(video_base_path, image_id)
            video = cv2.VideoCapture(video_path)

            while video.isOpened():
                ret, frame = video.read()
                if frame is None:
                    break
                if ret == True:
                    img_Image = Image.fromarray(np.uint8(frame))
                    clip_sim = cal_clip_sim(img_Image, caption)
                    sim.append(clip_sim)
            video.release()

            # get maximum similarity for every 2s clip (for 8fps videos)
            max_sim = list()
            for i in range(0, len(sim), 16):
                max_sim.append(max(sim[i : i + 15]))
            mxlen = min(len(max_sim), len(gt_score))
            max_sim = max_sim[:mxlen]
            gt_score = gt_score[:mxlen]

            # normalization
            score = np.array(max_sim)
            score = (score - np.min(score)) / (np.max(score) - np.min(score) + 1e-8)
            gt_score = np.array(gt_score) / 4.0
            gt_score = (gt_score - np.min(gt_score)) / (
                np.max(gt_score) - np.min(gt_score) + 1e-8
            )

            save_dict = dict()
            save_dict["video_name"] = image_id
            save_dict["text similarity"] = sim
            # save_dict["gt saliency"] = gt_saliency[image_id]
            save_dict["ndcg@15"] = ndcg_score(
                [gt_score], [score], k=math.ceil(score.shape[0] * 0.15)
            )
            save_dict["ndcg@all"] = ndcg_score(
                [gt_score], [score], k=math.ceil(score.shape[0] * 1)
            )

            all_ndcg_15.append(save_dict["ndcg@15"])
            all_ndcg_all.append(save_dict["ndcg@all"])
            with open("path to save result" + file, "r") as f:
                result = json.load(f)
            result.append(save_dict)
            with open("path to save result" + file, "w") as f:
                json.dump(result, f)
            pbar.update(1)

    with open("path to save result" + file, "r") as f:
        result = json.load(f)
    result.append(
        {"mean ndcg@15": np.mean(all_ndcg_15), "mean ndcg@all": np.mean(all_ndcg_all)}
    )
    print(
        {"mean ndcg@15": np.mean(all_ndcg_15), "mean ndcg@all": np.mean(all_ndcg_all)}
    )
    with open("path to save result" + file, "w") as f:
        json.dump(result, f)
