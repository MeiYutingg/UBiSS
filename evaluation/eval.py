import os, sys
import jsonlines
import json
from pytorch_lightning.utilities.distributed import rank_zero_only
from .evalcap.utils_caption_evaluate import evaluate_on_coco_caption
from .evalsum.evaluate_summary import evaluate_summary

@rank_zero_only
def evaluate(eval_cfg, mode):
    fps = eval_cfg.fps
    sum_ratio = eval_cfg.sum_ratio
    num_frames = eval_cfg.num_frames
    # calculate validation split metrics
    val_caption_coco_file_path = eval_cfg.val_caption_coco_file_path
    val_summary_anno_path = eval_cfg.val_summary_anno_path
    for file in os.listdir(eval_cfg.val_result_path):
        file_path = os.path.join(eval_cfg.val_result_path, file)
        eval_result_path = os.path.join(eval_cfg.eval_result_path, file)
        collected_result = list()
        video_keys = set()
        cnt = 0
        with jsonlines.open(file_path, "r") as reader:
            try:
                for line in reader:
                    tmp = line
                    if tmp['image_id'] not in video_keys:
                        tmp['id'] = cnt
                        collected_result.append(tmp)
                        cnt += 1
                    video_keys.add(tmp['image_id'])
            except:
                pass
        if len(collected_result) > 0:
            with open(file_path, "w") as f:
                json.dump(collected_result, f)
        evaluate_on_coco_caption(file_path, val_caption_coco_file_path, eval_result_path)
        evaluate_summary(file_path, val_summary_anno_path, eval_result_path, fps, sum_ratio, num_frames, mode)
        if eval_cfg.save_val_result == False:
            os.remove(file_path)
        if eval_cfg.save_eval_result == False:
            os.remove(eval_result_path)
            
@rank_zero_only
def evaluate_inference(eval_cfg, mode):
    fps = eval_cfg.fps
    sum_ratio = eval_cfg.sum_ratio
    num_frames = eval_cfg.num_frames
    # calculate test split metrics
    test_caption_coco_file_path = eval_cfg.test_caption_coco_file_path
    test_summary_anno_path = eval_cfg.test_summary_anno_path
    file_path = os.path.join(eval_cfg.test_result_path, 'test.json')
    test_eval_result_path = os.path.join(eval_cfg.test_result_path, 'test_metrics.json')
    collected_result = list()
    video_keys = set()
    cnt = 0
    with jsonlines.open(file_path, "r") as reader:
        try:
            for line in reader:
                tmp = line
                if tmp['image_id'] not in video_keys:
                    tmp['id'] = cnt
                    collected_result.append(tmp)
                    cnt += 1
                video_keys.add(tmp['image_id'])
        except:
            pass
    if len(collected_result) > 0:
        with open(file_path, "w") as f:
            json.dump(collected_result, f)
    evaluate_on_coco_caption(file_path, test_caption_coco_file_path, test_eval_result_path)
    evaluate_summary(file_path, test_summary_anno_path, test_eval_result_path, fps, sum_ratio, num_frames, mode)
