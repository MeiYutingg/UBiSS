import os

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl

import h5py
import numpy as np
import json
import jsonlines
from random import shuffle
from .utils.comm_utils import *
from .utils.caption_tensorizer import build_tensorizer
from .utils.tsv_file import TSVFile


class QVHighlights(Dataset):

    def __init__(
        self,
        num_frames,
        visual_data_path,
        caption_tsv_path,
        sumseg_path,
        time_token_bins,
        video_keys,
        tokenizer,
        tensorizer,
        is_train,
    ):
        self.num_frames = num_frames
        self.caption_tsv = TSVFile(caption_tsv_path)
        self.video_keys = video_keys
        self.tokenizer = tokenizer
        self.tensorizer = tensorizer
        self.is_train = is_train
        self.visual_data_file = h5py.File(visual_data_path, "r")
        self.visual_data = []
        self.key2index = self.prepare_caption_key_to_index()
        self.time_info = dict()
        self.video_duration = dict()
        self.time_token_bins = time_token_bins

        # load video features
        for video in video_keys:
            self.visual_data.append(self.visual_data_file[video])
        with open(sumseg_path, "r", encoding="utf-8") as f:
            for video_infos in jsonlines.Reader(f):
                self.time_info[video_infos["vname"]] = video_infos["sum_segments"]
                self.video_duration[video_infos["vname"]] = video_infos["duration"]

    def prepare_caption_key_to_index(self):
        tsv = self.caption_tsv
        return {tsv.get_key(i): i for i in range(tsv.num_rows())}

    def __len__(self):
        self.len = len(self.video_keys)
        return self.len

    def __getitem__(self, index):
        if self.is_train == True:
            video_name = str(self.visual_data[index]["video_name"][...])
            vis_feats = torch.Tensor(self.visual_data[index]["features"][...])
            gt_score = torch.Tensor(self.visual_data[index]["gtscore"])
            caption = json.loads(self.caption_tsv[self.key2index[video_name]][1])[0][
                "caption"
            ]

            # prepare visual feature indexes, padding to num_frames
            length = len(vis_feats)
            if length >= self.num_frames:
                ids = torch.randperm(length)[: self.num_frames]
                ids = torch.sort(ids)[0]
            else:
                ids = torch.arange(length).view(1, 1, -1).float()
                ids = (
                    F.interpolate(ids, size=self.num_frames, mode="nearest")
                    .long()
                    .flatten()
                )

            # prepare caption for bert
            input_ids, attention_mask, segment_ids, masked_pos, mlm_targets = (
                self.tensorizer.tensorize_example_e2e(caption)
            )
            return (
                gt_score[ids],
                vis_feats[ids],
                input_ids,
                attention_mask,
                segment_ids,
                masked_pos,
                mlm_targets,
            )
        else:
            video = self.visual_data[index]
            video_name = str(video["video_name"][...])
            vis_feats = torch.Tensor(video["features"][...])
            gt_score = torch.Tensor(self.visual_data[index]["gtscore"])
            caption = json.loads(self.caption_tsv[self.key2index[video_name]][1])[0][
                "caption"
            ]

            input_ids, attention_mask, segment_ids, masked_pos = (
                self.tensorizer.tensorize_example_e2e(caption)
            )

            length = len(vis_feats)
            if length >= self.num_frames:
                ids = torch.randperm(length)[: self.num_frames]
                ids = torch.sort(ids)[0]
            else:
                ids = torch.arange(length).view(1, 1, -1).float()
                ids = (
                    F.interpolate(ids, size=self.num_frames, mode="nearest")
                    .long()
                    .flatten()
                )
            return (
                video_name,
                gt_score[ids],
                vis_feats[ids],
                input_ids,
                attention_mask,
                segment_ids,
                masked_pos,
            )


class QVHDataModule(pl.LightningDataModule):

    def __init__(self, bert_cfg, data_cfg, hpms, tokenizer, mode):
        super().__init__()
        self.tokenizer = tokenizer
        self.bert_cfg = bert_cfg
        self.data_cfg = data_cfg
        self.hpms = hpms
        self.mode = mode

        # split information
        with open(self.data_cfg.split_file, "r") as f:
            info = json.load(f)
            self.train_video = info["train_keys"]
            self.val_video = info["val_keys"]
            self.test_video = info["test_keys"]

        # visual CNN feature settings
        if self.mode == "2s":
            self.visual_data_path = data_cfg.paths.mean2s
            self.num_frames = data_cfg.num_frames.mean2s
        elif self.mode == "frame":
            self.visual_data_path = data_cfg.paths.frame
            self.num_frames = data_cfg.num_frames.frame

        # load caption tsv file
        self.train_cap_tsv_path = os.path.join(
            data_cfg.paths.caption, "train.caption.tsv"
        )
        self.val_cap_tsv_path = os.path.join(data_cfg.paths.caption, "val.caption.tsv")
        self.test_cap_tsv_path = os.path.join(
            data_cfg.paths.caption, "test.caption.tsv"
        )

        # load summary segment file
        self.train_sumseg_path = os.path.join(data_cfg.paths.summary, "train.jsonl")
        self.val_sumseg_path = os.path.join(data_cfg.paths.summary, "val.jsonl")
        self.test_sumseg_path = os.path.join(data_cfg.paths.summary, "test.jsonl")

    def setup(self, stage: str):
        # train/val datasets
        if stage == "fit":
            self.train_tensorizer = build_tensorizer(
                self.bert_cfg, self.tokenizer, is_train=True
            )
            self.val_tensorizer = build_tensorizer(
                self.bert_cfg, self.tokenizer, is_train=False
            )
        # test dataset
        if stage == "test":
            self.test_tensorizer = build_tensorizer(
                self.bert_cfg, self.tokenizer, is_train=False
            )

    def train_dataloader(self):
        dataset = QVHighlights(
            num_frames=self.num_frames,
            visual_data_path=self.visual_data_path,
            caption_tsv_path=self.train_cap_tsv_path,
            sumseg_path=self.train_sumseg_path,
            time_token_bins=self.data_cfg.time_token_bins,
            video_keys=self.train_video,
            tokenizer=self.tokenizer,
            tensorizer=self.train_tensorizer,
            is_train=True,
        )
        dataloader = DataLoader(
            dataset,
            num_workers=self.data_cfg.num_workers,
            batch_size=self.hpms.batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self.dataloader_len = len(dataloader)
        print("Number of training videos: {}".format(len(dataset)))
        return dataloader

    def val_dataloader(self):
        dataset = QVHighlights(
            num_frames=self.num_frames,
            visual_data_path=self.visual_data_path,
            caption_tsv_path=self.val_cap_tsv_path,
            sumseg_path=self.val_sumseg_path,
            time_token_bins=self.data_cfg.time_token_bins,
            video_keys=self.val_video,
            tokenizer=self.tokenizer,
            tensorizer=self.val_tensorizer,
            is_train=False,
        )
        dataloader = DataLoader(
            dataset,
            num_workers=self.data_cfg.num_workers,
            batch_size=1,
            shuffle=False,
            collate_fn=test_collate,
            pin_memory=True,
        )
        self.dataloader_len = len(dataloader)
        print("Number of validation videos: {}".format(len(dataset)))
        return dataloader

    def test_dataloader(self):
        dataset = QVHighlights(
            num_frames=self.num_frames,
            visual_data_path=self.visual_data_path,
            caption_tsv_path=self.test_cap_tsv_path,
            sumseg_path=self.test_sumseg_path,
            time_token_bins=self.data_cfg.time_token_bins,
            video_keys=self.test_video,
            tokenizer=self.tokenizer,
            tensorizer=self.test_tensorizer,
            is_train=False,
        )
        dataloader = DataLoader(
            dataset,
            num_workers=self.data_cfg.num_workers,
            batch_size=1,
            shuffle=False,
            collate_fn=test_collate,
            pin_memory=True,
        )
        print("Number of test videos: {}".format(len(dataset)))
        return dataloader
