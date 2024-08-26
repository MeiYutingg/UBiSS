import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import jsonlines

from .encoder import SumCapEncoder
from .load_bert import get_bert_model
from .loss.neuralNDCG import neuralNDCG


class SumCapModel(pl.LightningModule):
    def __init__(
        self,
        enc_cfg,
        bert_cfg,
        loss_cfg,
        eval_cfg,
        hpms,
        mode,
        pretrained_bert_path=None,
        learning_rate=0,
    ):
        super().__init__()
        self.enc_cfg = enc_cfg
        self.bert_cfg = bert_cfg
        self.encoder = SumCapEncoder(
            d_inp=enc_cfg.d_inp,
            n_layers=enc_cfg.n_layers,
            n_head=enc_cfg.n_head,
            d_model=enc_cfg.d_model,
            d_inner=enc_cfg.d_inner,
            use_drop_out=enc_cfg.use_drop_out,
            use_layer_norm=enc_cfg.use_layer_norm,
        )  # (batch, frame, d_model)
        self.bert_cfg.vocab_size = self.bert_cfg.vocab_size + hpms.time_token_bins + 2
        self.bert, _, self.tokenizer = get_bert_model(bert_cfg)

        self.loss_cfg = loss_cfg
        if self.loss_cfg.sum_loss == "MSE":
            self.sum_loss_func = nn.MSELoss(reduction="mean")
        elif self.loss_cfg.sum_loss == "neuralNDCG":
            self.sum_loss_func = neuralNDCG
        self.eval_cfg = eval_cfg
        self.hpms = hpms
        self.mode = mode
        self.epoch_num = 0
        if self.hpms.use_lr_finder == True:
            self.learning_rate = learning_rate

    def forward(self, *args, **kwargs):
        vis_feats, scores = self.encoder(kwargs["img_feats"])
        kwargs["img_feats"] = vis_feats
        bert_outputs = self.bert(*args, **kwargs)
        return scores, bert_outputs

    def training_step(self, batch, batch_idx):
        gt_scores = batch[0]
        inputs = {
            "img_feats": batch[1],
            "input_ids": batch[2],
            "attention_mask": batch[3],
            "token_type_ids": batch[4],
            "masked_pos": batch[5],
            "masked_ids": batch[6],
        }
        scores, bert_output = self(**inputs)
        bert_loss, logits = bert_output
        sum_loss = self.sum_loss_func(scores, gt_scores)

        # ablation

        if self.hpms.use_bert_loss == True and self.hpms.use_sum_loss == True:
            if self.current_epoch < self.hpms.language_warm_up_epoch:
                cap_loss_ratio = self.hpms.cap_loss_ratio
                loss = cap_loss_ratio * bert_loss
            else:
                sum_loss_ratio = self.hpms.sum_loss_ratio
                cap_loss_ratio = self.hpms.cap_loss_ratio
                loss = sum_loss_ratio * sum_loss + cap_loss_ratio * bert_loss
        elif self.hpms.use_bert_loss == True and self.hpms.use_sum_loss == False:
            loss = bert_loss
        elif self.hpms.use_bert_loss == False and self.hpms.use_sum_loss == True:
            loss = sum_loss
        else:
            raise NotImplementedError
        self.log("train/sum loss", sum_loss)
        self.log("train/bert loss", bert_loss)
        self.log("train/loss", loss)

        return loss

    def on_validation_start(self):
        self.val_result_path = os.path.join(
            self.eval_cfg.val_result_path, str(self.epoch_num) + ".json"
        )

    def validation_step(self, batch, batch_idx):
        (
            cls_token_id,
            sep_token_id,
            pad_token_id,
            mask_token_id,
            period_token_id,
        ) = self.tokenizer.convert_tokens_to_ids(
            [
                self.tokenizer.cls_token,
                self.tokenizer.sep_token,
                self.tokenizer.pad_token,
                self.tokenizer.mask_token,
                ".",
            ]
        )
        video_name, gt_score = batch[0], batch[1]

        inputs = {
            "is_decode": True,
            "img_feats": torch.unsqueeze(batch[2], 0),
            "input_ids": torch.unsqueeze(batch[3], 0),
            "attention_mask": torch.unsqueeze(batch[4], 0),
            "token_type_ids": torch.unsqueeze(batch[5], 0),
            "masked_pos": torch.unsqueeze(batch[6], 0),
            "do_sample": False,
            "bos_token_id": cls_token_id,
            "pad_token_id": pad_token_id,
            "eos_token_ids": [sep_token_id],
            "mask_token_id": mask_token_id,
            # for adding od labels
            "add_od_labels": self.bert_cfg.add_od_labels,
            "od_labels_start_posid": self.bert_cfg.max_seq_a_length,
            # hyperparameters of beam search
            "max_length": self.bert_cfg.max_gen_length,
            "num_beams": self.bert_cfg.num_beams,
            "temperature": self.bert_cfg.temperature,
            "top_k": self.bert_cfg.top_k,
            "top_p": self.bert_cfg.top_p,
            "repetition_penalty": self.bert_cfg.repetition_penalty,
            "length_penalty": self.bert_cfg.length_penalty,
            "num_return_sequences": self.bert_cfg.num_return_sequences,
            "num_keep_best": self.bert_cfg.num_keep_best,
        }
        scores, bert_outputs = self(**inputs)
        cap = bert_outputs[0]  # batch_size * num_keep_best * max_len
        conf = torch.squeeze(torch.exp(bert_outputs[1]), 1)
        # when there's only one caption
        vinfo = dict()
        vinfo["image_id"] = video_name
        # vinfo["caption_special"] = self.tokenizer.decode(cap[0][0].tolist(), skip_special_tokens=False)
        vinfo["caption"] = self.tokenizer.decode(
            cap[0][0].tolist(), skip_special_tokens=True
        )
        vinfo["conf"] = conf[0].item()  # caption's confidence
        vinfo["score"] = scores[0].tolist()
        with jsonlines.open(self.val_result_path, "a") as writer:
            writer.write(vinfo)
        return video_name

    def on_validation_end(self):
        self.epoch_num += 1

    def on_test_start(self):
        self.test_result_path = os.path.join(
            self.eval_cfg.test_result_path, "test.json"
        )

    def test_step(self, batch, batch_ids):
        (
            cls_token_id,
            sep_token_id,
            pad_token_id,
            mask_token_id,
            period_token_id,
        ) = self.tokenizer.convert_tokens_to_ids(
            [
                self.tokenizer.cls_token,
                self.tokenizer.sep_token,
                self.tokenizer.pad_token,
                self.tokenizer.mask_token,
                ".",
            ]
        )
        video_name, gt_score = batch[0], batch[1]

        inputs = {
            "is_decode": True,
            "img_feats": torch.unsqueeze(batch[2], 0),
            "input_ids": torch.unsqueeze(batch[3], 0),
            "attention_mask": torch.unsqueeze(batch[4], 0),
            "token_type_ids": torch.unsqueeze(batch[5], 0),
            "masked_pos": torch.unsqueeze(batch[6], 0),
            "do_sample": False,
            "bos_token_id": cls_token_id,
            "pad_token_id": pad_token_id,
            "eos_token_ids": [sep_token_id],
            "mask_token_id": mask_token_id,
            # for adding od labels
            "add_od_labels": self.bert_cfg.add_od_labels,
            "od_labels_start_posid": self.bert_cfg.max_seq_a_length,
            # hyperparameters of beam search
            "max_length": self.bert_cfg.max_gen_length,
            "num_beams": self.bert_cfg.num_beams,
            "temperature": self.bert_cfg.temperature,
            "top_k": self.bert_cfg.top_k,
            "top_p": self.bert_cfg.top_p,
            "repetition_penalty": self.bert_cfg.repetition_penalty,
            "length_penalty": self.bert_cfg.length_penalty,
            "num_return_sequences": self.bert_cfg.num_return_sequences,
            "num_keep_best": self.bert_cfg.num_keep_best,
        }
        scores, bert_outputs = self(**inputs)
        cap = bert_outputs[0]  # batch_size * num_keep_best * max_len
        conf = torch.squeeze(torch.exp(bert_outputs[1]), 1)
        # when there's only one caption
        vinfo = dict()
        vinfo["image_id"] = video_name
        vinfo["caption_special"] = self.tokenizer.decode(
            cap[0][0].tolist(), skip_special_tokens=False
        )
        vinfo["caption"] = self.tokenizer.decode(
            cap[0][0].tolist(), skip_special_tokens=True
        )
        vinfo["conf"] = conf[0].item()  # caption's confidence
        vinfo["score"] = scores[0].tolist()
        with jsonlines.open(self.test_result_path, "a") as writer:
            writer.write(vinfo)
        return video_name

    def configure_optimizers(self):
        hpms = self.hpms
        if hpms.use_lr_finder == False:
            enc_param = list(self.encoder.named_parameters())
            bert_param = list(self.bert.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            decay_param_tp = [
                (n, p) for n, p in bert_param if not any(nd in n for nd in no_decay)
            ]
            no_decay_param_tp = [
                (n, p) for n, p in bert_param if any(nd in n for nd in no_decay)
            ]
            decay_bert_param_tp = [(n, p) for n, p in decay_param_tp]
            no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in enc_param],
                    "weight_decay": hpms.encoder.weight_decay,
                    "lr": hpms.encoder.lr,
                },
                {
                    "params": [p for n, p in decay_bert_param_tp],
                    "weight_decay": hpms.bert.weight_decay,
                    "lr": hpms.bert.lr,
                },
                {
                    "params": [p for n, p in no_decay_bert_param_tp],
                    "weight_decay": 0.0,
                    "lr": hpms.bert.lr,
                },
            ]
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters, lr=hpms.encoder.lr, eps=hpms.adam_epsilon
            )
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[500], gamma=0.1
            )
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[500], gamma=0.1
            )

        return [optimizer], [scheduler]
