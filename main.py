import argparse, os, sys
import wandb
from omegaconf import OmegaConf

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model.transformer_encoder_bert import SumCapModel
from model.layers.bert.config import add_bert_parser
from dataset.data_module import QVHDataModule
from evaluation.eval import evaluate


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])

    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def nondefault_bert_args(opt):
    parser = argparse.ArgumentParser()
    parser = add_bert_parser(parser)
    args = parser.parse_args([])

    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


if __name__ == "__main__":
    sys.path.append(os.getcwd())

    # set parser
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    parser = add_bert_parser(parser)
    opt, unknown = parser.parse_known_args()
    configs = [OmegaConf.load(cfg) for cfg in opt.base]

    # merge configs
    cli = OmegaConf.from_dotlist(unknown)
    configs = OmegaConf.merge(*configs, cli)

    seed_everything(configs.seed, workers=True)

    # set up important trainer flags
    # merge trainer cli with config
    trainer_cfg = configs.lightning.get("trainer", OmegaConf.create())
    trainer_cfg.logger, trainer_cfg.deterministic = False, True
    for k in nondefault_trainer_args(opt):  # this incorporates cli into trainer configs
        trainer_cfg[k] = getattr(opt, k)

    if not "gpus" in trainer_cfg:
        cpu = True
    else:
        gpuinfo = trainer_cfg["gpus"]
        print(f"Running on GPUs {gpuinfo}")
        cpu = False

    trainer_opt = argparse.Namespace(**trainer_cfg)
    configs.lightning.trainer = trainer_cfg

    mode = configs.mode  # input is 2s mean-pooled / frame feature
    enc_cfg = configs.model.get("encoder", OmegaConf.create())
    bert_cfg = configs.model.get("bert", OmegaConf.create())  # get configs from yml
    loss_cfg = configs.model.get("loss", OmegaConf.create())
    pretrained_bert_path = configs.model.pretrained_bert_path
    data_cfg = configs.get("data", OmegaConf.create())
    eval_cfg = configs.get("evaluation", OmegaConf.create())
    wandb_cfg = configs.get("wandb", OmegaConf.create())
    hpms = configs.get("hparams", OmegaConf.create())

    # get bert configs
    for k in vars(opt):
        if k not in bert_cfg.keys():  # merge default bert attributes
            bert_cfg[k] = getattr(opt, k)
        elif k in nondefault_bert_args(opt):  # merge attributes set in command line
            bert_cfg[k] = getattr(opt, k)
    if mode == "2s":
        bert_cfg["max_img_seq_length"] = data_cfg.num_frames.mean2s
        eval_cfg["num_frames"] = data_cfg.num_frames.mean2s
    elif mode == "frame":
        bert_cfg["max_img_seq_length"] = data_cfg.num_frames.frame
        eval_cfg["num_frames"] = data_cfg.num_frames.frame

    # set trainer callbacks
    if not cpu:
        ngpu = len(configs.lightning.trainer.gpus.strip(",").split(","))
    else:
        ngpu = 0

    trainer_kwargs = dict()
    if configs.lightning.save_ckpt_path != "":
        checkpoint_callback = ModelCheckpoint(
            dirpath=configs.lightning.save_ckpt_path,
            filename="{epoch:03d}",
            save_last=True,
            every_n_epochs=1,
            save_top_k=-1,
        )
        trainer_kwargs["callbacks"] = [checkpoint_callback]

    # create paths
    eval_cfg.val_result_path = os.path.join(eval_cfg.result_path, "val_result")
    eval_cfg.eval_result_path = os.path.join(eval_cfg.result_path, "eval_result")
    eval_cfg.test_result_path = os.path.join(eval_cfg.result_path, "test_result")
    try:
        os.makedirs(eval_cfg.val_result_path, mode=0o777)
        os.makedirs(eval_cfg.eval_result_path, mode=0o777)
        os.makedirs(eval_cfg.test_result_path, mode=0o777)
    except:
        pass

    # set up logger
    # set wandb_cfg.mode: disabled/online to stop/start wandb from logging
    wandb_logger = WandbLogger(
        project="SumCap", config=configs, name=wandb_cfg.run_name, mode=wandb_cfg.mode
    )
    trainer_kwargs["logger"] = wandb_logger
    if hpms.use_lr_finder == False:
        trainer_kwargs["strategy"] = "ddp"
    # initialize trainer
    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

    # load model
    model = SumCapModel(
        enc_cfg=enc_cfg,
        bert_cfg=bert_cfg,
        loss_cfg=loss_cfg,
        eval_cfg=eval_cfg,
        hpms=hpms,
        mode=mode,
        pretrained_bert_path=pretrained_bert_path,
    )
    print("Model loaded!")

    # load data module
    tokenizer = model.tokenizer
    qvh_data = QVHDataModule(
        bert_cfg=bert_cfg, data_cfg=data_cfg, hpms=hpms, tokenizer=tokenizer, mode=mode
    )
    print("Data loaded!")

    # tune
    if hpms.use_lr_finder == True:
        trainer.tune(model, qvh_data)

    # train
    # decide whether to resume from a ckpt
    if configs.lightning.load_ckpt_path != "":
        trainer.fit(model, qvh_data, ckpt_path=configs.lightning.load_ckpt_path)
    else:
        trainer.fit(model, qvh_data)

    # evaluate
    evaluate(eval_cfg=eval_cfg, mode=mode)
