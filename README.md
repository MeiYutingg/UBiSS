# UBiSS
This is the official implementation of the ICLR24 paper *"UBiSS: A Unified Framework for Bimodal Semantic Summarization of Videos"*


## BIDS Dataset

We provide the extracted CNN and VideoSwinTransformer visual features of **BIDS** dataset in `data/visual_feauture`. You can load feature from .h5 files as the following example shows:

```
import numpy as np
import h5py

with h5py.File("data/visual_feature/qvhighlight_2s.h5", "r") as f:
    for video_name in f.keys():
        feature = np.array(f[video_name]['features']) # shape: [t, 1024]
```

For original videos, come  to ... to download (download website will be updated later). The feature provided is sufficient for running the training and evaluation script. You will only need original videos to calculate $NDCG_{TM}$.

## Training & Inference

Firstly, set up the environment by cloning this repo and running `conda env create -f environment.yml`. Then, you can activate the conda environment by `conda activate ubiss`.

### Training

An example training config is included in `config/transformer_encoder_bert.yml`. There are several important items to set:

- seed: the random seed during training
- model/loss/sum_loss: this should be either "neuralNDCG" or "MSE" for visual summary supervision
- data/num_workers: this can be set larger or smaller according to your computation device
- data/paths: the directory of data files
- lightning/epochs: training epoch number
- lightning/save_ckpt_path: where to save your checkpoints
- lightning/load_ckpt_path: where to load checkpoint for continual training
- hparams/language_warm_up_epoch: use how many epochs for textual summary warm-up
- evaluation/result_path: where to save evaluation results. This file directory should be pre-created and is recommended to be the same as the parent directory of `save_ckpt_path`
- wandb/mode: change it to "disabled" if you don't want to use wandb


After setting the config properly, you can run `python main.py --base config/transformer_encoder_bert.yml --gpus 0,1,2,3,4,5,6,7` to start training (p.s. if you use only one gpu, there should be a comma in the end, such as `--gpus 0,`). You can find checkpoints and mid-way validation results in `result_path` that you set. You can also check the loss function following the guide of your wandb.

### Inference and Models

We provide the checkpoints and results of UBiSS(MSE&NeuralNDCG) in [here](https://1drv.ms/f/c/97dec68abb271787/EoliBgXKnDdMgFkTdt0jVhIBW-snVz1HXZPaOBrXnqZ8Ug). For each model, two checkpoints are provided selected by its performance on the validation set according to CIDEr/$\tau$. The perfomance are as follows:

|                              | F-score | $tau$ | $rho$ | $NDCG_{VM}@15%$ | $NDCG_{VM}@all$ |
|------------------------------|---------|-------|-------|-----------------|-----------------|
| UBiSS(NeuralNDCG, epoch=054) | 20.85   | 18.47 | 23.01 | 67.12           | 85.25           |
| UBiSS(MSE, epoch=021)        | 19.42   | 16.81 | 21.05 | 65.84           | 84.69           |

|                              | B4   | M     | R-L   | C     | S     |
|------------------------------|------|-------|-------|-------|-------|
| UBiSS(NeuralNDCG, epoch=047) | 4.81 | 10.57 | 23.39 | 41.62 | 13.44 |
| UBiSS(MSE, epoch=090)        | 4.17 | 9.84  | 21.90 | 41.81 | 13.61 |


For inference in the test set of BIDS, you also need to set the `load_ckpt_path` and `result_path` of the config first (an example is provided in `config/inference`).  Then, comment Line 160 of `inference.py` and use `python inference.py --base config/inference.yml --gpus 0,` to inference the results. If you want to get evaluation results, uncomment L162 and comment L157, then run `python inference.py --base config/inference.yml --gpus 0,` for evaluation (p.s. either L162 or L157 of `inference.py` must be commented due to some parallel problems).

If you need to inference on your own videos, you need to first extract visual features into a .h5 file, then change the `data` part of inference config.

### Evaluation

For textual summary evaluation, we adopt [coco-caption](https://github.com/tylin/coco-caption) evaluation. Besides, you need to clone [CIDEr](https://github.com/vrama91/cider) under `evaluation/evalcap` before evaluation.

For visual summary evaluation, all metrics except for $NDCG_{TM}$ are calculated as in `evaluation\evalsum\evaluate_summary.py`. Since the calculation of $NDCG_{TM}$ is much slower than other metrics, you will need to modify the directories and run `evaluation\textndcg.py` to get this metric.

The pre-processing procedure in `evaluation\evalsum\evaluate_summary.py` is the same as data preprocessing.

## Acknowledgement

Many thanks to [SwinBERT](https://github.com/microsoft/SwinBERT), [Moment_detr](https://github.com/jayleicn/moment_detr), [CTVSUM](https://github.com/pangzss/pytorch-CTVSUM), and [PGL-SUM](https://github.com/e-apostolidis/PGL-SUM).

This work was partially supported by the the National Natural Science Foundation of China (No. 62072462) and Beijing Natural Science Foundation (No. L233008).

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{mei2024ubiss,
    title={UBiSS: A Unified Framework for Bimodal Semantic Summarization of Videos},
    author={Mei, Yuting and Yao, Linli and Jin, Qin},
    booktitle={Proceedings of the 2024 International Conference on Multimedia Retrieval},
    pages={1034--1042},
    year={2024}
}
```