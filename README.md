
# Modification Type/Purpose

Pulled from binary [classification fork](https://github.com/mawady/convit.git)
Type: -adding the pre_save stage for setting zero state of the model state dictionary.
Purpose: - for meta-heuristic learning of hyper-params or grid search.
> model can be reloaded at each stage from zero state for next hyperparameter iteration in hyperparameter search when re-running training with new parameters thus returning to initial state rather than last trained state when running as a subprocess(s’).
> grid search code to follow this fork


# New Commands

- Save model checkpoint at 0th state.
- Train:  

```
main.py --pre_save --pre_save_path path/to/zero_state_directory/checkpiont.path --pretrained --batch-size 30 --output_dir /path/to/models --model convit_tiny --epochs 5 --data-path /path/to/dataset --nb_classes 2 --nb_classes_pre 1000
```
 

below from [classification fork](https://github.com/mawady/convit.git)

- Train: 
```
python main.py --pretrained --batch-size 30 --output_dir /path/to/models --model convit_tiny --epochs 5 --data-path /path/to/dataset --nb_classes 2 --nb_classes_pre 1000
```

- Eval: 

``` python main.py --eval --model convit_tiny --resume /path/to/models/checkpoint.pth --data-path /path/to/dataset --nb_classes 2 --nb_classes_pre 1000
```

# ConViT : Vision Transformers with Convolutional Inductive Biases

This repository contains PyTorch code for ConViT. It builds on code from the [Data-Efficient Vision Transformer](https://github.com/facebookresearch/deit) and from [timm](https://github.com/rwightman/pytorch-image-models).

For details see the [ConViT paper](https://arxiv.org/abs/2103.10697) by Stéphane d'Ascoli, Hugo Touvron, Matthew Leavitt, Ari Morcos, Giulio Biroli and Levent Sagun.

If you use this code for a paper please cite:

```
@article{d2021convit,
  title={ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases},
  author={d'Ascoli, St{\'e}phane and Touvron, Hugo and Leavitt, Matthew and Morcos, Ari and Biroli, Giulio and Sagun, Levent},
  journal={arXiv preprint arXiv:2103.10697},
  year={2021}
}
```

# Usage

Install PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models):

```
conda install -c pytorch pytorch torchvision
pip install timm==0.3.2
```

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Evaluation
To evaluate ConViT-Ti on ImageNet test set, run:
```
python main.py --eval --model convit_tiny --pretrained --data-path /path/to/imagenet
```

This should give
```
Acc@1 73.116 Acc@5 91.710 loss 1.172
```

## Training
To train ConViT-Ti on ImageNet on a single node with 4 gpus for 300 epochs run:

```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model convit_tiny --batch-size 256 --data-path /path/to/imagenet
```

To train the same model on a subsampled version of ImageNet where we only use 10% of the images of each class, add ```--sampling_ratio 0.1```

## Multinode training
Distributed training is available via Slurm and [submitit](https://github.com/facebookincubator/submitit):

```
pip install submitit
```

To train ConViT-base on ImageNet on 2 nodes with 8 gpus each for 300 epochs:
```
python run_with_submitit.py --model convit_base --data-path /path/to/imagenet
```

# License
The majority of this repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
