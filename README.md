# HPML-project

### Run Command for Self-Supervised training
```
python -m torch.distributed.launch --nproc_per_node=2 train_self_supervised.py --arch resnet50 --epochs 100 --batch-size 32 --base-lr 0.3 --dataset CIFAR10 --data-dir ./ --run-name resnet50_run --gpu-type RTX3080
```

### Run Command for Classification Downstream Finetuning
```
python -m torch.distributed.launch --nproc_per_node=2 train_classification_downstream.py --arch resnet50 --epochs 100 --batch-size 64 --dataset CIFAR10 --data-dir ./ --run-name downstream_classification --gpu-type RTX3080
```