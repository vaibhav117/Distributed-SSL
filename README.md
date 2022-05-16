# HPML-project

### Run Command on Single Machine
```
python -m torch.distributed.launch --nproc_per_node=2 train_self_supervised.py --arch resnet50 --epochs 100 --batch-size 32 --base-lr 0.3 --dataset CIFAR10 --data-dir ./ --run-name resnet50_run --gpu-type RTX3080
```

### Run Command on Single Machine
```
python -m torch.distributed.launch --nproc_per_node=2 train_classification_downstream.py --arch resnet50 --epochs 100 --batch-size 32 --base-lr 0.3 --dataset CIFAR10 --data-dir ./ --run-name resnet50_trial_2 --gpu-type RTX3080
```