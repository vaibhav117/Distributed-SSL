import augmentations as aug
from distributed import init_distributed_mode

from pathlib import Path
import argparse
import json
import os
import random
import signal
import sys
import time
import urllib
from numpy import number
import wandb

from torch import nn, optim
from torchvision import datasets, transforms
import torch

import resnet

def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with Self Supervised", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="./", help='Path to the image net dataset')
    parser.add_argument("--dataset", type=str, default="CIFAR10", help='Dataset to be used')

    # Backbone
    parser.add_argument("--backbone-path", type=Path, default="./pretrained_backbones/", required=True, help='Path to the backbone feature extractor')
    parser.add_argument("--backbone-pretraining", type=str, default="self-supervised", required=True, help='Type of the backbone pretraining')
    

    # Model
    parser.add_argument("--arch", type=str, default="resnet50", help='Architecture of the backbone encoder network')
    parser.add_argument("--head-save-path", type=Path, default="./trained_heads", help='Path to store the final trained heads')

    # Optim
    parser.add_argument("--epochs", type=int, default=100, help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=32, help='Effective batch size (per worker batch size is [batch-size] / world-size)')

    # Running
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument("--tta-accuracy", type=float, default=70.0, help='TTA Accuracy')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    return parser

def main(args):
    torch.backends.cudnn.benchmark = True
    if args.dataset == "CIFAR10":
        number_of_classes = 10
        normal_mean = (0.4914, 0.4822, 0.4465)
        normal_std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == "imagenet" or args.dataset == "tiny-imagenet":
        number_of_classes = 1000
        normal_mean = [0.4914, 0.4822, 0.4465]
        normal_std = (0.2023, 0.1994, 0.2010)

    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    if args.rank == 0:
        os.makedirs(os.path.dirname(f"{args.head_save_path}/"), exist_ok=True)
        wandb.init(entity= 'hpml', project=f"HPML-project_classification-train_{args.arch}" , name=f"downstream_classification_backbone-pretraining{args.backbone_pretraining}_batchsize={args.batch_size}_", tags=["train-downstream-classification",f"backbone_pretraining_type:{args.backbone_pretraining}",f"batch_size:{args.batch_size}",f"GPU_count:{args.world_size}",f"TTA:{args.tta_accuracy}",f"dataset:{args.dataset}",f"backbone_arch:{args.arch}"])
        
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(normal_mean, normal_std)
        ])

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normal_mean, normal_std)
        ])

    if args.dataset == "imagenet":
        train_dataset = datasets.ImageFolder(f"{args.data_dir}/train", transform=transform_train)
    elif args.dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(root=f'{args.data_dir}/CIFAR10', train=True, download=True, transform=transform_train)
        eval_dataset = datasets.CIFAR10(root=f'{args.data_dir}/CIFAR10', download=True, train=False, transform=transform_valid)
    elif args.dataset == "tiny-imagenet":
        train_dataset = datasets.ImageFolder("./tiny-imagenet-200/train", transform=transform_train)
        eval_dataset = datasets.ImageFolder("./tiny-imagenet-200/test", transform=transform_valid)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset, shuffle=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=per_device_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=eval_sampler,
    )

    #------------------- Preparing the model ---------------
    backbone, embedding = resnet.__dict__[args.arch](zero_init_residual=True)
    head, model = get_model(args, backbone, embedding, number_of_classes, gpu)
    #---------------------------------------------------------
    supervised_optim = optim.SGD(head.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    best_train_accuracy = 0
    start_time = last_logging = time.time()

    for epoch in range(args.epochs):
        total_train = 0
        correct_train = 0
        for step, (image, targets) in enumerate(train_loader, start=epoch * len(eval_loader)):
            targets = targets.cuda(gpu, non_blocking=True)
            image = image.cuda(gpu, non_blocking=True)
            
            supervised_optim.zero_grad()
            outputs = model(image)
                        
            loss = torch.nn.CrossEntropyLoss()(outputs, targets)
            loss.backward()
            supervised_optim.step()

            _, predicted = outputs.max(1)
            total_train += targets.size(0)
            correct_train += predicted.eq(targets).sum().item()
            train_accuracy = correct_train/total_train
            
        if args.rank == 0:
            current_time = time.time()
            stats = dict(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    time=int(current_time - start_time),
                    train_accuracy=train_accuracy,
                )
            print(json.dumps(stats))
            if train_accuracy > best_train_accuracy:
                torch.save(head.state_dict(), args.head_save_path / f"{args.backbone_pretraining}_trained_head.pth")
            wandb.log({ "loss": loss, "train_accuracy": train_accuracy, "best_train_accuracy": best_train_accuracy, "runtime": int(current_time - start_time), "epoch": epoch })



def get_model(args, backbone, embedding_size, number_of_classes, gpu):
    # Backbone weights
    backbone_weights = torch.load(args.backbone_path, map_location="cpu")
    backbone.load_state_dict(backbone_weights)
    backbone.requires_grad_(False)

    # head weights
    head = nn.Sequential(
                nn.Linear(embedding_size, 1024), 
                nn.ReLU(),
                nn.Linear(1024,number_of_classes)
            ).cuda(gpu)

    weight_init(head)
    head.requires_grad_(True)

    model = nn.Sequential(backbone, head).cuda(gpu)
    return head, torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
 

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Self-Supervised training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)