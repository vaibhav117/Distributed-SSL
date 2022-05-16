from ast import arg
from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import copy
from matplotlib import image
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import augmentations as aug


def get_arguments():
    parser = argparse.ArgumentParser(description="Use a pretrained Resnet model on a classification problem", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="/path/to/CIFAR10", help='Path to the dataset')
    parser.add_argument("--dataset", type=str, default="CIFAR10", help='Dataset to be used')

    # Hardware
    parser.add_argument("--gpu-type", type=str, default="V100", required=True, help='Name of the run')
    parser.add_argument('--num_gpu', type=int, default="4", help='num of gpus')

    # Checkpoints
    parser.add_argument("--tta-accuracy", type=float, default=85.0, help='TTA Accuracy')
    parser.add_argument("--run-name", type=str, default="run_name", required=True, help='Name of the run')
    parser.add_argument("--exp-dir", type=Path, default="./experiments", help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60, help='Print logs to the stats.txt file every [log-freq-time] seconds')
    
    # Model
    parser.add_argument("--arch", type=str, default="resnet50_pretrained", help='Architecture of the pretrained network')
    
    # Optim
    parser.add_argument('--optim', default='SGD', type=str, help='optimizer')
    parser.add_argument("--lr", type=float, default=0.1, help='Base learning rate')
    parser.add_argument("--wd", type=float, default=5e-4, help='Weight decay')
    parser.add_argument("--momentum", type=float, default=0.9, help='Momentum')

    # Training
    parser.add_argument("--epochs", type=int, default=100, help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=128, help='Effective batch size')
    
    # Running
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

    return parser


def main(args):
    args.device = None
    if torch.cuda.is_available():
        args.device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True

    if args.dataset == "CIFAR10":
        number_of_classes = 10
        normal_mean = (0.4914, 0.4822, 0.4465)
        normal_std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == "imagenet" or args.dataset == "tiny-imagenet":
        number_of_classes = 1000
        normal_mean = [0.4914, 0.4822, 0.4465]
        normal_std = (0.2023, 0.1994, 0.2010)

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    wandb.init(entity= 'hpml', project=f"HPML-project_{args.arch}" , name=f"{args.run_name}_batchsize={args.batch_size}_lr={args.lr}", tags=["train",f"{args.gpu_type}",f"{args.batch_size}",f"{args.tta_accuracy}",f"{args.dataset}",f"{args.arch}"])
    os.makedirs(os.path.dirname(f"{args.exp_dir}/{args.run_name}/"), exist_ok=True)
    stats_file = open(args.exp_dir / args.run_name / "stats.txt", "a", buffering=1)
    print(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)

    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(normal_mean, normal_std)
        ])

    eval_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        transforms.Normalize(normal_mean, normal_std)
        ])

    if args.dataset == "imagenet":
        train_dataset = datasets.ImageFolder(f"{args.data_dir}/train", transform=train_transforms)
    if args.dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(root=f'{args.data_dir}/CIFAR10', train=True, download=True, transform=train_transforms)
        eval_dataset = datasets.CIFAR10(root=f'{args.data_dir}/CIFAR10', train=False, download=True, transform=eval_transforms)
    elif args.dataset == "tiny-imagenet":
        train_dataset = datasets.ImageFolder("./tiny-imagenet-200/train", transform=train_transforms)

    eff_batch_size = int(args.batch_size * args.num_gpu)

    data_loaders = {
            'train':
            torch.utils.data.DataLoader(train_dataset,
                                        batch_size=eff_batch_size,
                                        num_workers=args.num_workers),
            'validation':
            torch.utils.data.DataLoader(eval_dataset,
                                        batch_size=eff_batch_size,
                                        num_workers=args.num_workers)
        }

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, number_of_classes)
    model = model.to(device=args.device)
    # for param in model.parameters():
    #     param.requires_grad = False 

    # Using DataParallel
    if args.num_gpu == 2:
        CUDA_VISIBLE_DEVICES=0,1
        model = torch.nn.DataParallel(model, device_ids=[0,1])
    if args.num_gpu == 4:
        CUDA_VISIBLE_DEVICES=0,1,2,3
        model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])

    criterion = nn.CrossEntropyLoss()
    if args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        
    start_epoch = 0
    start_time = last_logging = time.time()
    best_train_accuracy = 0

    for epoch in range(start_epoch, args.epochs):

        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            current_loss = 0.0
            current_corrects = 0

            for step, (inputs, labels) in enumerate(data_loaders[phase], 1):
                inputs = inputs.to(device=args.device)
                labels = labels.to(device=args.device)

                optimizer.zero_grad()

                # Forward training poss
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # We want variables to hold the loss statistics
                current_loss += loss.item() * inputs.size(0)
                current_corrects += torch.sum(preds == labels.data)

                current_time = time.time()

                # if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                if current_time - last_logging > args.log_freq_time:
                    stats = dict(
                        epoch=epoch,
                        step=step,
                        loss=loss.item(),
                        time=int(current_time - start_time),
                        lr=args.lr
                    )
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
                    last_logging = current_time

            epoch_loss = current_loss / len(data_loaders[phase].dataset)
            epoch_acc = current_corrects.double() / len(data_loaders[phase].dataset)

            # Make a copy of the model if the accuracy on the validation set has improved
            if phase == 'validation' and epoch_acc > best_train_accuracy:
                best_train_accuracy = epoch_acc
                # best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), args.exp_dir / args.run_name / "best_model.pth")

            current_time = time.time()
            wandb.log({ "train_loss": epoch_loss, "train_accuracy": epoch_acc, "best_train_accuracy": best_train_accuracy, "runtime": int(current_time - start_time), "epoch": epoch })


    # Now we'll load in the best model weights and return it
    # model.load_state_dict(best_model_wts)
    model.load_state_dict(torch.load(args.exp_dir / args.run_name / "best_model.pth")) 

    was_training = model.training
    model.eval()
    current_loss = 0.0
    current_corrects = 0

    with torch.no_grad():
        start_time = time.time()
        for step, (inputs, labels) in enumerate(data_loaders['validation']):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            current_loss += loss.item() * inputs.size(0)
            current_corrects += torch.sum(preds == labels.data)

        valid_loss = current_loss / len(data_loaders['validation'].dataset)
        valid_acc = current_corrects.double() / len(data_loaders['validation'].dataset)

        current_time = time.time()
        wandb.log({ "valid_loss": valid_loss, "valid_accuracy": valid_acc, "runtime": int(current_time - start_time)})


        model.train(mode=was_training)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Supervised training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)