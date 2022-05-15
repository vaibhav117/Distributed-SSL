from ast import arg
from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
from matplotlib import image
import wandb

import torch
import torch.nn.functional as F
from torch import embedding, nn, optim
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import augmentations as aug
from distributed import init_distributed_mode

import resnet


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet", required=True, help='Path to the image net dataset')
    parser.add_argument("--dataset", type=str, default="CIFAR10", help='Dataset to be used')

    # Checkpoints
    parser.add_argument("--tta-accuracy", type=float, default=70.0, help='TTA Accuracy')
    parser.add_argument("--run-name", type=str, default="run_name", required=True, help='Name of the run')
    parser.add_argument("--gpu-type", type=str, default="A100", required=True, help='Name of the run')
    parser.add_argument("--exp-dir", type=Path, default="./experiments", help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60, help='Print logs to the stats.txt file every [log-freq-time] seconds')
    parser.add_argument("--eval-run-at", type=int, default=5, help='Numeber of epochs after which eval loop is to be run')
    parser.add_argument("--eval-epochs", type=int, default=5, help='Numeber of epochs to train the eval head in the eval loop')

    # Model
    parser.add_argument("--arch", type=str, default="resnet50", help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="8192-8192-8192", help='Size and number of layers of the MLP expander head')

    # Optim
    parser.add_argument("--epochs", type=int, default=100, help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=2048, help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.2, help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6, help='Weight decay')

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0, help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0, help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0, help='Covariance regularization loss coefficient')

    # Running
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')

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
        image_size = 32
    elif args.dataset == "tiny-imagenet":
        number_of_classes = 1000
        normal_mean = [0.4914, 0.4822, 0.4465]
        normal_std = (0.2023, 0.1994, 0.2010)
        image_size = 64
    elif args.dataset == "imagenet" :
        number_of_classes = 1000
        normal_mean = [0.4914, 0.4822, 0.4465]
        normal_std = (0.2023, 0.1994, 0.2010)
        image_size = 224

    init_distributed_mode(args)
    print(args)
    gpu = torch.device(args.device)

    if args.rank == 0:
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(entity= 'hpml', project=f"HPML-project_{args.arch}" , name=f"{args.run_name}_batchsize={args.batch_size}_lr={args.base_lr}", tags=["train",f"{args.gpu_type}",f"{args.batch_size}",f"{args.world_size}",f"{args.tta_accuracy}",f"{args.dataset}",f"{args.arch}"])
        os.makedirs(os.path.dirname(f"{args.exp_dir}/{args.run_name}/"), exist_ok=True)
        stats_file = open(args.exp_dir / args.run_name / "stats.txt", "a", buffering=1)
        print(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

    train_transforms = aug.TrainTransform()

    eval_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(image_size),
        transforms.Normalize(normal_mean, normal_std)
        ])

    if args.dataset == "imagenet":
        train_dataset = datasets.ImageFolder(f"{args.data_dir}/train", transform=transform_train)
    elif args.dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(root=f'{args.data_dir}/CIFAR10', train=True, download=True, transform=train_transforms)
        eval_dataset = datasets.CIFAR10(root=f'{args.data_dir}/CIFAR10', download=True, train=False, transform=eval_transforms)
    elif args.dataset == "tiny-imagenet":
        train_dataset = datasets.ImageFolder("./tiny-imagenet-200/train", transform=train_transforms)
        eval_dataset = datasets.ImageFolder("./tiny-imagenet-200/test", transform=eval_transforms)

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

    model = VICReg(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    optimizer = LARS(
        model.parameters(),
        lr=0,
        weight_decay=args.wd,
        weight_decay_filter=exclude_bias_and_norm,
        lars_adaptation_filter=exclude_bias_and_norm,
    )

    #------------------- ADDING EVAL CLASSIFIER ---------------
    eval_backbone, embedding = resnet.__dict__[args.arch](zero_init_residual=True)
    eval_head = nn.Linear(embedding, number_of_classes).cuda(gpu)
    eval_head.weight.data.normal_(mean=0.0, std=0.01)
    eval_head.bias.data.zero_()
    eval_model = nn.Sequential(eval_backbone, eval_head)
    eval_model.cuda(gpu)
    eval_backbone.load_state_dict(model.module.backbone.state_dict())
    eval_backbone.requires_grad_(False)
    eval_head.requires_grad_(True)
    eval_model = torch.nn.parallel.DistributedDataParallel(eval_model, device_ids=[gpu])
    #---------------------------------------------------------

    if (args.exp_dir / args.run_name / "model.pth").is_file():
        if args.rank == 0:
            print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / args.run_name / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        eval_head.load_state_dict(ckpt["head"])
    else:
        start_epoch = 0

    start_time = last_logging = time.time()
    scaler = torch.cuda.amp.GradScaler()
    best_train_accuracy = 0
    train_accuracy = 0
    head_loss = 0
    supervised_optim = optim.SGD(eval_head.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        for step, ((x,y), _) in enumerate(train_loader, start=epoch * len(train_loader)):
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)

            lr = adjust_learning_rate(args, optimizer, train_loader, step)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, backbone_out_x, backbone_out_y = model.forward(x, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            current_time = time.time()

            if args.rank == 0 and current_time - last_logging > args.log_freq_time:
                stats = dict(
                    epoch=epoch,
                    step=step,
                    loss=loss.item(),
                    time=int(current_time - start_time),
                    lr=lr,
                )
                print(json.dumps(stats))
                print(json.dumps(stats), file=stats_file)
                last_logging = current_time
        
        if epoch%args.eval_run_at == 0:
            eval_backbone.load_state_dict(model.module.backbone.state_dict())
            eval_backbone.requires_grad_(False)
            eval_head.requires_grad_(True)

           
            for eval_epoch in range(args.eval_epochs):
                total_train = 0
                correct_train = 0
                for step, (image, targets) in enumerate(eval_loader, start=epoch * len(eval_loader)):
                    targets = targets.cuda(gpu, non_blocking=True)
                    image = image.cuda(gpu, non_blocking=True)
                    
                    supervised_optim.zero_grad()

                    outputs = eval_model(image)
                        
                    head_loss = torch.nn.CrossEntropyLoss()(outputs, targets)
                    head_loss.backward()
                    supervised_optim.step()

                    _, predicted = outputs.max(1)
                    total_train += targets.size(0)
                    correct_train += predicted.eq(targets).sum().item()
                    train_accuracy = correct_train/total_train
                if args.rank == 0:
                    wandb.log({ "backbone_loss": loss, "head_loss": head_loss, "train_accuracy": train_accuracy, "best_train_accuracy": best_train_accuracy, "runtime": int(current_time - start_time), "epoch": epoch })

            if args.rank == 0:
                if train_accuracy > best_train_accuracy:
                    best_train_accuracy = train_accuracy
                    torch.save(model.module.backbone.state_dict(), args.exp_dir / args.run_name / "best_backbone.pth")
                    torch.save(eval_head.state_dict(), args.exp_dir / args.run_name / "best_head.pth")

        if args.rank == 0:
            state = dict(
                epoch=epoch + 1,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                head=eval_head.state_dict(),
            )
            wandb.log({ "backbone_loss": loss, "head_loss": head_loss, "train_accuracy": train_accuracy, "best_train_accuracy": best_train_accuracy, "runtime": int(current_time - start_time), "epoch": epoch })
            torch.save(state, args.exp_dir / args.run_name / "model.pth")
    if args.rank == 0:
        torch.save(model.module.backbone.state_dict(), args.exp_dir / args.run_name / "final_resnet50_backbone.pth")
        torch.save(eval_head.state_dict(), args.exp_dir / args.run_name / "final_resnet50_eval_head.pth")


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class VICReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_features = int(args.mlp.split("-")[-1])
        self.backbone, self.embedding = resnet.__dict__[args.arch](
            zero_init_residual=True
        )
        self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        backbone_out_x = self.backbone(x)
        backbone_out_y = self.backbone(y)
        x = self.projector(backbone_out_x)
        y = self.projector(backbone_out_y)

        repr_loss = F.mse_loss(x, y)

        x = torch.cat(FullGatherLayer.apply(x), dim=0)
        y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.args.sim_coeff * repr_loss
            + self.args.std_coeff * std_loss
            + self.args.cov_coeff * cov_loss
        )
        return loss, backbone_out_x.detach().clone(), backbone_out_y.detach().clone()


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class LARS(optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        weight_decay=0,
        momentum=0.9,
        eta=0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                dp = p.grad

                if dp is None:
                    continue

                if g["weight_decay_filter"] is None or not g["weight_decay_filter"](p):
                    dp = dp.add(p, alpha=g["weight_decay"])

                if g["lars_adaptation_filter"] is None or not g[
                    "lars_adaptation_filter"
                ](p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            update_norm > 0, (g["eta"] * param_norm / update_norm), one
                        ),
                        one,
                    )
                    dp = dp.mul(q)

                param_state = self.state[p]
                if "mu" not in param_state:
                    param_state["mu"] = torch.zeros_like(p)
                mu = param_state["mu"]
                mu.mul_(g["momentum"]).add_(dp)

                p.add_(mu, alpha=-g["lr"])


def batch_all_gather(x):
    x_list = FullGatherLayer.apply(x)
    return torch.cat(x_list, dim=0)


class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
