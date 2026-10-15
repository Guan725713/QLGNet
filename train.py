import os
import random
import argparse
import numpy as np
import torch.nn.functional as F
from torch import nn

# 为了更强的 GPU 可复现性（建议放在 import torch 之前）
os.environ["PYTHONHASHSEED"] = "41"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # CUDA matmul 更确定（部分环境需要）

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader.dataset import MedicalDataSets
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize

# from new.utils.model_report import print_paper_like_model_stats, print_param_table
from utils.util import AverageMeter
import utils.losses as losses
from utils.metrics import iou_score


from QLGNet import QLGNet

# =========================
# 全局随机种子
# =========================
SEED = 41

def seed_torch(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cudnn 可复现
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True,warn_only=True)
    except Exception as e:
        print(f"[Warn] torch.use_deterministic_algorithms(True) failed: {e}")

seed_torch(SEED)


# =========================
# DataLoader worker 里也要固定 numpy/random
# =========================
def seed_worker(worker_id: int):
    # 这个 initial_seed 是 DataLoader 根据 generator 派发的
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class DeepSupervisionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # criterion = losses.__dict__['BCEDiceLoss']().cuda()
        # self.base_loss = losses.BCEDiceLoss()
        self.base_loss = losses.__dict__['BCEDiceLoss']().cuda()

    def forward(self, preds, target):
        """
        preds = (out, pred1, pred2)
        """
        out, p1, p2 = preds

        target1 = F.interpolate(target, size=p1.shape[2:], mode='nearest')
        target2 = F.interpolate(target, size=p2.shape[2:], mode='nearest')

        loss_out = self.base_loss(out, target)
        loss_p1 = self.base_loss(p1, target1)
        loss_p2 = self.base_loss(p2, target2)

 
        loss = loss_out + 0.4 * loss_p1 + 0.2 * loss_p2
        return loss

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="QLGNet",
                    choices=["mobileuvit", "QLGNet"], help='model')

parser.add_argument('--base_dir', type=str, default="/mnt/d/data/BUSI/BUSI/train", help='dir')
parser.add_argument('--train_file_dir', type=str,default="/mnt/d/code/Mobile-U-ViT/BUSI_7_3/splits_txt/split1_train.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="/mnt/d/code/Mobile-U-ViT/BUSI_7_3/splits_txt/split1_val.txt",
                    help='dir')

parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
args = parser.parse_args()


def getDataloader():
    img_size = 256
    train_transform = Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])

    val_transform = Compose([
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])

    db_train = MedicalDataSets(
        base_dir=args.base_dir, split="train", transform=train_transform,
        train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir
    )
    db_val = MedicalDataSets(
        base_dir=args.base_dir, split="val", transform=val_transform,
        train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir
    )
    print("train num:{}, val num:{}".format(len(db_train), len(db_val)))

    # 关键：generator 固定 shuffle 顺序（跨 run 一致）
    g = torch.Generator()
    g.manual_seed(SEED)

    trainloader = DataLoader(
        db_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,                 
        worker_init_fn=seed_worker,    
        generator=g,                   
        pin_memory=True,
        persistent_workers=True        
    )

    valloader = DataLoader(
        db_val,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        worker_init_fn=seed_worker,    
        pin_memory=True,
        persistent_workers=True
    )

    return trainloader, valloader


def get_model(args):
    if args.model == "mobileuvit":
        model = mobileuvit()
    elif args.model == "QLGNet":
        model = QLGNet()
    else:
        raise ValueError("model err")
    return model.cuda()


def train(args):
    base_lr = args.base_lr
    trainloader, valloader = getDataloader()
    model = get_model(args)

    print("train file dir:{} val file dir:{}".format(args.train_file_dir, args.val_file_dir))
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = losses.__dict__['BCEDiceLoss']().cuda()

    print("{} iterations per epoch".format(len(trainloader)))

    best_iou = 0.0
    iter_num = 0
    max_epoch = 200
    max_iterations = len(trainloader) * max_epoch

    for epoch_num in range(max_epoch):
        model.train()
        avg_meters = {
            'loss': AverageMeter(),
            'iou': AverageMeter(),
            'val_loss': AverageMeter(),
            'val_iou': AverageMeter(),
            'val_dice': AverageMeter(),
            'SE': AverageMeter(),
            'PC': AverageMeter(),
            'F1': AverageMeter(),
            'ACC': AverageMeter()
        }

        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(non_blocking=True), label_batch.cuda(non_blocking=True)

            criterion = DeepSupervisionLoss().cuda()

            outputs = model(volume_batch)
            loss = criterion(outputs, label_batch)
            if isinstance(outputs, (tuple, list)):
                main_out = outputs[0]  #  只用最终输出算指标
            else:
                main_out = outputs
            iou, dice, _, _, _, _, _ = iou_score(main_out, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1
            avg_meters['loss'].update(loss.item(), volume_batch.size(0))
            avg_meters['iou'].update(iou, volume_batch.size(0))

        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                inp, target = sampled_batch['image'], sampled_batch['label']
                inp = inp.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

                criterion = DeepSupervisionLoss().cuda()
                output = model(inp)
                loss = criterion(output, target)
                if isinstance(output, (tuple, list)):
                    main_out = output[0]  #  只用最终输出算指标
                else:
                    main_out = output
                iou, dice, SE, PC, F1, _, ACC = iou_score(main_out, target)
                avg_meters['val_loss'].update(loss.item(), inp.size(0))
                avg_meters['val_iou'].update(iou, inp.size(0))
                avg_meters['val_dice'].update(dice, inp.size(0))
                avg_meters['SE'].update(SE, inp.size(0))
                avg_meters['PC'].update(PC, inp.size(0))
                avg_meters['F1'].update(F1, inp.size(0))
                avg_meters['ACC'].update(ACC, inp.size(0))

        print(
            'epoch [%d/%d]  train_loss : %.4f, train_iou: %.4f '
            '- val_loss %.4f - val_iou %.4f -val_dice %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_ACC %.4f'
            % (epoch_num, max_epoch, avg_meters['loss'].avg, avg_meters['iou'].avg,
               avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_dice'].avg,
               avg_meters['SE'].avg, avg_meters['PC'].avg, avg_meters['F1'].avg, avg_meters['ACC'].avg)
        )

        if avg_meters['val_iou'].avg > best_iou:
            os.makedirs('BUSI', exist_ok=True)

            train_file_name = os.path.basename(args.train_file_dir)
            train_stub = os.path.splitext(train_file_name)[0]

            save_path = os.path.join('BUSI', f'{args.model}_model_BUSI_seed_{train_stub}.pth')
            torch.save(model.state_dict(), save_path)

            best_iou = avg_meters['val_iou'].avg
            print(f"=> saved best model to {save_path} (best_iou={best_iou:.4f})")

    return "Training Finished!"


if __name__ == "__main__":
    train(args)
