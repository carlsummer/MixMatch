# mixmatch training process
import os
import torch
import numpy as np
from config import HP
import torchvision.transforms as transforms
import random
import torch.utils.data as data
import dataset.cifar10 as dataset
from tensorboardX import SummaryWriter

from loss import MixUpLoss
from model import WideResnet50_2
from utils import *
from argparse import ArgumentParser
import torch.optim as optim
import torch.nn as nn

# seed init
torch.manual_seed(HP.seed)
torch.cuda.manual_seed(HP.seed)
random.seed(HP.seed)
np.random.seed(HP.seed)

# stochastic transformation for training
transform_train = transforms.Compose([
    dataset.RandomPadandCrop(32),
    dataset.RandomFlip(),
    dataset.ToTensor(),
])

# inference /validation / test
transform_val = transforms.Compose([
    dataset.ToTensor(),
])

# labeled dataloader / 2 unlabeled dataloaders / validation dataloader
train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_cifar10(r"E:\datasets",
                                                                                n_labeled=HP.n_labeled,
                                                                                transform_train=transform_train,
                                                                                transform_val=transform_val)
labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=HP.batch_size, shuffle=True, drop_last=True)
unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=HP.batch_size, shuffle=True, drop_last=True)
val_loader = data.DataLoader(val_set, batch_size=HP.batch_size, shuffle=False, drop_last=False)

logger = SummaryWriter("./log")


# shadow ema model
def new_ema_model():
    model = WideResnet50_2()
    model = model.to(HP.device)
    for param in model.parameters():
        param.detach_()  # disable gradient trace 去除梯度
    return model


# save func
def save_checkpoint(model_, ema_model_, epoch_, optm, checkpoint_path):
    save_dict = {
        "epoch": epoch_,
        "model_state_dict": model_.state_dict(),
        'ema_model_state_dict': ema_model_.state_dict(),
        'optimizer_state_dict': optm.state_dict(),
    }
    torch.save(save_dict, checkpoint_path)


# evaluation func: loss(CE),
def evaluate(model_, val_loader_, crit):
    model_.eval()
    sum_loss = 0.
    acc1, acc5 = 0., 0.
    with torch.no_grad():
        for batch in val_loader_:
            # load eval data
            inputs_x, trg_x = batch
            inputs_x, trg_x = inputs_x.to(HP.device), trg_x.long().to(HP.device)
            out_x = model_(inputs_x)  # model inference
            top1, top5 = accuracy(out_x, trg_x, topk=(1, 5))
            acc1 += top1
            acc5 += top5
            sum_loss += crit(out_x, trg_x)
    loss = sum_loss / len(val_loader_)
    acc1 = acc1 / len(val_loader_)
    acc5 = acc5 / len(val_loader_)
    model_.train()
    return acc1, acc5, loss


def train():
    parser = ArgumentParser(description='Model Training')
    parser.add_argument(
        "--c",
        default=None,
        type=str,
        help='train from seratch or resume from checkpoint'
    )
    args = parser.parse_args()

    # new models: model/ema_model and WeightEMA instance
    model = WideResnet50_2()
    model = model.to(HP.device)
    ema_model = new_ema_model()
    model_ema_opt = WeightEMA(model, ema_model)

    # loss
    criterion_val = nn.CrossEntropyLoss()  # for eval
    criterion_train = MixUpLoss()  # for training

    opt = optim.Adam(model.parameters(), lr=HP.init_lr, weight_decay=0.001)  # optimizer with L2 regular

    start_epoch, step = 0, 0
    if args.c:
        checkpoint = torch.load(args.c)
        model.load_state_dict(checkpoint['model_state_dict'])
        ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint['epoch']
        print('resume From $s.' % args.c)
    else:
        print('Training from scratch!')

    model.train()
    eval_loss = 0
    n_unlabeled = len(unlabeled_trainloader)  # as regist count for training step

    # train loop
    for epoch in range(start_epoch, HP.epochs):
        print("Start epoch: %d,step: %d" % (epoch, n_unlabeled))
        for i in range(n_unlabeled):  # one unlabeled data turn as an epoch
            # inputs_x:[N,3,H,W],trg_x:[N,]
            inputs_x, trg_x = next(iter(labeled_trainloader))  # get one batch from alabeled dataloader
            # inputs_u / inputs_u2 -> [N,3,H,W]
            (inputs_u, inputs_u2), _ = next(iter(unlabeled_trainloader))
            inputs_x, trg_x, inputs_u, inputs_u2 = inputs_x.to(HP.device), trg_x.long().to(HP.device), inputs_u.to(
                HP.device), inputs_u2.to(HP.device)

            # Algorithm Line7-Line8 Label Guessing
            with torch.no_grad():
                out_u = model(inputs_u)  # Aug K=1,inference [N,10]
                out_u2 = model(inputs_u2)  # Aug K=2,inference [N,10]
                q = label_guessing(out_u, out_u2)  # average post distribution [N,10]
                q = sharpen(q, T=HP.T)  # [ N,10],

            # Algorithm Line10-Line15: Label Guessing
            # mixuped_x: [3*N,3,H,W],mixuped_out:[3*N,10]
            mixuped_x, mixuped_out = mixup(x=inputs_x, u=inputs_u, u2=inputs_u2, trg_x=trg_x, out_u=q, out_u2=q)

            # model forward
            mixuped_logits = model(mixuped_x)  # [3*N,10]
            logits_x = mixuped_logits[:HP.batch_size]  # [N,10]
            logits_u = mixuped_logits[HP.batch_size:]  # [2*N,10]

            # eq. (2) - (5)
            loss_x, loss_u = criterion_train(logits_x, mixuped_out[:HP.batch_size], logits_u,
                                             mixuped_out[HP.batch_size:])
            loss = loss_x + lambda_rampup(step, max_v=HP.lambda_u) * loss_u  # eq. (5)

            logger.add_scalar('Loss/Train', loss, step)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model_ema_opt.step()

            if not step % HP.verbose_step:  # evaluation
                acc1, acc5, eval_loss = evaluate(model, val_loader, criterion_val)
                logger.add_scalar('Loss/Dev', eval_loss, step)
                logger.add_scalar('Acc1', acc1, step)
                logger.add_scalar('Acc5', acc5, step)

            if not step % HP.save_step:  # save model
                model_path = 'model_%d_%d.pth' % (epoch, step)
                save_checkpoint(model, ema_model, epoch, opt, os.path.join('./model_save', model_path))

            print('Epoch: [%d/%d],step:%d,Train Loss:%.5f,Dev Loss:%.5f,Acc1:%.3f,Acc5:%.3f' %
                  (epoch, HP.epochs, step, loss.item(), eval_loss, acc1, acc5))
            step += 1
            logger.flush()
    logger.close()


if __name__ == '__main__':
    train()
