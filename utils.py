import torch
import numpy as np


# make training more stable
class WeightEMA:
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.weight_decacy = 0.0004

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param) # 是把ema_param的产生copy 给param

    def step(self):
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:  # model weights only!
                ema_param.mul_(self.alpha)
                ema_param.add_(param * (1 - self.alpha))
                # apply weight
                param.mul_((1 - self.weight_decacy))


def lambda_rampup(step, MAX_STEP=1e6, max_v=75):
    """
    3.5 Hyperparameters: rampup
    :param step:  training step
    :param MAX_STEP: max step
    :param max_v: max value of lambda_u
    :return: current value of lambda_u
    """
    return np.clip(a=max_v * (step / MAX_STEP), a_min=0, a_max=max_v)


# label guessing = post distribution average + sharpen
def label_guessing(out_u, out_u2):
    """
    label guessing: eq. (6),K =2(default) as the paper said
    :param out_u: [N,10], model output(logits output)
    :param out_u2:[N,10]
    :return: average label guessing, [N,10]
    [[0.22,0.32......],   => sum =1
    [0.01,0.3,0.03...],   => sum = 1
    .....]
    """
    q = (torch.softmax(out_u, dim=-1) + torch.softmax(out_u2, dim=-1)) / 2.
    # algorithm 1,line 7
    return q


def sharpen(p, T):
    """
    sharpen: eq. (7), algorithm 1 line 8
    :param p: post disatribution: [N,10]
    [[0.22,0.32........], =>sum =1
    [0.01,0.3,0.03.....], =>sum =1
    ...]
    [0.1,0.1,0.1,.... 0.2,0.1]
    :param T: temperature
    :return: sharpened result
    """
    p_power = torch.pow(p, 1. / T)
    return p_power / torch.sum(p_power, dim=-1, keepdim=True)


def mixup(x, u, u2, trg_x, out_u, out_u2, alpha=0.75):
    """
    mixup: eq. (8)-(11), algorithm: Line12-Line14
    :param x: labeled x [N,3,H,W]
    :param u: the first unlabeled data,[N,3,H,W]
    :param u2: the second unlabeled data,[N,3,H,W]
    :param trg_x: labeled x target(y),[N, ]=[0,7,8...]
    :param out_u:q_b,after lable guessing
    :param out_u2: q_b
    :param alpha: Beta hype
    :return: Mixuped result: x: [3*N,3,H,W], y:[3*N,10]
    """
    batch_size = x.size(0)  # batch size = HP.batch_size
    n_classes = out_u.size(1)  # classes number: 10
    device = x.device
    # [0.1,0.3,0.01.....] dim=10
    # [0.,0.,0.,0.,0.,0.,0.,0.,1.,0.] dim = 10
    # target x batch to onehot
    trg_x_onehot = torch.zeros(size=(batch_size, n_classes)).float().to(device)
    # [0,0,0,0,0,0,0,0,0,0]
    # trg_x [7]
    # [0,0,0,0,0,0,0,1,0,0]
    trg_x_onehot.scatter_(1, trg_x.view(-1, 1), 1.)

    # mixup
    x_cat = torch.cat([x, u, u2], dim=0)
    trg_cat = torch.cat([trg_x_onehot, out_u, out_u2], dim=0)
    n_item = x_cat.size(0)  # N*3

    lam = np.random.beta(alpha, alpha)  # eq.(8)
    lam_prime = max(lam, 1 - lam)  # eq.(9)
    rand_idx = torch.randperm(n_item)  # a rand index sequence: [0,2,1],[1,0,2].
    x_cat_shuffled = x_cat[rand_idx]  # x2
    trg_cat_shuffled = trg_cat[rand_idx]

    x_cat_mixup = lam_prime * x_cat + (1 - lam_prime) * x_cat_shuffled  # eq. (9)
    trg_cat_mixup = lam_prime * trg_cat + (1 - lam_prime) * trg_cat_shuffled  # eq.(10)

    return x_cat_mixup, trg_cat_mixup


def accuracy(output, target, topk=(1,)):
    """
    topk acc
    :param output: [N,10]
    :param target: [N,]
    :param topk: top1,top3,top5
    :return: acc list
    """
    maxk = max(topk)  # max k , topk=(1,3,5)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()  # [maxk,N]
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100. / batch_size))
    return res  # [50,85,99]
