import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skvideo.measure import niqe
import torch


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(1.0 / batch_size))

    return res


def compute_psnr_ssim(restored, clean):
    restored = restored.detach().cpu().numpy()
    clean = clean.detach().cpu().numpy()

    psnr = 0
    ssim = 0
    psnr_list = []

    for i in range(restored.shape[0]):
        # compute psnr
        psnr_value = peak_signal_noise_ratio(
            clean[i], restored[i], data_range=1)
        psnr += psnr_value
        psnr_list.append(psnr_value)

        # compute ssim
        h, w = clean[i].shape[1:]
        win_size = min(7, min(h, w))
        if win_size % 2 == 0:
            win_size -= 1
        if win_size < 3:
            win_size = 3

        ssim_value = structural_similarity(
            clean[i],
            restored[i],
            data_range=1,
            channel_axis=0,
            win_size=win_size
        )
        ssim += ssim_value

    psnr /= restored.shape[0]
    ssim /= restored.shape[0]

    return psnr, ssim, psnr_list


def compute_niqe(image):
    image = np.clip(image.detach().cpu().numpy(), 0, 1)
    image = image.transpose(0, 2, 3, 1)
    niqe_val = niqe(image)

    return niqe_val.mean()


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0
