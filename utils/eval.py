from __future__ import print_function, absolute_import
import torch

__all__ = ['accuracy']

def accuracy(output, target, topk=(1,),target_mix=None):
    """Computes the precision@k for the specified values of k"""
    _, p_1 =output.topk(1)
    if target_mix is not None:
        target = target_mix.topk(1)[1].view(-1)
    #miss_base = ((p_1.t()<100)*(p_1.t()!=target)).sum()
    #miss_novel = ((p_1.t()>99)*(p_1.t()!=target)).sum()
    #miss_all = (p_1.t()!=target).sum()
    #if miss_all==0:
    #    miss_base = torch.ones(1)[0].cuda()/2.
    #    miss_novel = torch.ones(1)[0].cuda()/2.
    #    miss_all = torch.ones(1)[0].cuda()

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    #res.extend([miss_base.float()/miss_all.float(),miss_novel.float()/miss_all.float()])
    return res
