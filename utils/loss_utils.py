#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
import itertools

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l1_loss_mask(network_output, gt, mask = None):
    if mask is None:
        return l1_loss(network_output, gt)
    else:
        return torch.abs((network_output - gt) * mask).sum() / mask.sum()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, mask=None, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if mask is not None:
        img1 = img1 * mask + (1 - mask)
        img2 = img2 * mask + (1 - mask)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
def tv(inp):
    """
    Computes the Total Variation (TV) loss for smoothness.
    Args:
        inp (torch.Tensor): Input tensor of shape (B, C, H, W) or (C, H, W).
    Returns:
        torch.Tensor: Total variation loss.
    """
    dx = inp[ :, :-1] - inp[:, 1:]
    dy = inp[ :-1, :] - inp[1:, :]
    loss_smooth_ren = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
    return loss_smooth_ren


def second_order_tv(ren_disparity, depth_mask_seg, gamma, num_mask_channels,seg_mask=None):
    """
    Computes a weighted combination of TV losses for disparity maps.
    Args:
        ren_disparity (torch.Tensor): Rendered disparity map of shape (C, H, W).
        depth_mask_seg (torch.Tensor): Depth mask segmentation of the same shape.
        gamma (float): Weighting factor between masked and unmasked TV loss.
        num_mask_channels (int): Number of mask channels to repeat.
        seg_mask: segmentation masks
    Returns:
        torch.Tensor: Second-order smoothness loss.
    """
    # Repeat the disparity map to match the number of mask channels
    # temp = depth_mask_seg * ren_disparity.repeat(num_mask_channels, 1, 1).contiguous()
    # Weighted sum of TV losses for masked and original disparity
    # loss_smooth_ren = gamma * tv(temp) + (1 - gamma) * tv(ren_disparity)
    loss_smooth_ren=0
    if seg_mask!=None:
        unique_regions = torch.unique(seg_mask)
        for region in unique_regions:
            mask = (seg_mask == region).float()
            temp = mask * ren_disparity
            loss_smooth_ren += gamma * tv(temp)
        loss_smooth_ren /= len(unique_regions)
    else:
        # temp = depth_mask_seg * ren_disparity.repeat(num_mask_channels, 1, 1).contiguous()
        # loss_smooth_ren = gamma * tv(temp) + (1 - gamma) * tv(ren_disparity)
        loss_smooth_ren =(gamma) * tv(ren_disparity)
    return loss_smooth_ren

class prototype_cl(torch.nn.Module):
    def __init__(self, num_classes=19, feat_dim=100, temperature=0.2, queue_len=10, device='cuda'):
        super().__init__()
        self.temperature = temperature
        self.queue_len = queue_len
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        # Class-wise prototype buffers, initialized on CPU, moved to device once
        self.register_buffer("queue", torch.randn(num_classes, queue_len, feat_dim))
        self.queue = self.queue.to(device)  # Move once

    def construct_region(self, fea, pred):
        pred = torch.argmax(pred, dim=0)  # [H,W]
        val = torch.unique(pred)

        fea = fea.view(self.feat_dim, -1)  # [feat_dim, H*W]
        pred = pred.view(-1)                # [H*W]

        # Use list comprehension + stack to avoid repeated cat calls
        new_fea = torch.stack([fea[:, pred == c].mean(1) for c in val], dim=0)
        return new_fea, val

    def info_nce_multi_positive(self, q, positives, negatives):
        # Normalize once inplace
        q = torch.nn.functional.normalize(q, dim=0)
        positives = torch.nn.functional.normalize(positives, dim=0)
        negatives = torch.nn.functional.normalize(negatives, dim=0)

        pos_logits = (q.T @ positives) / self.temperature  # [1, N_pos]
        neg_logits = (q.T @ negatives) / self.temperature  # [1, N_neg]

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        pos_weights = torch.ones_like(pos_logits) / pos_logits.size(1)
        targets = torch.cat([pos_weights, torch.zeros_like(neg_logits)], dim=1)

        log_probs = torch.log_softmax(logits, dim=1)
        loss = -torch.sum(targets * log_probs)
        return loss

    def contrastive_loss(self, feat, res, uid):
        keys, vals = self.construct_region(feat, res)  # keys: [N, feat_dim], vals: [N]
        keys = torch.nn.functional.normalize(keys, dim=1)
        device = keys.device

        contrast_loss = torch.tensor(0., device=device)

        if len(vals) == 0:
            return contrast_loss  # no valid classes

        # Pre-fetch all queues for present classes only, move to device once
        queues_for_vals = self.queue[vals].to(device)  # [N, queue_len, feat_dim]

        # For each class, compute multi-positive loss
        for i, cls_ind in enumerate(vals):
            query = keys[i].unsqueeze(1)  # [feat_dim, 1]
            l_pos = queues_for_vals[i].permute(1, 0)  # [feat_dim, queue_len]

            # Negatives: all other classes in queue
            neg_mask = torch.ones(self.num_classes, dtype=torch.bool, device=device)
            neg_mask[cls_ind] = False
            l_neg = self.queue[neg_mask].to(device).reshape(-1, self.feat_dim).permute(1, 0)

            contrast_loss += self.info_nce_multi_positive(query, l_pos, l_neg)

            # Update queue inplace: overwrite queue slot uid with key
            self.queue[cls_ind, uid] = keys[i].detach()

        return contrast_loss


import torch.nn.functional as F

class protoCL(torch.nn.Module):
    def __init__(self, num_classes=19, feat_dim=100, temperature=0.2, queue_len=10, device='cuda'):
        super().__init__()
        self.temperature = temperature
        self.queue_len = queue_len
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        # self.criterion = torch.nn.CrossEntropyLoss()

        self.register_buffer("queue", F.normalize(torch.randn(num_classes, queue_len, feat_dim), dim=-1))
        self.register_buffer("queue_ptr", torch.zeros(num_classes, dtype=torch.long))
        self.register_buffer("queue_filled", torch.zeros(num_classes, dtype=torch.long))

    def construct_region(self, fea, pred):
        pred = torch.argmax(pred, dim=0)
        val = torch.unique(pred)

        fea = fea.view(self.feat_dim, -1)
        pred = pred.view(-1)

        new_fea = torch.stack([fea[:, pred == c].mean(1) for c in val], dim=0)
        return new_fea, val

    def dequeue_and_enqueue(self, cls_idx, new_feat):
        ptr = self.queue_ptr[cls_idx].item()
        self.queue[cls_idx, ptr] = F.normalize(new_feat.detach(), dim=0)
        self.queue_ptr[cls_idx] = (ptr + 1) % self.queue_len
        if self.queue_filled[cls_idx] < self.queue_len:
            self.queue_filled[cls_idx] += 1

    def info_nce_multi_positive(self, query, positives, negatives):
        query = F.normalize(query, dim=0)
        positives = F.normalize(positives, dim=0)
        negatives = F.normalize(negatives, dim=0)

        pos_logits = (query.T @ positives) / self.temperature
        neg_logits = (query.T @ negatives) / self.temperature

        logits = torch.cat([pos_logits, neg_logits], dim=1)
        pos_weights = torch.ones_like(pos_logits) / pos_logits.size(1)
        targets = torch.cat([pos_weights, torch.zeros_like(neg_logits)], dim=1)

        log_probs = torch.log_softmax(logits, dim=1)
        # log_probs = self.criterion(logits, dim=1)
        loss = -torch.sum(targets * log_probs)
        return loss

    def contrastive_loss(self, feat, res, uid):
        keys, vals = self.construct_region(feat, res)
        keys = F.normalize(keys, dim=1)
        contrast_loss = torch.tensor(0.0, device=keys.device)

        for i, cls_idx in enumerate(vals):
            valid_len = self.queue_filled[cls_idx].item()
            if valid_len == 0:
                continue  # skip this class, no valid entries

            query = keys[i].unsqueeze(1)  # [feat_dim, 1]
            l_pos = self.queue[cls_idx, :valid_len].permute(1, 0)  # [feat_dim, valid_len]

            neg_mask = torch.ones(self.num_classes, dtype=torch.bool, device=self.device)
            neg_mask[cls_idx] = False
            valid_neg_mask = self.queue_filled > 0
            combined_mask = neg_mask & valid_neg_mask
            if combined_mask.sum() == 0:
                continue  # no valid negatives

            l_neg = self.queue[combined_mask].reshape(-1, self.feat_dim).permute(1, 0)

            contrast_loss += self.info_nce_multi_positive(query, l_pos, l_neg)
            self.dequeue_and_enqueue(cls_idx, keys[i])

        return contrast_loss
    
def smooth_3d_intra_class(classes, feat, opacity, neighbors):
    """
    Optimized intra-class feature smoothness loss.
    Vectorized to avoid Python for-loops.
    """
    N, K = neighbors.shape
    D = feat.shape[1]

    with torch.no_grad():
        # 1. Gather neighbor features and class labels
        neighbor_feats = feat[neighbors.tolist()]           # [N, K, D]
        neighbor_classes = classes[neighbors.tolist()]      # [N, K]
        center_feats = feat.unsqueeze(1)           # [N, 1, D]
        center_classes = classes.unsqueeze(1)      # [N, 1]

        same_class_mask = neighbor_classes == center_classes  # [N, K]

        sim = F.cosine_similarity(center_feats, neighbor_feats, dim=2)  # [N, K]

        sim = sim * same_class_mask.float()  # [N, K], similarities only for same-class neighbors

        # 5. Compute smoothness loss
        num_valid = same_class_mask.sum(dim=1).clamp(min=1)  # avoid div-by-zero
        smoothness = ((1 - sim).sum(dim=1)) / num_valid      # [N]

    return smoothness.mean()

@torch.no_grad()
def _mask_diagonal_(logits):
    N = logits.shape[0]
    logits.fill_diagonal_(-float('inf'))
    return logits

def smooth_3d_inter_class(labels, features, opacity, neighbors, temperature=0.3, k=3, chunk_size=2048):
    N, D = features.shape
    device = features.device
    features = F.normalize(features, dim=-1)

    losses = []

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)

        with torch.no_grad():
            anchor_feat = features[start:end]                  # (C, D)
            anchor_labels = labels[start:end].unsqueeze(1)     # (C, 1)

            # Similarity (C, N)
            sim = torch.matmul(anchor_feat, features.T) / temperature

            # Mask self-similarity (C, C)
            idx = torch.arange(start, end, device=device)
            sim[:, idx] = -float('inf')

            # Top-k selection
            topk_vals, topk_idx = torch.topk(sim, k=k, dim=1)
            topk_labels = labels[topk_idx]
            pos_mask = (topk_labels == anchor_labels)
            neg_mask = ~pos_mask

            exp_logits = torch.exp(topk_vals)
            pos_exp = exp_logits * pos_mask
            neg_exp = exp_logits * neg_mask

            pos_sum = pos_exp.sum(dim=1)
            neg_sum = neg_exp.sum(dim=1)
            valid_mask = (pos_sum > 0) & (neg_sum > 0)

        if valid_mask.any():
            loss = -torch.log(pos_sum[valid_mask] / (pos_sum[valid_mask] + neg_sum[valid_mask]))
            losses.append(loss)

    if not losses:
        return torch.tensor(0.0, device=device)

    return torch.cat(losses).mean()