import os.path as osp
import pickle
import shutil
import tempfile
import datetime

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmcv.runner import build_optimizer, build_runner

from IPython import embed
from mmseg.ops import resize

import copy
import kornia
import random
import torch.nn as nn
from mmseg.models.backbones.prompt import *
from mmseg.models.backbones.lora import *
# from mmseg.apis.test import np2tmp

import torch.nn.functional as F
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)



def update_ema_variables_cotta(ema_model, model, alpha_teacher, iteration=None):
    # Use the "true" average until the exponential average is more correct
    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    if True:
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model

def update_ema_variables(ema_model, model, alpha_model, alpha_prompt, iteration=None):
    # Use the "true" average until the exponential average is more correct
    if iteration:
        alpha_teacher = min(1 - 1 / (iteration + 1), alpha_teacher)

    if True:
        for ema_param, (name, param) in zip(ema_model.parameters(), model.named_parameters()):
            #ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
            if "prompt" in name:
                ema_param.data[:] = alpha_prompt * ema_param[:].data[:] + (1 - alpha_prompt) * param[:].data[:]
            else:
                ema_param.data[:] = alpha_model * ema_param[:].data[:] + (1 - alpha_model) * param[:].data[:]
    return ema_model

def np2tmp(array, temp_file_name=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.

    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False).name
    np.save(temp_file_name, array)
    return temp_file_name

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def entropy_loss(logits):
    probs = torch.softmax(logits, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1).mean()

def bottleneck_entropy(features):
    """
    Computes per-pixel entropy across channel activations at the bottleneck.
    Args:
        features (torch.Tensor): shape [B, C, H, W]
    Returns:
        torch.Tensor: scalar entropy loss
    """
    # Normalize features across channels (softmax-like)
    probs = torch.softmax(features, dim=1)  # shape: [B, C, H, W]
    log_probs = torch.log_softmax(features, dim=1)
    entropy_map = -(probs * log_probs).sum(dim=1)  # shape: [B, H, W]
    return entropy_map.mean()


def prompt_entropy_loss(features, prompt_mask=None):
    """
    Computes entropy loss over the feature map, optionally masked by a prompt region.

    Args:
        features (torch.Tensor): shape [B, C, H, W]
        prompt_mask (torch.Tensor, optional): shape [B, 1, H, W] with 0/1 indicating where prompt is applied

    Returns:
        torch.Tensor: scalar entropy loss
    """
    probs = F.softmax(features, dim=1)
    log_probs = F.log_softmax(features, dim=1)
    entropy = -(probs * log_probs).sum(dim=1)  # shape [B, H, W]

    if prompt_mask is not None:
        prompt_mask = prompt_mask.float()
        if prompt_mask.shape != entropy.shape:
            prompt_mask = F.interpolate(prompt_mask, size=entropy.shape[-2:], mode='nearest')
        entropy = entropy * prompt_mask.squeeze(1)  # Apply mask

        # Avoid division by zero
        valid_count = prompt_mask.sum()
        if valid_count > 0:
            return entropy.sum() / valid_count
        else:
            return torch.tensor(0.0, device=features.device)
    else:
        return entropy.mean()


import torch.nn.functional as F


class StageConsistencyLoss(nn.Module):
    def __init__(self, weight=0.1, align_method='upsample'):
        """
        Args:
            weight (float): Loss weight.
            align_method (str): 'upsample' (Stage 3 → Stage 2) or 'downsample' (Stage 2 → Stage 3).
        """
        super().__init__()
        self.weight = weight
        self.align_method = align_method
        self.mse_loss = nn.MSELoss()

    def forward(self, feat_s2, feat_s3):
        """
        Args:
            feat_s2: Stage 2 features [B, C2, H/8, W/8]
            feat_s3: Stage 3 features [B, C3, H/16, W/16]
        Returns:
            loss: Consistency loss term.
        """
        # Step 1: Align spatial resolutions
        if self.align_method == 'upsample':
            # Upsample Stage 3 to match Stage 2
            feat_s3_aligned = F.interpolate(feat_s3, size=feat_s2.shape[2:], mode='bilinear', align_corners=False)
            target = feat_s3_aligned
            source = feat_s2
        elif self.align_method == 'downsample':
            # Downsample Stage 2 to match Stage 3
            feat_s2_aligned = F.interpolate(feat_s2, size=feat_s3.shape[2:], mode='bilinear', align_corners=False)
            target = feat_s2_aligned
            source = feat_s3
        else:
            raise ValueError("align_method must be 'upsample' or 'downsample'")

        # Step 2: Align channel dimensions (if needed)
        if source.shape[1] != target.shape[1]:
            # Project source to target's channel dim using 1x1 conv
            channel_adjust = nn.Conv2d(source.shape[1], target.shape[1], kernel_size=1).to(source.device)
            source = channel_adjust(source)

        # Step 3: Compute MSE loss
        loss = self.mse_loss(source, target)

        return self.weight * loss

# Inside test.py or tools/GaPaTTA.py
# Add at top
import torch
import torch.nn.functional as F
import numpy as np
import math
import torch.nn as nn

class FeatureBuffer:
    def __init__(self, max_len=10):
        self.buffer = []
        self.max_len = max_len

    def add(self, feat):
        if len(self.buffer) >= self.max_len:
            self.buffer.pop(0)
        self.buffer.append(feat.detach().cpu())

    def get_tensor(self):
        return torch.cat(self.buffer, dim=0)  # [N, C]

    def __len__(self):
        return len(self.buffer)

class LaplaceLastLayer:
    def __init__(self, prior_precision=1.0, temperature=1.0):
        self.mean = None
        self.cov_diag = None
        self.prior_precision = prior_precision
        self.temperature = temperature

    def fit(self, feat_buffer: FeatureBuffer):
        features = feat_buffer.get_tensor()
        self.mean = features.mean(dim=0)
        centered = features - self.mean
        self.cov_diag = centered.pow(2).mean(dim=0) + 1.0 / self.prior_precision

    def predict_uncertainty(self, feat):
        centered = feat - self.mean.cuda()
        var = centered.pow(2) / self.cov_diag.cuda()
        predictive_variance = var.sum(dim=1) * self.temperature
        entropy = 0.5 * torch.log(2 * torch.pi * torch.exp(torch.tensor(1.0)).cuda() * predictive_variance)
        return entropy  # [1]
def update_optimizer_lr(optimizer, domain_idx):
    prompt_lr_base = 0.0001
    model_lr_base = 0.0001
    decay = 1 ** domain_idx

    new_prompt_lr = prompt_lr_base * decay
    new_model_lr = model_lr_base * decay

    optimizer.param_groups[0]['lr'] = new_prompt_lr
    optimizer.param_groups[1]['lr'] = new_model_lr

    print(f"[✓] Domain {domain_idx}: prompt_lr={new_prompt_lr:.6f}, stage2_lr={new_model_lr:.6f}")

def single_gpu_gapatta(args,
                    model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    anchor=None,
                    ema_model=None,
                    anchor_model=None,
                    dynamic_ema=False,
                    dynamic_prompt_only=False,
                    domain_idx=0):

    anchor_model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    total_iters = len(data_loader)
    switch_iter = int(total_iters * 0.8)
    max_iter = total_iters

    param_prompt_list = []
    param_stage2_list = []

    for name, param in model.named_parameters():
        name_lower = name.lower()

        # # 启用 prompt 参数训练
        if "prompt" in name:
            param.requires_grad = True
            param_prompt_list.append(param)
            # print(f"[Prompt] Unfrozen: {name}")

        # 选择性微调 attention 参数
        # elif "attn" in name_lower or "attention" in name_lower:
        #     param.requires_grad = True
        #     param_stage2_list.append(param)
        #     # print(f"[Attn] Unfrozen: {name}")

        else:
            # 其余默认冻结
            param.requires_grad = True
            param_stage2_list.append(param)
            # print(f"[Frozen] {name}")

    # fine-tune attn in block 3 only.
    # for name, param in model.named_parameters():
    #     if "prompt" in name:
    #         param.requires_grad = True
    #         param_prompt_list.append(param)
    #     elif 'block3' in name and 'attn' in name:
    #         param.requires_grad = True
    #         param_stage2_list.append(param)
    #         # print(f"[✓] Finetune: {name}")
    #     else:
    #         param.requires_grad = False




    optimizer = torch.optim.Adam([
        {"params": param_prompt_list, "lr": args.prompt_lr},
        {"params": param_stage2_list, "lr": args.model_lr}
    ], lr=1e-5, betas=(0.9, 0.999))

    update_optimizer_lr(optimizer, domain_idx)

    for i, data in enumerate(data_loader):
        if len(param_prompt_list) == 0 and i < switch_iter:
            print(f"[Iter {i}] Skipping warmup because no prompt parameters found.")
            continue

        model.eval()
        ema_model.eval()

        img_id = 4 if len(data['img']) == 14 else 0
        img = data['img'][img_id].cuda()
        img_metas = data['img_metas'][img_id].data[0]

        # === Pixel-wise entropy-inspired uncertainty estimation (Gaussian entropy form) ===
        # ------------------------------------------------------------------
        # Pixel-wise uncertainty estimation for local prompt placement.
        # We compute an entropy-inspired uncertainty score by measuring
        # channel-wise deviations from global channel statistics and mapping
        # them to a Gaussian entropy form. This score is used only for
        # ranking pixels (top-rho selection), not for probabilistic modeling.
        # ------------------------------------------------------------------

        with torch.no_grad():
            features = ema_model.module.extract_feat(img, img_metas)
            s2_feat = features[2]
            feat_np = s2_feat.squeeze(0).cpu().numpy()
            mean_feat = np.mean(feat_np, axis=(1, 2), keepdims=True)
            var_map = np.mean((feat_np - mean_feat) ** 2, axis=0)
            entropy_map = 0.5 * np.log(2 * np.pi * np.e) + 0.5 * np.log(var_map + 1e-6)

            model.module.backbone.prompt.if_mask = True
            model.module.backbone.prompt.update_uncmap(entropy_map)

            threshold = np.quantile(entropy_map, 0.5)
            mask = (entropy_map >= threshold).astype(np.float32)
            model.module.backbone.prompt.update_mask(mask)

        # === confidence calibration + pseudo label ===
        with torch.no_grad():
            result, probs, preds = ema_model(return_loss=False, **data)

            # === Step 1: Convert probs ===
            probs_tensor = probs[0]  # numpy or torch

            if isinstance(probs_tensor, np.ndarray):
                probs_tensor = torch.from_numpy(probs_tensor).cuda()

            if probs_tensor.dim() == 3:
                probs_tensor = probs_tensor.unsqueeze(0)  # [1, C, H, W]

            # print(f"[Debug] probs_tensor shape: {probs_tensor.shape}")  # [1, C, H, W]

            # === Step 2: Entropy ===
            probs_tensor = probs_tensor.permute(0, 2, 3, 1)  # [1, H, W, C]
            entropy = (-probs_tensor * (probs_tensor + 1e-8).log()).sum(dim=-1)  # [1, H, W]
            # print(f"[Debug] entropy shape: {entropy.shape}")

            # === Step 3: Pseudo label ===
            hard_pseudo = torch.from_numpy(preds[4]).long().cuda()  # [H, W] or [1, H, W]
            if hard_pseudo.dim() == 3:
                target = hard_pseudo.clone()  # Already [1, H, W]
            else:
                target = hard_pseudo.unsqueeze(0).clone()  # Convert to [1, H, W]

            # === Step 4: Sanity check ===
            assert entropy.shape == target.shape, f"Entropy {entropy.shape} vs Target {target.shape}"

            # === Step 5: Mask + assign ===
            mask = entropy >= args.entropy_thresh  # [1, H, W]
            target[mask] = 255
            target = target.unsqueeze(1).long()  # [1, 1, H, W]

        # === forward loss ===
        loss_dict = model(return_loss=True, img=img, img_metas=img_metas, gt_semantic_seg=target)
        ce_loss = loss_dict["decode.loss_seg"].mean()

        features = model.module.extract_feat(img, img_metas)
        s2_feat = features[2]
        s3_feat = features[3]

        s3_feat = F.interpolate(s3_feat, size=s2_feat.shape[-2:], mode='bilinear', align_corners=False)
        mapper = nn.Conv2d(s3_feat.shape[1], s2_feat.shape[1], kernel_size=1).to(s2_feat.device)
        s3_feat = mapper(s3_feat)
        feat_consistency = F.mse_loss(s2_feat, s3_feat.detach())

        prompt_mask = model.module.backbone.prompt.get_prompt_mask(s2_feat.shape[-2:])
        mask_np = prompt_mask.squeeze().detach().cpu().numpy()
        import matplotlib.pyplot as plt

        # print(f"[DEBUG] Prompt mask sum: {prompt_mask.sum().item()}")
        plt.imsave(f"vis/mask_iter{i}.png", mask_np, cmap='gray')
        entropy_loss = prompt_entropy_loss(features[-1], prompt_mask)

        with torch.no_grad():
            feat_anchor = anchor_model.module.extract_feat(img, img_metas)[2]
        loss_feature_align = F.mse_loss(s2_feat, feat_anchor)


        total_loss = (
            ce_loss
            + args.prompt_weight * entropy_loss
            + args.consistency_weight * feat_consistency
            + args.align_weight * loss_feature_align
        )

        optimizer.zero_grad()
        if total_loss.requires_grad:
            total_loss.backward()
            optimizer.step()

        prompt_rate = args.ema_rate - np.average(entropy_map) * args.scale
        ema_model = update_ema_variables(ema_model, model,
                                         alpha_model=args.ema_rate,
                                         alpha_prompt=prompt_rate)

        # for nm, m in model.named_modules():
        #     for npp, p in m.named_parameters():
        #         if npp in ['weight', 'bias'] and p.requires_grad:
        #             mask = (torch.rand(p.shape) < 0.01).float().cuda()
        #             with torch.no_grad():
        #                 p.data = anchor[f"{nm}.{npp}"] * mask + p * (1. - mask)

        if isinstance(result, list):
            results.extend([np2tmp(_) if efficient_test else _ for _ in result])
        else:
            results.append(np2tmp(result) if efficient_test else result)

        for _ in range(data['img'][0].size(0)):
            prog_bar.update()

    return results

def single_gpu_cotta(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    anchor=None,
                    ema_model=None,
                    anchor_model=None):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    anchor_model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    param_list = []
    out_dir = "./cotta/"+str(datetime.datetime.now())
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_list.append(param)
            print(name)
        else:
            param.requires_grad=False
    optimizer = torch.optim.Adam(param_list, lr=0.00006/8, betas=(0.9, 0.999))#Batchsize=1 now, was 8 during cityscapes training
    for i, data in enumerate(data_loader):
        model.eval()
        ema_model.eval()
        anchor_model.eval()
        with torch.no_grad():
            result, probs, preds = ema_model(return_loss=False, **data)
            _, probs_, _ = anchor_model(return_loss=False, **data)
            mask = (probs_[4][0] > 0.69).astype(np.int64) # 0.74 was the 5% quantile for cityscapes, therefore we use 0.69 here
            result = [(mask*preds[4][0] + (1.-mask)*result[0]).astype(np.int64)]
            weight = 1.
        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)
        if isinstance(result, list):
            if len(data['img'])==14:
                img_id = 4 #The default size without flip
            else:
                img_id = 0
            loss = model.forward(return_loss=True, img=data['img'][img_id], img_metas=data['img_metas'][img_id].data[0], gt_semantic_seg=torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(0))
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        torch.mean(weight*loss["decode.loss_seg"]).backward()
        optimizer.step()
        optimizer.zero_grad()
        ema_model = update_ema_variables_cotta(ema_model = ema_model, model = model, alpha_teacher=args.ema_rate)
        for nm, m  in model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape)<0.01).float().cuda()
                    with torch.no_grad():
                        p.data = anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def single_gpu_tent(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    print('-------------model:::')
    print(model)
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    param_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.requires_grad and ("norm" in name or "bn" in name):
                param_list.append(param)
                print(name)
            else:
                param.requires_grad=False

    optimizer = torch.optim.Adam(param_list, lr=0.00006/8, betas=(0.9, 0.999))
    for i, data in enumerate(data_loader):

        with torch.no_grad():
            result = model(return_loss=False, **data)


        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)
            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)

        if isinstance(result, list):
            loss = model.forward(return_loss=True, img=data['img'][0], img_metas=data['img_metas'][0].data[0], gt_semantic_seg=torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(0))
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            loss = model(return_loss=True, img=data['img'][0], img_metas=data['img_metas'][0].data[0], gt_semantic_seg=result)
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        torch.mean(loss["decode.loss_seg"]).backward()
        optimizer.step()
        optimizer.zero_grad()

        batch_size = data['img'][0].size(0)
        if i ==999:
            return results
        for _ in range(batch_size):
            prog_bar.update()
    return results



def single_gpu_test(args,
                    model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False):
    """Test with single GPU.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during infernece. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    out_dir = "./baseline/"+str(datetime.datetime.now())
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, **data)

        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Default: False.

    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if isinstance(result, list):
            if efficient_test:
                result = [np2tmp(_) for _ in result]
            results.extend(result)
        else:
            if efficient_test:
                result = np2tmp(result)
            results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results with CPU."""
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results with GPU."""
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results

import time
import cv2
def draw_map(x):
    tic=time.time()
    name = str(tic)
    H=1080 // 2
    W=1920 // 2
    x_visualize = x
    print('ssssssssssss',x.shape)

    x_visualize =  x_visualize[0]
    x_visualize = (((x_visualize - np.min(x_visualize))/(np.max(x_visualize)-np.min(x_visualize)))*255).astype(np.uint8) #归一化并映射到0-255的整数，方便伪彩色化
    savedir = './cotta/uncer/'
    import os
    if not os.path.exists(savedir+'dense'):
        os.mkdir(savedir+'dense')
    x_visualize = cv2.applyColorMap(x_visualize, cv2.COLORMAP_WINTER)  # 伪彩色处理
    cv2.imwrite(savedir+'dense/'+name+'.jpg',x_visualize) #保存可视化图像
