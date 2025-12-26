import random

import torch.nn as nn
from mmcv.cnn.bricks import PLUGIN_LAYERS
import torch
import numpy as np
import copy
from scipy.ndimage import zoom
import torch.nn.functional as F
import cv2



class SparsePrompter_uncertainty(nn.Module):
    def __init__(self, shape=(540, 960), sparse_rate=0.025):
        super().__init__()
        self.shape_h, self.shape_w = shape
        self.ratio = sparse_rate
        self.pnum = int(self.shape_h * self.shape_w * sparse_rate)
        self.patch = nn.Parameter(torch.randn(1, 320, 1, 1))  # For input image
        self.feature_prompts = nn.ParameterList([
            nn.Parameter(torch.randn(self.pnum, C, 1, 1)) for C in [320]
        ])
        self.if_mask = True
        self.uncmap = None

    def forward_feature_prompt(self, x, uncertainty_maps):
        """Apply prompts at high-uncertainty locations in bottleneck feature map."""
        prompted_features = []
        prompt_idx = 0

        for i, feat in enumerate(x):
            if feat.shape[1] == 320 and prompt_idx < len(self.feature_prompts):
                feat = feat.clone()
                device = feat.device
                resized_prompt = F.interpolate(
                    self.feature_prompts[prompt_idx], size=feat.shape[2:], mode='bilinear', align_corners=False
                ).to(device)

                H, W = feat.shape[2], feat.shape[3]
                try:
                    unc_map = uncertainty_maps[i].detach().squeeze().cpu().numpy()
                    topk = unc_map.flatten().argsort()[::-1][:self.pnum]
                    coords = [(idx // W, idx % W) for idx in topk]

                    for j, (h, w) in enumerate(coords):
                        if h < H and w < W:
                            feat[:, :, h, w] += resized_prompt[j % self.pnum, :, 0, 0]
                except Exception as e:
                    print(f"[Prompt] Warning: Skipping prompt injection due to error: {e}")

                prompted_features.append(feat)  # ✅ 放到 try-except 外面

            else:
                prompted_features.append(feat)  # 保证没注入 prompt 的层也保留

        # print("[Debug] self.pnum =", self.pnum)
        # print("[Debug] feature_prompts shape:", self.feature_prompts[prompt_idx].shape)
        # print(f"[Debug] Injected prompt on {len(coords)} coords, max val in feat: {feat.max().item()}")

        return prompted_features

    # def update_uncmap(self, uncmap):
    #     if isinstance(uncmap, torch.Tensor):
    #         self.uncmap = uncmap.detach().cpu().squeeze().numpy()
    #     elif isinstance(uncmap, np.ndarray):
    #         self.uncmap = uncmap
    #     else:
    #         raise TypeError("Uncertainty map must be a tensor or ndarray")
    #     self.prompt_mask = uncmap.to(self.patch.device)  # 更新 prompt_mask

    def update_uncmap(self, uncmap):
        """Update internal uncertainty map and generate prompt mask tensor."""
        if isinstance(uncmap, np.ndarray):
            uncmap = torch.from_numpy(uncmap)  # 转换成 Tensor
        if uncmap.dim() == 2:
            uncmap = uncmap.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif uncmap.dim() == 3:
            uncmap = uncmap.unsqueeze(0)  # [1, 1, H, W]
        uncmap = uncmap.float().to(self.patch.device)  # 保证类型正确且在同一设备
        self.uncmap = uncmap
        self.prompt_mask = uncmap


    def downsample_map(self, target_shape):
        zoom_factors = [t / o for t, o in zip(target_shape, self.uncmap.shape)]
        return zoom(self.uncmap, zoom_factors, order=1)

    # def select_position(self, shape, num_points):
    #     h, w = shape
    #     coords = [(i, j) for i in range(h) for j in range(w)]
    #     return random.sample(coords, num_points)

    def select_position(self, shape, k):
        h, w = map(int, shape)
        if h == 0 or w == 0:
            # print(f"[Prompt] ❗ Warning: Invalid resize shape h={h}, w={w}. Skipping prompt injection.")
            return []

        unc_np = self.uncmap
        if isinstance(unc_np, torch.Tensor):
            unc_np = unc_np.detach().cpu().numpy()

        if unc_np.ndim == 4:
            unc_np = unc_np.squeeze()  # from (1,1,H,W) -> (H,W)

        try:
            resized_unc = cv2.resize(unc_np, (w, h), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            # print(f"[Prompt] ❗ cv2.resize error: {e}, uncmap shape: {unc_np.shape}, target: (w={w}, h={h})")
            return []

        flat = resized_unc.flatten()
        topk_indices = flat.argsort()[::-1][:k]
        coords = [(idx // w, idx % w) for idx in topk_indices]
        # print(f"[Debug] Prompt resize target shape: (h={h}, w={w})")
        return coords

    def get_prompt_mask(self, shape):
        h, w = shape
        dynamic_pnum = max(1, int(h * w * self.ratio))
        pos = self.select_position((h, w), dynamic_pnum)

        # print(">> dynamic_pnum:", dynamic_pnum)
        # print(">> selected:", len(pos))

        mask = torch.zeros((1, 1, h, w), device=self.patch.device)
        for hi, wi in pos:
            if hi < h and wi < w:
                mask[0, 0, hi, wi] = 1.0
        # print(f"[DEBUG] prompt_mask shape: {self.prompt_mask.shape}, sum: {self.prompt_mask.sum().item()}")
        # print(f"[Prompt Debug] sparse_rate={self.ratio}, h={h}, w={w}, mask_pixels={dynamic_pnum}")
        return mask

    # def get_masked_prompt(self, shape):
    #     h, w = shape
    #
    #     dynamic_pnum = max(1, int(h * w * self.ratio))
    #     pos = self.select_position((h, w), dynamic_pnum)
    #
    #     # prompt = torch.zeros((512, h, w), device=self.patch.device)
    #     prompt = torch.zeros_like(self.prompt_mask.expand(1, 512, h, w))  # 用广播替代
    #     patch_vector = self.patch[0].squeeze()
    #     for i, (hi, wi) in enumerate(pos):
    #         if i >= dynamic_pnum:
    #             break
    #         prompt[:, hi, wi] = patch_vector
    #     return prompt
    def get_masked_prompt(self, shape):
        h, w = shape
        dynamic_pnum = max(1, int(h * w * self.ratio))
        pos = self.select_position((h, w), dynamic_pnum)

        # Resize mask to current feature map shape
        resized_mask = F.interpolate(self.prompt_mask, size=(h, w), mode='nearest')  # [1, 1, H, W]

        # Create prompt feature map
        prompt = torch.zeros_like(resized_mask.expand(-1, 320, -1, -1))  # [1, 512, H, W]

        # Broadcast patch_vector to each selected location
        patch_vector = self.patch.squeeze()  # shape: [512]

        for i, (hi, wi) in enumerate(pos):
            if i >= dynamic_pnum:
                break
            if hi >= h or wi >= w:
                # print(f"[Prompt Debug] Skipping invalid coord: hi={hi}, wi={wi} for shape=({h}, {w})")
                continue
            prompt[0, :, hi, wi] = patch_vector  # ✅ fixed line

        return prompt

    # def update_mask(self):
    #     self.prompt_lst = []
    #     scales = [
    #         (270, 480), (405, 720), (540, 960),
    #         (675, 1200), (810, 1440), (945, 1680), (1080, 1920)
    #     ]
    #     for i, shape in enumerate(scales * 2):  # include flipped variants
    #         mask = self.get_masked_prompt(shape)
    #         if i % 2 == 1:
    #             mask = torch.flip(mask, dims=[2])
    #         self.prompt_lst.append(mask)
    # def update_mask(self, shape):
    #     """
    #     shape: Tuple[int, int], e.g., (H, W) from current bottleneck feature
    #     """
    #     self.prompt_mask = self.get_masked_prompt(shape)

    def update_mask(self, mask):
        if isinstance(mask, np.ndarray):
            h, w = mask.shape[-2:]
        elif isinstance(mask, torch.Tensor):
            h, w = mask.size()[-2:]
        else:
            raise TypeError("Unsupported mask type")
        self.prompt_mask = self.get_masked_prompt((h, w))

    def forward(self, x, img_metas=None, position=None):
        """Apply prompt to the input image (used for low-level prompting)."""
        # if not self.if_mask:
        #     return x
        #
        # if self.patch.dim() == 3:
        #     self.patch = self.patch.unsqueeze(0)
        #
        # prompt_data = F.interpolate(self.patch, size=x.shape[2:], mode='bilinear', align_corners=False)
        # return x + prompt_data

        return x

