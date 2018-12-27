# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
import math


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, rois):

        # Compute level ids
        #s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))
        levels = []
        for roi in rois:
            # Eqn.(1) in FPN paper
            s = math.sqrt((roi[:, 3] - roi[:, 1] + 1) * (roi[:, 4] - roi[:, 2] + 1))
            level = math.floor(self.lvl0 + math.log2(s / self.s0 + self.eps))
            level = max(min(level, self.k_max), self.k_min)
            levels.append(level - self.k_min)

        return torch.LongTensor(levels)


class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[float]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            poolers.append(
                RoIAlignFunction(
                    output_size[0], output_size[1], spatial_scale=scale
                )
            )
        #self.poolers = nn.ModuleList(poolers)
        self.poolers = poolers
        self.output_size = output_size
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -math.log2(scales[0])  # 2
        lvl_max = -math.log2(scales[-1])  # 5
        self.map_levels = LevelMapper(lvl_min, lvl_max)


    def forward(self, x, rois):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        if num_levels == 1:
            return self.poolers[0](x[0], rois)

        levels = self.map_levels(rois)

        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        result = torch.zeros((num_rois, num_channels, output_size, output_size))

        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level)

        return result
