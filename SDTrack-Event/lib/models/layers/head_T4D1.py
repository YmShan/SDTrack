import torch.nn as nn
import torch
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)
from lib.models.layers.frozen_bn import FrozenBatchNorm2d

from ...models.SDTrack.SDTrack_tiny_LIF_T4D1_model import LIF



class Conv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
        super().__init__()
        if freeze_bn:
            pass
        else:
            self.channel = out_planes
            self.lif = LIF()
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                          padding=padding, dilation=dilation, bias=True)
            self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        T, B, _, H, W = x.shape
        x = self.lif(x)            # [2, B, 320, 16, 16]
        x = self.conv(x.flatten(0, 1))
        x = self.bn(x).reshape(T, B, self.channel, H, W).contiguous()

        return x


class CenterPredictor(nn.Module, ):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(CenterPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        # corner predict
        self.conv1_ctr = Conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_ctr = Conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = Conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = Conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # size regress
        self.conv1_offset = Conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_offset = Conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = Conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = Conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = Conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_size = Conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = Conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = Conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x)

        # assert gt_score_map is None
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)

        return score_map_ctr, bbox, size_map, offset_map

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        T, B, _, H, W = x_ctr4.shape
        score_map_ctr = self.conv5_ctr(x_ctr4.flatten(0, 1)).reshape(T, B, 1, H, W).contiguous()

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        T2, B2, _, H2, W2 = x_offset4.shape
        score_map_offset = self.conv5_offset(x_offset4.flatten(0, 1)).reshape(T2, B2, 2, H2, W2).contiguous()

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        T3, B3, _, H3, W3 = x_size4.shape
        score_map_size = self.conv5_size(x_size4.flatten(0, 1)).reshape(T3, B3, 2, H3, W3).contiguous()

        return _sigmoid(score_map_ctr.mean(0)), _sigmoid(score_map_size.mean(0)), score_map_offset.mean(0)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build_box_head(cfg, hidden_dim):
    stride = cfg.MODEL.BACKBONE.STRIDE

    if cfg.MODEL.HEAD.TYPE == "MLP":
        mlp_head = MLP(hidden_dim, hidden_dim, 4, 3)  # dim_in, dim_hidden, dim_out, 3 layers
        return mlp_head
    elif "CORNER" in cfg.MODEL.HEAD.TYPE:
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        channel = getattr(cfg.MODEL, "NUM_CHANNELS", 256)
        print("head channel: %d" % channel)
        in_channel = hidden_dim
        out_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        if cfg.MODEL.HEAD.TYPE == "CORNER":
            corner_head = Corner_Predictor(inplanes=in_channel, channel=out_channel,
                                           feat_sz=feat_sz, stride=stride)
        else:
            raise ValueError()
        return corner_head
    elif cfg.MODEL.HEAD.TYPE == "CENTER":
        in_channel = hidden_dim
        out_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        center_head = CenterPredictor(inplanes=in_channel, channel=out_channel,
                                      feat_sz=feat_sz, stride=stride)
        return center_head
    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)
