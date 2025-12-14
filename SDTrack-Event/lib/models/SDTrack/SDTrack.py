"""
Basic SDTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head

from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.SDTrack.SDTrack_tiny_model import SDTrack_tiny as tiny
from lib.models.SDTrack.SDTrack_base_model import SDTrack_base as base
from lib.train.admin import env_settings  # 导入环境设置

class SDTrack(nn.Module):
    """ This is the base class for SDTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,

                ):
        x, aux_dict = self.backbone(z=template, x=search)

        # Forward head
        feat_last = x               # [1, 768, 320]
        if isinstance(x, list):
            feat_last = x[-1]
        feat_last = feat_last.permute(0, 2, 1)
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]         # [B, 256, 768]
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)                  # [B, 320, 16, 16]

        if self.head_type == "CORNER":
            # run the corner head
            pred_box= self.box_head(opt_feat)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


# 计算模型的参数量
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params


def build_SDTrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('SDTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''


    if cfg.MODEL.BACKBONE.TYPE == 'tiny':
        backbone = tiny()

        hidden_dim = 360            # lite 360    base 768
        patch_start_index = 1
        checkpoint = torch.load(env_settings().pretrained_networks + '/SDTrack-tiny-1x4.pth', map_location="cpu")

    elif cfg.MODEL.BACKBONE.TYPE == 'base':
        backbone = base()

        hidden_dim = 768            # lite 360    base 768
        patch_start_index = 1

        checkpoint = torch.load(env_settings().pretrained_networks + '/SDTrack-base-1x4.pth', map_location="cpu")

    else:
        raise NotImplementedError

    # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = SDTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    # print(model)

    missing_keys, unexpected_keys = model.backbone.load_state_dict(checkpoint["model"], strict=False)
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print('missing keys:')
    print(missing_keys)
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    print('unexpected keys:')
    print(unexpected_keys)
    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

    # 输出模型的总参数量
    print(f"模型的总参数量: {count_parameters(model)}")
    # print(model)


    return model

